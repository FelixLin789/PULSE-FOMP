"""Paired, resumable FineWeb-Edu runs. Counts next-token targets with no wrapping."""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import os
import random
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

import review_v53 as audit
from pulse.experiment import FairMuonAdamW, _cosine_lr_multiplier

DATASET = "HuggingFaceFW/fineweb-edu"
REVISION = "1db02b5ece9677c879ceb35ea7da8065ec58f831"
SOURCE_HASH = "0c4258fa39670f2c0df7ce48ddcae3039b33068358f4d88fddb41f7ea7e3c6c9"


class FineWeb:
    def __init__(self, directory):
        import tiktoken
        from huggingface_hub import HfApi
        self.root=Path(directory)
        self.root.mkdir(exist_ok=True,parents=True)
        self.enc=tiktoken.get_encoding("gpt2")
        if self.enc.n_vocab!=50257 or self.enc.eot_token!=50256:
            raise ValueError("Unexpected tokenizer")
        api=HfApi()
        self.paths=sorted(f.path for f in api.list_repo_tree(DATASET,"sample/10BT",repo_type="dataset",
                                                              revision=REVISION) if f.path.endswith(".parquet"))
        if not self.paths: raise RuntimeError("No pinned FineWeb-Edu sample-10BT files")
        self.records=[]
        self.maps={"train":[],"val":[]}
        self.ends={"train":[],"val":[]}

    def add_shard(self):
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download
        i=len(self.records)
        if i==len(self.paths): raise RuntimeError("Pinned corpus exhausted")
        uri=self.paths[i]
        manifest=self.root/f"{i:04d}.json"
        files={s:self.root/f"{i:04d}.{s}.u16" for s in self.maps}
        if not manifest.exists():
            print("Preparing FineWeb-Edu",uri,flush=True)
            parquet=hf_hub_download(DATASET,uri,repo_type="dataset",revision=REVISION)
            hashes={s:hashlib.sha256() for s in files}
            counts={s:0 for s in files}
            docs={s:0 for s in files}
            with tempfile.TemporaryDirectory() as td:
                tmp={s:Path(td)/(s+".bin") for s in files}
                with tmp["train"].open("wb") as tr,tmp["val"].open("wb") as va:
                    handles={"train":tr,"val":va}
                    for batch in pq.ParquetFile(parquet).iter_batches(batch_size=256,columns=["text"]):
                        texts=[r for r in batch.column(0).to_pylist() if r and r.strip()]
                        encodings=self.enc.encode_ordinary_batch(texts,num_threads=4)
                        for t,ids in zip(texts,encodings):
                            # Identical normalized documents are assigned to the same side of the split.
                            digest=hashlib.sha256(" ".join(t.split()).encode()).digest()
                            key="val" if digest[0]<4 else "train"
                            ids.append(self.enc.eot_token)
                            b=np.asarray(ids,dtype="<u2").tobytes()
                            handles[key].write(b); hashes[key].update(b)
                            counts[key]+=len(ids); docs[key]+=1
                for s,path in files.items():
                    pending=path.with_suffix(".pending."+s)
                    shutil.copyfile(tmp[s],pending)
                    os.replace(pending,path)
            audit.save_json(manifest,{"dataset":DATASET,"revision":REVISION,"path":uri,"docs":docs,"tokens":counts,
                "sha256":{k:h.hexdigest() for k,h in hashes.items()},"schema":1,"tokenizer":"gpt2"})
        rec=json.loads(manifest.read_text())
        if (rec["revision"],rec["path"],rec["schema"])!=(REVISION,uri,1):
            raise ValueError("Foreign/incompatible corpus manifest")
        for s,path in files.items():
            if path.stat().st_size!=2*rec["tokens"][s] or audit.file_hash(path)!=rec["sha256"][s]:
                raise IOError(f"Invalid token shard {path}")
            self.maps[s].append(np.memmap(path,mode="r",dtype="<u2"))
            self.ends[s].append((self.ends[s][-1] if self.ends[s] else 0)+rec["tokens"][s])
        self.records.append(rec)

    def read(self,split,start,n):
        while not self.ends[split] or self.ends[split][-1]<start+n:
            self.add_shard()
        parts=[]; end=start+n
        while start<end:
            i=bisect.bisect_right(self.ends[split],start)
            lo=self.ends[split][i-1] if i else 0
            stop=min(end,self.ends[split][i])
            parts.append(self.maps[split][i][start-lo:stop-lo])
            start=stop
        return torch.from_numpy(np.concatenate(parts).astype(np.int64))


def views(tokens, batch, length, accum, device):
    # Consecutive blocks share exactly one boundary input. Every target is counted once.
    flat=tokens.to(device)
    return flat.unfold(0,length+1,length).reshape(accum,batch,length+1)


def train_step(name,net,opt,batches,step,total,amp):
    net.train()
    n_micro=len(batches)
    mult=_cosine_lr_multiplier(step,total,max(1,int(.03*total)),.1)
    for group in opt.param_groups: group["lr"]=6e-4*mult
    opt.zero_grad(set_to_none=True)
    for m,b in enumerate(batches):
        if name=="pulse":
            # Preserve the published v5.3 call schedule, including its critic/prior phase mismatch.
            out=net.train_step_hybrid_local(b,opt,async_core=True,core_lr_mult=mult,use_amp_bf16=amp,
                        grad_accum_steps=n_micro,is_first_micro=m==0,is_last_micro=m==n_micro-1,
                        update_critic=(step%2==0),observe_prior=(step%8==0))
            loss=out["ce_loss"]
        else:
            with torch.autocast(b.device.type,dtype=torch.bfloat16,enabled=amp):
                if name=="no_core":
                    _,emb,_,_,_=net._run_backbone(b[:,:-1],return_state=False)
                    ce=net._ce_loss_impl if net._compiled_ce is None else net._compiled_ce
                    value=ce(emb,net.tok_emb.weight,b[:,1:])
                else:
                    value=net.compute_loss(b)
            (value/n_micro).backward()
            if m==n_micro-1:
                torch.nn.utils.clip_grad_norm_((p for p in net.parameters() if p.requires_grad),1.)
                opt.step()
                loss=float(value.detach())
    return loss


def core_extra(net):
    net._ensure_causal_adam()
    states=[*net.local_opt._layer_states,{"w":net._causal_adam_w,"b":net._causal_adam_b}]
    fw=net.fast_weights
    return {"adam":[{k:{"m":v.m,"v":v.v,"step":v.step} for k,v in d.items()} for d in states],
            "latent":[fw._latent_mean,fw._latent_var,fw._latent_count,fw._prior_centroids],
            "counter":net._step_counter,"last":[net._last_core_mse,net._last_causal_mse]}


def restore_core(net,data):
    net._ensure_causal_adam()
    states=[*net.local_opt._layer_states,{"w":net._causal_adam_w,"b":net._causal_adam_b}]
    device=net.tok_emb.weight.device
    for state,src in zip(states,data["adam"],strict=True):
        for k,d in state.items():
            d.m=src[k]["m"].to(device); d.v=src[k]["v"].to(device); d.step=src[k]["step"]
    net.fast_weights.load_latent_state(*data["latent"])
    net._step_counter=data["counter"]
    net._ema_steps_host=int(net.ema_energy_steps.item())
    net._last_core_mse,net._last_causal_mse=data["last"]


def optim(net):
    return FairMuonAdamW([p for p in net.parameters() if p.requires_grad],lr=6e-4,betas=(.9,.95),weight_decay=.1)


def save_pair(path,step,models,opts,settings):
    payload={"settings":settings,"step":step,"models":{k:v.state_dict() for k,v in models.items()},
             "opts":{k:v.state_dict() for k,v in opts.items()},"core":core_extra(models["pulse"]),
             "cpu_rng":torch.get_rng_state(),"python_rng":random.getstate()}
    if next(models["pulse"].parameters()).is_cuda:
        payload["cuda_rng"]=torch.cuda.get_rng_state()
    tmp=path.with_suffix(".pending")
    torch.save(payload,tmp)
    os.replace(tmp,path)


def resume(path,models,opts,settings):
    if not path.exists(): return 0
    ck=torch.load(path,weights_only=True,map_location="cpu")
    if ck["settings"]!=settings: raise ValueError("Training settings differ from saved pair")
    for k,net in models.items():
        net.load_state_dict(ck["models"][k])
        opts[k].load_state_dict(ck["opts"][k])
    restore_core(models["pulse"],ck["core"])
    torch.set_rng_state(ck["cpu_rng"])
    random.setstate(ck["python_rng"])
    if "cuda_rng" in ck: torch.cuda.set_rng_state(ck["cuda_rng"])
    return ck["step"]


def training_ablation(preset, corpus, device, seeds=(41,42,43), steps=16):
    # Explicit retraining, with matching initial weights, optimizer and minibatches.
    # RLS has no write call in the pretraining routine. Test actual writes and reads in the edit benchmark.
    rows=[]
    for seed in seeds:
        torch.manual_seed(seed)
        torch.set_grad_enabled(True)
        full=audit.PULSEv4Model(getattr(audit.PULSEv4Config,preset)()).to(device)
        off=audit.PULSEv4Model(getattr(audit.PULSEv4Config,preset)()).to(device)
        off.load_state_dict(full.state_dict())
        opts=[optim(full),optim(off)]
        k=full.config.vocab_size
        for step in range(steps):
            start=(step+int(seed)*steps)*256
            tokens=corpus.read("train",start,257)
            if int(tokens.max())>=k: raise ValueError("Tokenizer/preset mismatch")
            b=views(tokens,2,128,1,device)
            train_step("pulse",full,opts[0],b,step,steps,device.type=="cuda")
            train_step("no_core",off,opts[1],b,step,steps,device.type=="cuda")
        diffs=[(a.detach()-b.detach()).abs().max().item() for a,b in zip(full.io_parameters(),off.io_parameters())]
        full.eval();off.eval()
        held=corpus.read("val",0,513).to(device)[None]
        with torch.no_grad():
            vfull=audit.nll_rows(audit.logits(full,held[:,:-1],"full"),held[:,1:]).mean().item()
            voff=audit.nll_rows(audit.logits(off,held[:,:-1],"no_core"),held[:,1:]).mean().item()
        rows.append({"seed":seed,"steps":steps,"max_backbone_weight_delta":max(diffs),
                     "core_updates_full":int(full.ema_energy_steps),"core_updates_off":int(off.ema_energy_steps),
                     "rls_fact_count_full":int(full.hippocampus.stored_count),
                     "full_nll":vfull,"no_core_nll":voff,"full_ppl":math.exp(vfull),"no_core_ppl":math.exp(voff)})
    return rows


@torch.no_grad()
def heldout(models,corpus,device):
    # Fixed document-hash-held-out token positions for every checkpoint.
    held=corpus.read("val",0,512*64+1)
    b=views(held,2,512,32,device)
    report={}
    for key,net in models.items():
        net.eval()
        vals=[]
        with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):
            for batch in b:
                z=audit.logits(net,batch[:,:-1],"no_core")
                vals.extend(audit.nll_rows(z,batch[:,1:]).double().mean(1).cpu().tolist())
        report[key]={"nll_by_window":vals,"ppl":math.exp(float(np.mean(vals))),"tokens":held.numel()-1}
    diff=np.subtract(report["pulse"]["nll_by_window"],report["transformer"]["nll_by_window"])
    report["paired"]={"delta_nll":float(diff.mean()),"ci95":audit.boot_ci(diff)}
    return report


def synchronize(device):
    if device.type=="cuda": torch.cuda.synchronize()


def run(args):
    device=torch.device("cuda")
    if not torch.cuda.is_available() or "L4" not in torch.cuda.get_device_name():
        raise RuntimeError("L4 required")
    torch.set_float32_matmul_precision("high")
    from pulse.v3_blocks import _probe_fla_gdn_backend
    with torch.random.fork_rng(devices=[0]):
        backend=_probe_fla_gdn_backend(device)
    torch.manual_seed(42);random.seed(42)
    out=Path(args.out); out.mkdir(exist_ok=True,parents=True)
    corpus=FineWeb(out/"data")
    audit.save_json(out/"backend.json",{k:v for k,v in backend.items() if k!="fn"})
    if not (out/"paired_training_ablation.json").exists():
        audit.save_json(out/"paired_training_ablation.json",training_ablation("colab_30m",corpus,device))
    models=audit.make_models("colab_100m",device)
    opts={k:optim(v) for k,v in models.items()}
    settings={"source":SOURCE_HASH,"revision":REVISION,"seed":42,"batch":args.batch,"accum":args.accum,
              "length":512,"goal":1_000_000_000,"optimizer":"FairMuonAdamW","schema":1,
              "compile":not args.eager,"precision":"bf16","core_schedule":"published_v53"}
    batch,accum,length=settings["batch"],settings["accum"],settings["length"]
    stride=batch*accum*length
    total=math.ceil(settings["goal"]/stride)
    ck=out/"paired_latest.pt"
    step=resume(ck,models,opts,settings)
    if not args.eager:
        for net in models.values(): net.compile_fast_paths()
    params={k:{"total":net.num_parameters(),"grad":sum(p.numel() for p in net.parameters() if p.requires_grad)}
            for k,net in models.items()}
    if not all(100_000_000<=g["total"]<=150_000_000 for g in params.values()): raise ValueError(params)
    audit.save_json(out/"design.json",dict(settings,parameters=params,tokens_per_step=stride,final_tokens_each=total*stride))
    start=step; times={k:[] for k in models}; pilot_steps=32
    last_save=step
    if step<total and time.time()>args.deadline-180:
        return
    while step<total:
        if time.time()>args.deadline-180:
            break
        raw=corpus.read("train",step*stride,stride+1)
        b=views(raw,batch,length,accum,device)
        losses={}
        # Alternate order to balance any thermal/cache/order effect. Both see the very same tensor.
        order=list(models) if step%2==0 else list(reversed(models))
        for name in order:
            synchronize(device)
            t=time.perf_counter()
            losses[name]=train_step(name,models[name],opts[name],b,step,total,True)
            synchronize(device)
            times[name].append(time.perf_counter()-t)
        step+=1
        if not all(math.isfinite(x) for x in losses.values()): raise FloatingPointError(losses)
        pilot_ready=step-start==pilot_steps
        if step%100==0 or pilot_ready or step==total:
            audit.save_json(out/"training_status.json",{"step":step,"tokens_each":step*stride,"loss_last_micro":losses,
                "complete":step==total,"time":time.time(),"source_shards":[r["path"] for r in corpus.records]})
            print(step,step*stride,losses,flush=True)
        if pilot_ready or step%1000==0 or step==total:
            save_pair(ck,step,models,opts,settings);last_save=step
            audit.save_json(out/f"val_{step:06d}.json",heldout(models,corpus,device))
        if pilot_ready:
            sec={k:float(np.mean(t[-24:])) for k,t in times.items()}
            wall_hours=sum(sec.values())*(total-step)/3600
            # Includes additional allowance for validation, data preparation, checkpoints and restarts.
            required=wall_hours*1.20*args.rate
            available=max(0.,args.available-max(0.,time.time()-args.started)*args.rate/3600)
            pred={"paired_steps":pilot_steps,"discarded_warmup":8,"seconds_per_update":sec,
                  "training_tokens_per_second":{k:stride/v for k,v in sec.items()},
                  "projected_training_hours_remaining":wall_hours,"ccu_estimate_with_20pct_allowance":required,
                  "price_for_estimate":args.rate,"available_estimate":available,"tokens_completed_each":step*stride,
                  "calibration_only":args.pilot,"can_finish_in_estimated_budget":required<=available}
            audit.save_json(out/"calibration.json",pred); print(pred,flush=True)
            if args.pilot or (required>available and not args.allow_partial): break
    if step!=last_save: save_pair(ck,step,models,opts,settings)
    done=step==total
    audit.save_json(out/"training_status.json",{"step":step,"tokens_each":step*stride,"complete":done,
                    "updated":time.time(),"next_unseen_target_index":step*stride+1})
    if done:
        # An interruption in final evaluation can be resumed from the completed pair.
        val=corpus.read("val",0,512_000)
        audit.run_audit(models,val,out/"final_audit",device,n=24,encoder=corpus.enc,fit_steps=2000)


if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("--out",required=True)
    cli.add_argument("--batch",type=int,default=8)
    cli.add_argument("--accum",type=int,default=4)
    cli.add_argument("--rate",type=float,required=True)
    cli.add_argument("--available",type=float,required=True)
    cli.add_argument("--started",type=float,required=True)
    cli.add_argument("--deadline",type=float,required=True)
    cli.add_argument("--pilot",action="store_true")
    cli.add_argument("--eager",action="store_true")
    cli.add_argument("--allow-partial",action="store_true",
                     help="Continue within the allocated session even if the full 1B goal exceeds the estimate.")
    run(cli.parse_args())
