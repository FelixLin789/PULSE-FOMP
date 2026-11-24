"""Paired checkpoint interventions and retrieval tests for the v5.3 source bundle."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from pulse.pulse_v4 import PULSEBackbone, PULSEv4Config, PULSEv4Model
from pulse.pulse_v6 import PULSEv6Model
from pulse.scalable_transformer import ScalableTransformerConfig, ScalableTransformerLM


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def file_hash(path):
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def make_models(preset, device):
    return {
        "pulse": PULSEv4Model(getattr(PULSEv4Config, preset)()).to(device),
        "transformer": ScalableTransformerLM(getattr(ScalableTransformerConfig, preset)()).to(device),
    }


def fresh_copy(net):
    new=type(net)(copy.deepcopy(net.config)).to(next(net.parameters()).device)
    new.load_state_dict(net.state_dict(),strict=True)
    if isinstance(net,PULSEv4Model):
        fw=net.fast_weights
        new.fast_weights.load_latent_state(fw._latent_mean,fw._latent_var,fw._latent_count,fw._prior_centroids)
    return new.eval()


def load_pair(run, device):
    cfg = json.loads((run / "experiment_results.json").read_text(encoding="utf-8"))
    if (cfg["config"]["preset"], cfg["config"]["pulse_version"]) != ("colab_30m", "v4"):
        raise ValueError("Expected the 30M AttnRes v5.3 run (class version v4).")
    pair = make_models("colab_30m", device)
    records = {}
    for name, net in pair.items():
        ck = run / "checkpoints" / (name + "_latest.pt")
        data = torch.load(ck, weights_only=True, map_location="cpu")
        if data["step"] != 2000:
            raise ValueError(f"{ck.name} contains step {data['step']}, expected 2000")
        net.load_state_dict(data["model_state_dict"], strict=True)
        if name == "pulse":
            net.fast_weights.load_latent_state(data["latent_mean"], data["latent_var"],
                                              data["latent_count"], data["prior_centroids"])
            if net.hippocampus.has_stored_keys:
                raise ValueError("Expected a pre-edit checkpoint with an empty RLS fact bank.")
        net.eval()
        records[name] = {"sha256": file_hash(ck), "step": int(data["step"]),
                         "parameters": net.num_parameters()}
        del data
    return pair, records


def logits(net, x, mode="full"):
    if isinstance(net, ScalableTransformerLM):
        return net(x, use_cache=False)[0]
    if isinstance(net, PULSEv6Model):
        return net(x, return_state=False, use_memory=mode=="full")["logits"]
    return net(x, run_core=mode != "no_core", use_fast_weights=mode == "full",
               return_state=False)["logits"]


def nll_rows(logit, y, cap=0.0):
    parts = []
    for i in range(0, y.size(1), 128):
        z = logit[:, i:i+128].float()
        if cap > 0:
            z = cap * torch.tanh(z / cap)
        parts.append(F.cross_entropy(z.transpose(1, 2), y[:, i:i+128], reduction="none"))
    return torch.cat(parts, 1)


def boot_ci(array, stat=np.mean, seed=8109):
    a = np.asarray(array, dtype=np.float64)
    if a.size == 0:
        return None
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(a), size=(1000, len(a)))
    return np.quantile(stat(a[draws], axis=1), [.025, .975]).tolist()


def auc(pos, neg):
    # p(positive > negative) with half credit for ties.
    a, b = np.asarray(pos), np.asarray(neg)
    sorted_b = np.sort(b)
    return float(np.mean((np.searchsorted(sorted_b, a, "left") +
                          np.searchsorted(sorted_b, a, "right")) / (2 * len(b))))


def ranged_windows(stream, width, count, seed):
    n = len(stream) // width
    if n < count:
        raise ValueError(f"Only {n} disjoint windows for requested count {count}")
    starts = np.random.default_rng(seed).choice(n, count, replace=False) * width
    return [stream[int(a):int(a)+width].long() for a in starts], starts.tolist()


def arms(pair):
    return [(name, pair["pulse"], name) for name in ("full", "no_fast", "no_core")] + [
        ("transformer", pair["transformer"], "full")]


@torch.no_grad()
def perplexity(pair, stream, lengths, n, device):
    windows, starts = ranged_windows(stream, max(lengths) + 1, n, 42003)
    result = {"window_starts": starts, "by_length": {}}
    for length in lengths:
        rows = {}
        for name, net, mode in arms(pair):
            metrics = {"all": [], "common_suffix_128": [], "tail_after_512": []}
            for window in windows:
                row = window[-length-1:].to(device)[None, :]
                z = logits(net, row[:, :-1], mode)
                cap = net.config.logit_softcap if isinstance(net, PULSEv4Model) and length > 2048 else 0
                loss = nll_rows(z, row[:, 1:], cap)
                metrics["all"].append(loss.double().mean().item())
                metrics["common_suffix_128"].append(loss[:, -min(128, length):].double().mean().item())
                if length > 512:
                    metrics["tail_after_512"].append(loss[:, 512:].double().mean().item())
                del z, loss
            rows[name] = {k: {"window_nll": v, "ppl": math.exp(float(np.mean(v)))}
                          for k, v in metrics.items() if v}
        result["by_length"][length] = {"arms": rows, "paired_nll_vs_full": {}}
        for name in ("no_fast", "no_core", "transformer"):
            for metric in rows["full"]:
                d = np.subtract(rows[name][metric]["window_nll"], rows["full"][metric]["window_nll"])
                result["by_length"][length]["paired_nll_vs_full"][name+"_"+metric] = {
                    "delta_nll": float(d.mean()), "ci95": boot_ci(d),
                    "ppl_ratio": math.exp(float(d.mean()))}
    return result


@torch.no_grad()
def common_scores(net, x, y, mode="no_core"):
    t = y.size(1)
    z = logits(net, torch.cat([x, y[:, :-1]], 1), mode)[:, -t:].float()
    lp = z.log_softmax(-1)
    nll = -lp.gather(-1, y[..., None]).squeeze(-1)
    prob = lp.exp()
    entropy = -(prob * lp).sum(-1)
    sd = (prob * (-lp - entropy[..., None]).square()).sum(-1).clamp_min(.25).sqrt()
    normalized = (nll - entropy) / sd
    bos = torch.full_like(x[:, :1], min(net.config.vocab_size - 1, 50256))
    u = logits(net, torch.cat([bos, y[:, :-1]], 1), "no_core").float().log_softmax(-1)
    unll = -u.gather(-1, y[..., None]).squeeze(-1)
    neg_pmi = (nll-unll).mean(1)
    ch_a = .60 * normalized[:, :min(3,t)].mean(1) + 1.20 * neg_pmi
    if t >= 7:
        fc = torch.cat([nll[:, :3], nll[:, 6:]], 1).mean(1)
        fu = torch.cat([unll[:, 1:3], unll[:, 6:]], 1).mean(1)
        hat = ((nll[:, 3:6].mean(1)-fc) + .50*(unll[:, 3:6].mean(1)-fu)).relu()
    elif t >= 3:
        a, b = t//3, max(t//3+1, (2*t)//3)
        hat = (nll[:, a:b].mean(1) - torch.cat([nll[:, :a], nll[:, b:]], 1).mean(1)).relu()
    else:
        hat = normalized.relu().amax(1)
    ch_b = .50 * (nll.mean(1) + .35 * hat) - 5
    shared = F.softplus(torch.logsumexp(torch.stack((2*ch_a, 2*ch_b)), 0)/2 + 2.5, beta=1.2)
    return {"nll": nll.mean(1), "negative_pmi": neg_pmi, "legacy_ab": shared}


def honesty(pair, stream, n, device, encoder=None):
    windows, _ = ranged_windows(stream, 136, n, 9021)
    rows = torch.stack(windows).to(device)
    src, truth = rows[:, :128], rows[:, 128:]
    false = truth.roll(1, 0)
    if n < 2:
        raise ValueError("Cross-document pairing needs at least two windows.")
    # Random-position token corruption is a corruption detector, not an entity benchmark.
    g = torch.Generator().manual_seed(612)
    pos = torch.randint(truth.size(1), (n,), generator=g).to(device)
    i = torch.arange(n, device=device)
    bad = truth.clone()
    bad[i, pos] = false[i, pos]
    valid_swap = bad.ne(truth).any(1)
    sets = {"shuffled_continuation": (src, truth, false), "random_token": (
        src[valid_swap], truth[valid_swap], bad[valid_swap])}
    if encoder is not None:
        names = ["Alice","David","Sarah","James","Peter","Julia","Laura","Henry"]
        patterns = [("Today {a} carried the key, while {b} carried the book.", " The key was carried by {a}."),
                    ("The letter was signed by {a}. The envelope was delivered by {b}.", " It was {a} who signed the letter."),
                    ("At the meeting {b} thanked {a} for buying the tickets.", " The person who bought the tickets was {a}."),
                    ("Of the two students, {b} took the bus and {a} rode the train.", " {a} was the one who rode the train.")]
        triples = []
        for j in range(min(n,224)):
            a, b = names[j%8], names[(j%8+1+(j//32)%7)%8]
            p, c = patterns[(j//8)%4]
            texts = [p.format(a=a,b=b), c.format(a=a), c.format(a=b)]
            triples.append(tuple(torch.tensor(encoder.encode_ordinary(s), device=device)[None] for s in texts))
        sets["controlled_name_swap"] = triples
    all_results = {}
    for kind, items in sets.items():
        triples = items if isinstance(items, list) else [
            tuple(x[j:j+1] for x in items) for j in range(len(items[0])) if not torch.equal(items[1][j], items[2][j])]
        if not triples:
            continue
        scores = {}
        for name, net, mode in arms(pair):
            acc = {}
            for p, good, wrong in triples:
                for label, cand in (("true",good),("false",wrong)):
                    with torch.no_grad():
                        out = common_scores(net, p, cand, mode)
                        if name in ("full","no_fast"):
                            out["native"] = net.continuation_free_energy(p, cand, use_fast_weights=name=="full")
                    for metric, v in out.items():
                        acc.setdefault(metric, {"true": [], "false": []})[label].append(float(v.item()))
            scores[name] = {m: {**a,"auroc":auc(a["false"],a["true"])} for m,a in acc.items()}
        refs = [("full","native","no_core","legacy_ab"),("full","native","no_fast","native"),
                ("full","negative_pmi","transformer","negative_pmi"),
                ("full","legacy_ab","transformer","legacy_ab")]
        deltas = {}
        rng = np.random.default_rng(1488)
        indices = rng.integers(len(triples),size=(500,len(triples)))
        for a,k,b,l in refs:
            av,bv=scores[a][k],scores[b][l]
            ap,an,bp,bn=[np.array(v) for v in (av["false"],av["true"],bv["false"],bv["true"])]
            d=[auc(ap[idx],an[idx])-auc(bp[idx],bn[idx]) for idx in indices]
            deltas[f"{a}.{k} minus {b}.{l}"]={"difference":av["auroc"]-bv["auroc"],
                                               "paired_ci95":np.quantile(d,[.025,.975]).tolist()}
        all_results[kind] = {"pairs":len(triples),"scores":scores,"comparisons":deltas}
    return all_results


def mqar_sample(length, pairs, vocab, seed):
    """Controlled multi-query AR variant. Query labels never occur in the query suffix."""
    rng = np.random.default_rng(seed)
    pool = min(512, (vocab-8)//3)
    if not (0 < pairs <= pool and 4*pairs+4 < length):
        raise ValueError("Invalid MQAR shape")
    keys = rng.choice(pool, pairs, replace=False)+4
    vals = rng.choice(pool, pairs, replace=False)+4+pool
    x = rng.integers(4+2*pool, min(vocab, 4+3*pool), length)
    # Disjoint pair slots and one last-quarter query per key, in a fresh order.
    slots = np.sort(rng.choice(np.arange(0, 3*length//4-1, 2), pairs, replace=False))
    qpos = np.sort(rng.choice(np.arange(3*length//4, length), pairs, replace=False))
    order = rng.permutation(pairs)
    x[slots], x[slots+1], x[qpos] = keys, vals, keys[order]
    return torch.from_numpy(x), torch.from_numpy(qpos), torch.from_numpy(vals[order]), slots[order]


@torch.no_grad()
def mqar(pair, lengths, loads, episodes, device, seed_base=410927):
    report = {}
    for length in lengths:
        for load in loads:
            if 4*load+4 >= length:
                continue
            cell = {k:[] for k in pair}
            gaps = []
            for j in range(episodes):
                x, q, y, slots = mqar_sample(length, load, pair["pulse"].config.vocab_size, seed_base+1009*j+load)
                gaps.append((q.numpy()-slots).tolist())
                for k,net in pair.items():
                    z = logits(net, x[None].to(device), "no_core")
                    guess = z[0,q.to(device)].argmax(-1).cpu()
                    cell[k].append(guess.eq(y).tolist())
                    del z
            stats = {}
            for k, rows in cell.items():
                outcomes = np.asarray(rows)
                stats[k] = {"hits":int(outcomes.sum()),"queries":int(outcomes.size),
                            "accuracy":float(outcomes.mean()),"episode_ci95":boot_ci(outcomes.mean(1)),
                            "all_queries_correct":float(outcomes.all(1).mean()),"hits_by_query":rows}
            diff = np.array(cell["pulse"],int).mean(1)-np.array(cell["transformer"],int).mean(1)
            report[f"{length}/{load}"] = {"models":stats,"gap_tokens":gaps,
                                            "paired_delta":float(diff.mean()),"paired_ci95":boot_ci(diff)}
    return report


def retrieval_fit(pair, steps, device, log_path=None):
    # Common query-only supervision on short contexts. Never optimize on long test cases.
    optim = {k:torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=6e-4) for k,m in pair.items()}
    history = []
    reached_short_accuracy = False
    for step in range(steps):
        examples = [mqar_sample(256, 8, pair["pulse"].config.vocab_size, 12000000+16*step+i) for i in range(8)]
        x,q,y = [torch.stack([e[j] for e in examples]).to(device) for j in range(3)]
        for k,net in pair.items():
            net.train()
            optim[k].zero_grad(set_to_none=True)
            z = logits(net, x, "no_core")
            loss = F.cross_entropy(z[torch.arange(len(x),device=device)[:,None],q].flatten(0,1), y.flatten())
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(),1)
            optim[k].step()
        if (step + 1) % 100 == 0 or step + 1 == steps:
            check = mqar(pair, (256,), (8,), 24, device, seed_base=910927)
            short = {name: values["accuracy"] for name, values in check["256/8"]["models"].items()}
            history.append({"step": step + 1, "short_accuracy": short})
            reached_short_accuracy = all(value >= .95 for value in short.values())
            progress = {
                "steps_completed": step + 1,
                "criterion": "Both models reach at least 95% held-out short-context accuracy",
                "positive_control_passed": reached_short_accuracy,
                "history": history,
            }
            print("MQAR short-context control", progress["history"][-1], flush=True)
            if log_path is not None:
                save_json(log_path, progress)
            if reached_short_accuracy:
                break
    for net in pair.values():
        net.eval()
    return {"steps_completed": steps if not history else history[-1]["step"],
            "positive_control_passed": reached_short_accuracy, "history": history}


def passkeys(pair, stream, encoder, lengths, n, device):
    import re
    enc = encoder.encode_ordinary
    question = enc("\nWhat is the five-digit passkey? The passkey is")
    result = {}
    for length in lengths:
        for depth in (.1,.5,.9):
            outcomes = {k:[] for k in pair}
            for j in range(n):
                rng=np.random.default_rng(49334+j)
                code = str(rng.integers(10000,100000))
                msg = enc("\nThe five-digit passkey is "+code+". Remember this passkey.\n")
                available = length-len(msg)-len(question)
                if available <= 0:
                    raise ValueError("Passkey context too short.")
                start = int(rng.integers(0,len(stream)-available))
                filler = stream[start:start+available].long().tolist()
                at = int(depth*available)
                prompt = torch.tensor(filler[:at]+msg+filler[at:]+question,device=device)[None]
                for k,net in pair.items():
                    tokens = []
                    x = prompt
                    for _ in range(8):
                        with torch.no_grad():
                            tok = int(logits(net,x,"no_core")[0,-1].argmax())
                        tokens.append(tok)
                        decoded = encoder.decode(tokens)
                        if re.search(r"^\s*\d{5}\D|^\s*\d{6}|^\s*[^\d\s]",decoded):
                            break
                        x = torch.cat([x,torch.tensor([[tok]],device=device)],1)
                    hit = bool(re.match(r"^\s*"+code+r"(?!\d)",encoder.decode(tokens)))
                    outcomes[k].append(hit)
            result[f"{length}/{depth}"]={k:{"correct":sum(v),"n":n,"accuracy":float(np.mean(v)),
                "hits":v,"wilson95":wilson(sum(v),n)} for k,v in outcomes.items()}
    return result


def wilson(hit,n):
    z = 1.95996398454
    p = hit / n
    center = (p+z*z/(2*n))/(1+z*z/n)
    margin = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [max(0.,center-margin),min(1.,center+margin)]


class LowRank(nn.Module):
    def __init__(self, base, rank):
        super().__init__()
        self.base = base
        self.a = nn.Linear(base.in_features, rank, bias=False, device=base.weight.device)
        self.b = nn.Linear(rank, base.out_features, bias=False, device=base.weight.device)
        nn.init.zeros_(self.b.weight)

    def forward(self,x):
        return self.base(x)+self.b(self.a(x))


def install_lora(net, seed):
    torch.manual_seed(seed)
    net.requires_grad_(False)
    # Every mixer has a width-by-width output projection. Identical ranks and sites at 30M/100M.
    for block in net.blocks:
        mixer = block if isinstance(net,PULSEBackbone) else block.attn
        mixer.o_proj = LowRank(mixer.o_proj,8)
    net.emb_out = LowRank(net.emb_out,8)
    params=[p for p in net.parameters() if p.requires_grad]
    return params


def lora_replay(net, opt, params, cue, target, replay, steps, old=None, replay_weight=.5, start_step=0):
    net.train()
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        # One target per fact, just like the RLS bypass. Both LoRA arms see exactly the same replay.
        facts=logits(net,cue,"no_core")[:,-1]
        r=replay[(start_step+step)%len(replay)]
        general=logits(net,r[:,:-1],"no_core")
        edit_loss=F.cross_entropy(facts,target[:,0])
        if old is not None:
            ox,oy=old
            old_loss=F.cross_entropy(logits(net,ox,"no_core")[:,-1],oy[:,0])
            edit_loss=(edit_loss+old_loss)*.5
        loss=(1-replay_weight)*edit_loss+replay_weight*F.cross_entropy(general.transpose(1,2),r[:,1:])
        loss.backward()
        nn.utils.clip_grad_norm_(params,1)
        opt.step()
    net.eval()


def edits(pair, stream, device, seed, n=12, steps=48):
    k=pair["pulse"].config.semantic_clock
    rng=np.random.default_rng(seed)
    vocab=pair["pulse"].config.vocab_size
    # Two waves. Base set is held out from edits AND from optimizer replay.
    spans, _=ranged_windows(stream, 65, 2*n+24, seed)
    cues=torch.stack([s[:2*k] for s in spans[:2*n]]).to(device)
    targets=torch.tensor(rng.choice(np.arange(4,vocab),2*n,replace=False),device=device)[:,None]
    replay=[torch.stack(spans[2*n+i:2*n+i+2]).to(device) for i in range(0,16,2)]
    base=torch.stack(spans[2*n+16:]).to(device)
    def measure(net,mode):
        with torch.no_grad():
            pred=logits(net,cues,mode)[:,-1].float()
            base_nll=nll_rows(logits(net,base[:,:-1],mode),base[:,1:]).mean().item()
        hit=pred.argmax(1).eq(targets[:,0]).cpu().tolist()
        return {"old_top1":sum(hit[:n])/n,"new_top1":sum(hit[n:])/n,"hits":hit,
                "target_nll":float(F.cross_entropy(pred,targets[:,0])),"base_ppl":math.exp(base_nll)}
    result={}
    native=fresh_copy(pair["pulse"])
    stages={"pre":measure(native,"full")}
    for wave in range(2):
        for i in range(wave*n,(wave+1)*n):
            native.teach_fact_online(cues[i:i+1],targets[i:i+1])
        stages[f"after_wave_{wave+1}"]=measure(native,"full")
    stages["remove_fast_after_teaching"]=measure(native,"no_fast")
    stages["remove_core_after_teaching"]=measure(native,"no_core")
    native.sleep_consolidate(num_replay_samples=48,clear_fast_weights=True)
    stages["sleep_public_lm"]=measure(native,"full")
    stages["sleep_gate_disabled"]=measure(native,"no_fast")
    stages["fast_norm_after_sleep"]=float(native.hippocampus.w_fast.norm())
    result["pulse_rls"]=stages
    del native
    for name, original in pair.items():
        net=fresh_copy(original)
        st={"pre":measure(net,"no_core")}
        params=install_lora(net,seed)
        opt=torch.optim.AdamW(params,lr=0.002,weight_decay=0)
        st["adapter_parameters"]=sum(p.numel() for p in params)
        for wave in range(2):
            lo=wave*n
            old=(cues[:n],targets[:n]) if wave else None
            lora_replay(net,opt,params,cues[lo:lo+n],targets[lo:lo+n],replay,steps,old)
            st[f"after_wave_{wave+1}"]=measure(net,"no_core")
        result[name+"_lora_replay"]=st
    return result


def backing_bytes(state):
    stores={}
    def walk(x):
        if torch.is_tensor(x):
            s=x.untyped_storage()
            stores[(str(x.device),s.data_ptr())]=s.nbytes()
        elif isinstance(x,dict):
            for v in x.values(): walk(v)
        elif isinstance(x,(list,tuple)):
            for v in x: walk(v)
    walk(state)
    return sum(stores.values())


@torch.no_grad()
def invariants(pair, lengths, device):
    p=pair["pulse"]
    result={}
    for n in lengths:
        g=torch.Generator().manual_seed(3191+n)
        x=torch.randint(4,p.config.vocab_size,(1,n),generator=g).to(device)
        out=p(x,use_fast_weights=True,run_core=True,return_state=True)
        state_ref=out["logits"][:,[n//2-1,-1]].detach().clone()
        state=out["working_state"]
        record={"logical_bytes":p.working_state_bytes(state),"backing_bytes":backing_bytes(state),
                "empty_bank":not p.hippocampus.has_stored_keys}
        del out,state
        ref=logits(p,x,"full")[:,[n//2-1,-1]].clone()
        record["return_state_kernel_delta"]=float((ref-state_ref).abs().max())
        for mode in ("no_fast","no_core"):
            z=logits(p,x,mode)[:,[n//2-1,-1]]
            record["max_logit_delta_"+mode]=float((ref-z).abs().max())
            del z
        for name,net in pair.items():
            # Same shape and kernel path; change only the unseen suffix.
            suffix=x.clone()
            suffix[:,n//2:]=torch.randint(4,p.config.vocab_size,(1,n-n//2),generator=g).to(device)
            original=logits(net,x,"no_core")[:,n//2-1].clone()
            other=logits(net,suffix,"no_core")[:,n//2-1]
            record[name+"_future_suffix_delta"]=float((original-other).abs().max())
            del original,other
        result[n]=record
    return result


def run_audit(pair, stream, out, device, n=24, smoke=False, encoder=None, fit_steps=160):
    out=Path(out)
    lengths=(64,128) if smoke else (512,2048,4096,8192)
    scenarios={"interventions":lambda:invariants(pair,lengths,device),
               "ppl":lambda:perplexity(pair,stream,lengths,min(n,24),device),
               "corruptions":lambda:honesty(pair,stream,n,device,encoder),
               "mqar_zero_shot":lambda:mqar(pair,(64,128) if smoke else (256,512,4096,8192),
                                            (4,8) if smoke else (8,32,64),n,device),
               "edits":lambda:edits(pair,stream,device,40713,n=3 if smoke else 12,steps=2 if smoke else 48)}
    if encoder is not None:
        scenarios["passkey"]=lambda:passkeys(pair,stream,encoder,(512,4096,8192),n,device)
    for name,task in scenarios.items():
        print("Evaluating",name,flush=True)
        save_json(out/(name+".json"),task())
    if fit_steps:
        training = retrieval_fit(pair,fit_steps,device,out/"mqar_training_control.json")
        save_json(out/"mqar_task_trained.json",{"training":training,"train_context":256,
            "load":8,"evaluation":mqar(pair,(256,512,4096,8192),(8,32,64),n,device)})


if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("--run",type=Path,required=True)
    cli.add_argument("--out",type=Path,required=True)
    cli.add_argument("--episodes",type=int,default=24)
    args=cli.parse_args()
    import tiktoken
    torch.manual_seed(42)
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models,provenance=load_pair(args.run,dev)
    val=args.run/"data_cache"/"val_stream_tinystories.pt"
    save_json(args.out/"provenance.json",{"checkpoints":provenance,"validation_sha256":file_hash(val),
                                         "device":str(dev),"torch":torch.__version__})
    run_audit(models,torch.load(val,weights_only=True,map_location="cpu"),args.out,dev,
              n=args.episodes,encoder=tiktoken.get_encoding("gpt2"))
