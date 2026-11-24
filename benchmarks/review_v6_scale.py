"""FineWeb-Edu ablations and a paired, resumable 105M pretraining run."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

import review_v53 as audit
import review_v6 as v6
from review_scale import DATASET, REVISION, FineWeb, optim, synchronize, train_step as legacy_step, views
from pulse.experiment import _cosine_lr_multiplier
from pulse.pulse_v4 import PULSEv4Config, PULSEv4Model
from pulse.pulse_v6 import PULSEv6Config, PULSEv6Model
from pulse.v3_blocks import _probe_fla_gdn_backend


def train_step(model, optimizer, batches, step, steps, amp):
    if isinstance(model, PULSEv4Model):
        return legacy_step("pulse", model, optimizer, batches, step, steps, amp)
    model.train()
    multiplier = _cosine_lr_multiplier(step, steps, max(1, int(.03 * steps)), .1)
    for group in optimizer.param_groups:
        group["lr"] = .0006 * multiplier
    optimizer.zero_grad(set_to_none=True)
    batch_losses = []
    for tokens in batches:
        with torch.autocast(tokens.device.type, dtype=torch.bfloat16, enabled=amp):
            loss = model.compute_loss(tokens)
        (loss / len(batches)).backward()
        batch_losses.append(loss.detach())
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., foreach=amp)
    optimizer.step()
    return float(torch.stack(batch_losses).mean())


@torch.no_grad()
def evaluate(model, tokens, length, device):
    model.eval()
    windows = tokens.unfold(0, length + 1, length)
    rows = []
    amp = device.type == "cuda"
    for b in windows.split(2 if length <= 512 else 1):
        b = b.to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            logits = audit.logits(model, b[:, :-1], "no_core")
            cap = model.config.logit_softcap if isinstance(model, PULSEv4Model) and length > 2048 else 0
            rows.extend(audit.nll_rows(logits, b[:, 1:], cap).double().mean(1).cpu().tolist())
    return {"ppl": math.exp(float(np.mean(rows))), "window_nll": rows, "tokens": len(rows) * length}


def initialize_small(name, preset, backbone, device):
    if name == "v53":
        return backbone.to(device)
    if name == "transformer":
        return v6.make_pair(preset, device)["transformer"]
    config = getattr(PULSEv6Config, preset)()
    if name == "v6_no_attnres":
        config = replace(config, use_attn_res=False)
    elif name == "v6_local_only":
        config = replace(config, delta_layers=(), use_multiscale=False)
    elif name == "v6_no_memory":
        config = replace(config, memory_capacity=0)
    net = PULSEv6Model(config).to(device)
    if name == "v6_local_only":
        align_local_weights(net, backbone)
    else:
        net.load_v53_backbone(backbone.state_dict())
    return net


def align_local_weights(net, backbone):
    """Share compatible initial weights. Replacement mixers retain their own initialization."""
    source = backbone.state_dict()
    state = net.state_dict()
    for name, weight in state.items():
        original = name.replace(".canon_a.conv.", ".canon_a.local_c.conv.")
        if name.startswith("blocks.") and int(name.split(".")[1]) in backbone.config.delta_layers:
            stem = name.split(".")[2]
            if stem not in ("depth_attn", "w_gate", "w_up", "w_down"):
                continue
        if original in source and source[original].shape == weight.shape:
            state[name] = source[original]
    net.load_state_dict(state)


def online_recall(model, stream, seed, device):
    if not isinstance(model, PULSEv6Model) or model.memory is None:
        if isinstance(model, PULSEv6Model):
            data = v6.edit_data(stream, model, device, seed)
            return {"without_memory": v6.edit_metrics(model, data)}
        return {}
    data = v6.edit_data(stream, model, device, seed)
    cues, targets, _, _ = data
    half = len(cues) // 2
    model.teach_fact_online(cues[:half], targets[:half])
    model.teach_fact_online(cues[half:], targets[half:])
    online = v6.edit_metrics(model, data)
    disabled = v6.edit_metrics(model, data, mode="off")
    model.sleep_consolidate()
    sleep = v6.edit_metrics(model, data, fast=False)
    return {"online": online, "without_memory": disabled, "sleep_fast_off": sleep}


def run_small(corpus, out, device, deadline, *, steps=512, preset="colab_30m", seeds=(41, 42, 43),
              length=512, batch=4, accum=2, arms=None, iso_seconds=()):
    arms = arms or ("v53", "v6", "v6_no_attnres", "v6_local_only", "v6_no_memory", "transformer")
    amp = device.type == "cuda"
    stride = length * batch * accum
    val = corpus.read("val", 0, length * 64 + 1)
    edit_stream = corpus.read("val", val.numel(), 8192)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, seed in enumerate(seeds):
        torch.manual_seed(seed)
        backbone = PULSEv4Model(getattr(PULSEv4Config, preset)()).eval()
        # Use fresh backbone weights even when v5.3 has finished its own run.
        base_weights = {key: value.detach().clone() for key, value in backbone.state_dict().items()}
        for name in arms:
            path = out / f"{seed}_{name}.json"
            if path.exists():
                record = json.loads(path.read_text(encoding="utf-8"))
                if (record["steps"], record["length"], record["tokens"], record["preset"]) != (steps, length, stride * steps, preset):
                    raise ValueError(f"Small ablation settings differ from {path.name}.")
                rows.append(record)
                continue
            if time.time() + 30 >= deadline:
                return {"complete": False, "runs": rows}
            torch.manual_seed(seed)
            backbone.load_state_dict(base_weights, strict=True)
            net = initialize_small(name, preset, backbone, device)
            optimizer = optim(net)
            duration = []
            timed_points = []
            train_wall = 0.
            cutoffs = iter(sorted(iso_seconds))
            cutoff = next(cutoffs, None)
            last_loss = None
            for step in range(steps):
                if time.time() + 30 >= deadline:
                    return {"complete": False, "runs": rows}
                started_step = time.perf_counter()
                start = (index * steps + step) * stride
                tokens = corpus.read("train", start, stride + 1)
                batches = views(tokens, batch, length, accum, device)
                synchronize(device)
                tick = time.perf_counter()
                last_loss = train_step(net, optimizer, batches, step, steps, amp)
                synchronize(device)
                if not math.isfinite(last_loss):
                    raise FloatingPointError(f"{name} step {step+1}: {last_loss}")
                duration.append(time.perf_counter() - tick)
                train_wall += time.perf_counter() - started_step
                if cutoff is not None and train_wall >= cutoff:
                    measured = evaluate(net, val, length, device)
                    while cutoff is not None and train_wall >= cutoff:
                        timed_points.append({"target_seconds": cutoff, "actual_seconds": train_wall,
                                             "steps": step + 1, "tokens": (step + 1) * stride,
                                             "holdout": measured})
                        cutoff = next(cutoffs, None)
            metrics = evaluate(net, val, length, device)
            recorded = {
                "seed": seed, "model": name, "steps": steps, "length": length, "tokens": stride * steps, "preset": preset,
                "train_start": index * steps * stride, "parameters": net.num_parameters(),
                "train_last_micro_loss": last_loss, "median_seconds_per_step": float(np.median(duration[min(8, steps//2):])),
                "holdout": metrics, "edits": online_recall(net, edit_stream, seed, device),
            }
            if iso_seconds:
                recorded.update(iso_points=timed_points, training_wall_seconds=train_wall)
            audit.save_json(path, recorded)
            rows.append(recorded)
            print("FineWeb-Edu ablation", seed, name, metrics["ppl"], flush=True)
            del net, optimizer
            if name == "v53":
                # The legacy model was the only arm updating the detached predictive core.
                backbone = PULSEv4Model(getattr(PULSEv4Config, preset)()).eval()
            if amp:
                gc.collect()
                torch.cuda.empty_cache()

    pairs = {}
    for name in arms:
        if name == "v6" or "v6" not in arms:
            continue
        deltas = []
        for seed in seeds:
            ref = next(row for row in rows if row["seed"] == seed and row["model"] == "v6")
            arm = next(row for row in rows if row["seed"] == seed and row["model"] == name)
            deltas.append(float(np.mean(np.subtract(arm["holdout"]["window_nll"], ref["holdout"]["window_nll"]))))
        pairs[name] = {"seeds": seeds, "delta_nll_vs_v6": deltas, "mean_delta_nll": float(np.mean(deltas))}
    return {"complete": True, "runs": rows, "paired_validation": pairs,
            "local_only_change": "Gated delta layers replaced by sliding attention and dilated taps removed"}


def save_pair(path, step, models, opts, settings):
    tmp = path.with_suffix(".pending")
    state = {"settings": settings, "step": step, "rng": torch.get_rng_state(),
             "models": {name: model.state_dict() for name, model in models.items()},
             "optim": {name: opt.state_dict() for name, opt in opts.items()}}
    if next(models["pulse"].parameters()).is_cuda:
        state["cuda_rng"] = torch.cuda.get_rng_state()
    torch.save(state, tmp)
    os.replace(tmp, path)


def resume_pair(path, models, opts, settings):
    if not path.exists():
        return 0
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["settings"] != settings:
        raise ValueError("The saved 105M pair uses a different corpus, model, or training schedule.")
    for name, model in models.items():
        model.load_state_dict(checkpoint["models"][name])
        opts[name].load_state_dict(checkpoint["optim"][name])
    torch.set_rng_state(checkpoint["rng"])
    if "cuda_rng" in checkpoint:
        torch.cuda.set_rng_state(checkpoint["cuda_rng"])
    return checkpoint["step"]


def scale_session(corpus, args, device, batch, accum, compile_models):
    out = args.out / "scale"
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    pair = v6.make_pair("colab_100m", device)
    optimizers = {name: optim(net) for name, net in pair.items()}
    if compile_models:
        for model in pair.values():
            model.compile_fast_paths()
    settings = {"code_sha256": args.code_sha256, "dataset": DATASET, "revision": REVISION, "seed": 42,
                "batch": batch, "accum": accum, "length": 512, "goal": args.goal, "compile": compile_models,
                "optimizer": "FairMuonAdamW", "precision": "bf16"}
    checkpoint = out / "paired_latest.pt"
    step = resume_pair(checkpoint, pair, optimizers, settings)
    length = settings["length"]
    stride = batch * accum * length
    total = math.ceil(args.goal / stride)
    params = {name: model.num_parameters() for name, model in pair.items()}
    audit.save_json(out / "design.json", dict(settings, parameters=params, final_tokens_each=total * stride))
    started_step = step
    saved_step = step
    duration = {name: [] for name in pair}
    val = corpus.read("val", 0, length * 128 + 1)

    def keep_checkpoint():
        save_pair(checkpoint, step, pair, optimizers, settings)
        if step == total or step == started_step + 48 or step % 1000 == 0:
            metrics = {name: evaluate(net, val, length, device) for name, net in pair.items()}
            diff = np.subtract(metrics["pulse"]["window_nll"], metrics["transformer"]["window_nll"])
            metrics["paired"] = {"mean_nll_delta": float(diff.mean()), "ci95": audit.boot_ci(diff)}
            audit.save_json(out / f"val_{step:06d}.json", metrics)

    while step < total and time.time() < args.deadline - 180:
        raw = corpus.read("train", step * stride, stride + 1)
        if time.time() >= args.deadline - 180:
            break
        batches = views(raw, batch, length, accum, device)
        losses = {}
        order = tuple(pair) if step % 2 == 0 else tuple(reversed(pair))
        for name in order:
            synchronize(device)
            tick = time.perf_counter()
            losses[name] = train_step(pair[name], optimizers[name], batches, step, total, True)
            synchronize(device)
            duration[name].append(time.perf_counter() - tick)
        step += 1
        if not all(math.isfinite(x) for x in losses.values()):
            raise FloatingPointError(losses)
        if step % 100 == 0 or step == total or step == started_step + 48:
            audit.save_json(out / "training_progress.json", {"step": step, "tokens_each": step * stride,
                "committed_step": saved_step, "loss": losses, "updated_utc": time.time()})
            print("FineWeb-Edu 105M", step, step * stride, losses, flush=True)
        if step == total or step == started_step + 48 or step % 1000 == 0:
            keep_checkpoint()
            saved_step = step
        if step == started_step + 48:
            sec = {name: float(np.mean(times[-32:])) for name, times in duration.items()}
            hours_left = (total - step) * sum(sec.values()) / 3600
            calibrated = {"step": step, "tokens_each": step * stride, "seconds_per_update": sec,
                          "tokens_per_second_each": {name: stride / s for name, s in sec.items()},
                          "estimated_ccu_to_goal": hours_left * 1.2 * args.rate, "rate": args.rate,
                          "estimated_budget_left": max(0., args.available - (time.time() - args.started) / 3600 * args.rate)}
            audit.save_json(out / "calibration.json", calibrated)
            print("105M throughput calibration", calibrated, flush=True)

    if step != saved_step:
        keep_checkpoint()
    complete = step == total
    outcome = {"complete": complete, "step": step, "tokens_each": step * stride, "goal": args.goal}
    if complete:
        long_val = corpus.read("val", val.numel(), 8 * 8192 + 1)
        for seq_len in (1024, 4096, 8192):
            if time.time() >= args.deadline - 60:
                break
            dest = out / f"final_ppl_{seq_len}.json"
            if not dest.exists():
                stats = {name: evaluate(net, long_val[:seq_len * 8 + 1], seq_len, device) for name, net in pair.items()}
                audit.save_json(dest, stats)
        outcome["final_extrapolation_complete"] = all((out / f"final_ppl_{n}.json").exists() for n in (1024, 4096, 8192))
    audit.save_json(out / "training_status.json", outcome)
    return outcome


def run(args):
    device = torch.device("cuda")
    if not torch.cuda.is_available() or "L4" not in torch.cuda.get_device_name():
        raise RuntimeError("NVIDIA L4 is required for this run.")
    torch.set_float32_matmul_precision("high")
    with torch.random.fork_rng(devices=[0]):
        probe = _probe_fla_gdn_backend(device)
    audit.save_json(args.out / "backend.json", {k: v for k, v in probe.items() if k != "fn"})
    corpus = FineWeb(args.out / "fineweb_edu")
    if not args.skip_small:
        ablation = args.out / "small_ablation.json"
        old = json.loads(ablation.read_text(encoding="utf-8")) if ablation.exists() else {}
        if not old.get("complete"):
            data = run_small(corpus, args.out / "ablation_runs", device,
                             args.deadline - 300, steps=args.small_steps)
            audit.save_json(ablation, data)
            if not data["complete"]:
                print("The 105M stage will continue after the remaining small ablations.", flush=True)
                return

    if args.small_only or time.time() > args.deadline - 240:
        return
    scale_dir = args.out / "scale"
    config_path = scale_dir / "design.json"
    checkpoint = scale_dir / "paired_latest.pt"
    if checkpoint.exists():
        design = json.loads(config_path.read_text(encoding="utf-8"))
        scale_session(corpus, args, device, design["batch"], design["accum"], design["compile"])
        return
    for batch, accum, compile_models in ((8, 4, True), (4, 8, True), (4, 8, False)):
        try:
            scale_session(corpus, args, device, batch, accum, compile_models)
            return
        except torch.OutOfMemoryError:
            if checkpoint.exists() or (batch, compile_models) == (4, False):
                raise
            print("L4 reached its allocation limit during setup. Retrying with a smaller common microbatch.", flush=True)
        except torch._dynamo.exc.BackendCompilerFailed:
            if checkpoint.exists() or not compile_models:
                raise
            print("Compiled setup failed. Retrying with the same loss and a smaller common microbatch.", flush=True)
        torch._dynamo.reset()
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--code-sha256", required=True)
    parser.add_argument("--goal", type=int, default=1_000_000_000)
    parser.add_argument("--small-steps", type=int, default=512)
    parser.add_argument("--rate", type=float, required=True)
    parser.add_argument("--available", type=float, required=True)
    parser.add_argument("--started", type=float, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    parser.add_argument("--skip-small", action="store_true")
    parser.add_argument("--small-only", action="store_true")
    run(parser.parse_args())
