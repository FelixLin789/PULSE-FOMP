"""L4 evidence collection before the 105M experiment."""
from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import time
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile, record_function

import review_v53 as audit
import review_v6 as v6
import review_v6_scale as scale
from pulse.pulse_v4 import PULSEv4Config, PULSEv4Model
from pulse.pulse_v6 import PULSEv6Config, PULSEv6Model
from pulse.v3_blocks import _probe_fla_gdn_backend


ARMS = ("v53", "v6", "v6_no_attnres", "v6_local_only", "transformer")


def summarize_runs(rows, rate, budgets=()):
    summary = {}
    for name in ARMS:
        variants = [row for row in rows if row["model"] == name]
        arm = {"seeds": [row["seed"] for row in variants]}
        if variants:
            arm.update(step_ppl=statistics.mean(row["holdout"]["ppl"] for row in variants),
                       median_ms=1000 * statistics.mean(row["median_seconds_per_step"] for row in variants),
                       parameters=variants[0]["parameters"], iso={})
        for budget in budgets:
            measurements = [p for row in variants for p in row.get("iso_points", [])
                            if p["target_seconds"] == budget]
            if measurements:
                arm["iso"][str(budget)] = {
                    "seeds": len(measurements), "ppl": statistics.mean(p["holdout"]["ppl"] for p in measurements),
                    "target_ccu": budget * rate / 3600,
                    "mean_steps": statistics.mean(p["steps"] for p in measurements),
                    "max_overshoot_seconds": max(p["actual_seconds"] - budget for p in measurements),
                }
        summary[name] = arm
    return summary


def rls_checkpoint(pair, stream, provenance, device):
    results = []
    for source, seed in zip(torch.tensor_split(stream, 3)[1:], (90713, 140713), strict=True):
        model = audit.fresh_copy(pair["pulse"])
        data = v6.edit_data(source, model, device, seed)
        cues, targets, _, _ = data
        before = v6.edit_metrics(model, data)
        for start in (0, 12):
            model.teach_fact_online(cues[start:start + 12], targets[start:start + 12])
        online = v6.edit_metrics(model, data)
        model.sleep_consolidate()
        fast_nonzero = int(model.memory.fast.count_nonzero())
        asleep = v6.edit_metrics(model, data, fast=False)
        drifts = {name: (score["base_ppl"] / before["base_ppl"] - 1.) * 100
                  for name, score in (("online", online), ("sleep", asleep))}
        results.append({"seed": seed, "online_hits": sum(online["hits"]), "sleep_hits": sum(asleep["hits"]),
                        "facts": len(cues), "fast_nonzero": fast_nonzero, "base_ppl_before": before["base_ppl"],
                        "base_ppl_online": online["base_ppl"], "base_ppl_sleep": asleep["base_ppl"],
                        "ppl_drift_pct": drifts, "passed": sum(online["hits"]) >= 19 and
                        sum(asleep["hits"]) >= 19 and fast_nonzero == 0 and max(map(abs, drifts.values())) < 1e-9})
    return {"checkpoint": provenance, "trials": results, "passed": all(r["passed"] for r in results)}


def profile_small(corpus, device, deadline):
    rows = {}
    for name in ("v6_no_attnres", "transformer"):
        for compiled in (False, True):
            if time.time() >= deadline:
                return rows
            torch.manual_seed(41)
            model = scale.initialize_small(name, "colab_30m", PULSEv4Model(PULSEv4Config.colab_30m()), device)
            if compiled:
                # Compile the loss graph for both arms, leaving the optimizer unchanged.
                model.compute_loss = torch.compile(model.compute_loss, dynamic=False)
            optimizer = scale.optim(model)
            batches = [scale.views(corpus.read("train", step * 4096, 4097), 4, 512, 2, device)
                       for step in range(16)]
            key = f"{name}.{'compiled' if compiled else 'eager'}"
            try:
                times = []
                for step, batch in enumerate(batches):
                    torch.cuda.synchronize()
                    tick = time.perf_counter()
                    scale.train_step(model, optimizer, batch, step, 512, True)
                    torch.cuda.synchronize()
                    times.append(time.perf_counter() - tick)
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as trace:
                    for step, batch in enumerate(batches[-5:], 16):
                        with record_function("full_training_step"):
                            scale.train_step(model, optimizer, batch, step, 512, True)
                            torch.cuda.synchronize()
                events = trace.key_averages()
                gpu = sorted(events, key=lambda e: e.self_device_time_total, reverse=True)
                cpu = sorted(events, key=lambda e: e.self_cpu_time_total, reverse=True)
                def read(events, field):
                    return [{"operator": e.key, "self_ms_per_step": getattr(e, field) / 5000,
                             "calls": e.count} for e in events[:20] if getattr(e, field) > 0]
                rows[key] = {"parameters": model.num_parameters(), "unprofiled_median_ms":
                             1000 * statistics.median(times[8:]), "profiled_steps": 5,
                             "cuda": read(gpu, "self_device_time_total"), "cpu": read(cpu, "self_cpu_time_total")}
            except Exception as exc:
                rows[key] = {"error": f"{type(exc).__name__}: {exc}"[:1600]}
            del model, optimizer, batches
            gc.collect()
            torch.cuda.empty_cache()
    return rows


@torch.no_grad()
def distance_sweep(checkpoint, device):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    weights = state["models"]["pulse"]
    default = PULSEv6Config.colab_30m()
    pool, samples, length = 256, 96, 1024
    distances = (240, 255, 256, 300, 480, 510, 511, 512, 513, 544, 640, 768)
    rng = np.random.default_rng(29084)
    keys = np.stack([rng.choice(pool, 8, replace=False) + 4 for _ in range(samples)])
    values = np.stack([rng.choice(pool, 8, replace=False) + 4 + pool for _ in range(samples)])
    noise = rng.integers(4 + 2 * pool, 4 + 3 * pool, (samples, length))
    starts = np.stack([rng.choice(np.arange(0, 200, 3), 7, replace=False) for _ in range(samples)])
    rows = {}
    for window, recurrence in ((256, True), (512, True), (1024, True), (512, False)):
        model = PULSEv6Model(replace(default, echoic_window_size=window)).to(device).eval()
        model.load_state_dict(weights)
        with ExitStack() as stack:
            if not recurrence:
                for block in (model.blocks[index] for index in default.delta_layers):
                    # Keep the block's residual/FFN and silence only its recurrence readout.
                    post = block._post_impl
                    def ablated(hidden, recurrent_output, context, post=post):
                        return post(hidden, torch.zeros_like(recurrent_output), context)
                    stack.enter_context(patch.object(block, "_post_impl", ablated))
            condition = {}
            for distance in distances:
                example = noise.copy()
                index = np.arange(samples)[:, None]
                example[index, starts] = 1
                example[index, starts + 1] = keys[:, 1:]
                example[index, starts + 2] = values[:, 1:]
                value_pos = length - 1 - distance
                example[:, value_pos-2:value_pos+1] = np.column_stack((np.ones(samples, dtype=int),
                                                                      keys[:, 0], values[:, 0]))
                example[:, -2:] = np.column_stack((np.full(samples, 2), keys[:, 0]))
                hits = 0
                for start in range(0, samples, 8):
                    tokens = torch.as_tensor(example[start:start + 8], device=device)
                    pos = torch.full((len(tokens), 1), length - 1, dtype=torch.long, device=device)
                    target = torch.as_tensor(values[start:start + 8, 0], device=device)
                    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                        hits += int(model.query_logits(tokens, pos).argmax(-1).flatten().eq(target).sum())
                condition[distance] = {"hits": hits, "queries": samples}
            rows[f"window_{window}_recurrence_{recurrence}"] = condition
    return {"old_checkpoint_step": state["step"], "test_length": length, "results": rows}


def main(args):
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    out = args.out / "pre105"
    out.mkdir(exist_ok=True)
    with torch.random.fork_rng(devices=[0]):
        backend = _probe_fla_gdn_backend(device)
    audit.save_json(out / "backend.json", {k: v for k, v in backend.items() if k != "fn"})
    if not (out / "rls.json").exists():
        pair, stream, provenance = v6.load_checkpoint_pair(args.legacy_run, device)
        audit.save_json(out / "rls.json", rls_checkpoint(pair, stream, provenance, device))
        del pair, stream
    corpus = scale.FineWeb(args.out / "fineweb_edu")
    small = scale.run_small(corpus, args.out / "ablation_runs", device, args.deadline - 120)
    audit.save_json(args.out / "small_ablation.json", small)
    audit.save_json(out / "same_steps.json", summarize_runs(small["runs"], args.rate))

    iso = scale.run_small(corpus, out / "iso_runs", device, args.deadline - 120,
                          arms=ARMS, iso_seconds=(30., 40.))
    audit.save_json(out / "iso_time.json", {"complete": iso["complete"],
              "ccu_per_hour_estimate": args.rate, "accounting": "training wall time excluding validation",
              "results": summarize_runs(iso["runs"], args.rate, (30., 40.))})
    if time.time() >= args.deadline - 180:
        return
    if not (out / "profiler.json").exists():
        audit.save_json(out / "profiler.json", profile_small(corpus, device, args.deadline - 120))
    if time.time() >= args.deadline - 180:
        return
    if not (out / "distance.json").exists():
        audit.save_json(out / "distance.json", distance_sweep(args.out / "audits/mqar_latest.pt", device))

    mqar = out / "mqar.json"
    prior = json.loads(mqar.read_text()) if mqar.exists() else {}
    if not prior.get("short_context_passed") and time.time() < args.deadline - 180:
        pair, _, _ = v6.load_checkpoint_pair(args.legacy_run, device)
        result = v6.mqar_train(pair, out, device, deadline=args.deadline - 180,
                              max_steps=args.mqar_max_steps, warmup=200, learning_rate=.0012)
        audit.save_json(mqar, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--legacy-run", type=Path, required=True)
    parser.add_argument("--rate", type=float, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    parser.add_argument("--mqar-max-steps", type=int, default=16000)
    main(parser.parse_args())
