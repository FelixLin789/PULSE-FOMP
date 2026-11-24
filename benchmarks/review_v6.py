"""Measure compact state, MQAR convergence, and LoRA replay on the same cue edits."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

import review_v53 as audit
from pulse.pulse_v4 import PULSEv4Config, PULSEv4Model
from pulse.pulse_v6 import PULSEv6Config, PULSEv6Model
from pulse.scalable_transformer import ScalableTransformerConfig, ScalableTransformerLM


def make_pair(preset, device):
    pulse = PULSEv6Model(getattr(PULSEv6Config, preset)()).to(device)
    transformer_config = getattr(ScalableTransformerConfig, preset)()
    if preset == "micro_test":
        transformer_config = replace(transformer_config, n_layers=4, d_ff=136)
    return {"pulse": pulse.eval(), "transformer": ScalableTransformerLM(transformer_config).to(device).eval()}


def load_checkpoint_pair(run, device):
    legacy, provenance = audit.load_pair(Path(run), device)
    # Match legacy v5.3 backbone settings so loaded weights produce identical logits.
    pulse = PULSEv6Model(
        replace(PULSEv6Config.colab_30m(), use_attn_res=True, delta_beta_floor=0.05)
    ).to(device).eval()
    pulse.load_v53_backbone(legacy["pulse"].state_dict())
    stream = torch.load(Path(run) / "data_cache/val_stream_tinystories.pt", weights_only=True, map_location="cpu")
    return {"pulse": pulse, "transformer": legacy["transformer"]}, stream, provenance


@torch.no_grad()
def state_audit(device, amp=False, compile_models=False):
    modes = {"v53": PULSEv4Model(PULSEv4Config.colab_30m()).to(device).eval(),
             "v6": PULSEv6Model(PULSEv6Config.colab_30m()).to(device).eval()}
    if compile_models:
        for net in modes.values():
            net.compile_fast_paths()
    rows = {}
    g = torch.Generator().manual_seed(82043)
    for name, model in modes.items():
        states = {}
        for length in (512, 4096, 8192):
            tokens = torch.randint(4, model.config.vocab_size, (1, length), generator=g).to(device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                baseline = torch.cuda.memory_allocated(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                if name == "v6":
                    result = model(tokens, last_logits_only=True)
                    state = result["working_state"]
                    del result
                    next_token = model(tokens[:, :1], state, last_logits_only=True)["working_state"]
                else:
                    _, _, _, blocks, _ = model._run_backbone(tokens, return_state=True)
                    state = {"blocks": blocks, "ssm": torch.zeros((1, model.config.top_dim), device=device)}
                    _, _, _, more, _ = model._run_backbone(tokens[:, :1], block_states=blocks, return_state=True)
                    next_token = {"blocks": more, "ssm": state["ssm"]}
            logical, backing = model.working_state_bytes(state), audit.backing_bytes(state)
            decoded = audit.backing_bytes(next_token)
            if logical != backing or backing != decoded:
                raise AssertionError((name, length, logical, backing, decoded))
            samples = {"logical_bytes": logical, "backing_bytes": backing, "continuation_backing_bytes": decoded}
            if device.type == "cuda":
                samples["peak_additional_allocation_bytes"] = torch.cuda.max_memory_allocated(device) - baseline
            states[length] = samples
            del state, next_token
            print(name, length, samples, flush=True)
        if len({row["backing_bytes"] for row in states.values()}) != 1:
            raise AssertionError(states)
        rows[name] = states
    return {"device": torch.cuda.get_device_name(device) if device.type == "cuda" else device.type,
            "dtype": "bf16 autocast" if amp else "float32", "compile": compile_models,
            "random_initialization": True, "batch": 1, "models": rows}


def mqar_batch(net, length, pairs, batch_size, seed, device):
    """Place disjoint [write, key, value] records and [read, key] queries."""
    pool = min(256, (net.config.vocab_size - 8) // 3)
    prefix = 3 * length // 4
    write_slots = np.arange(0, prefix - 2, 3)
    read_slots = np.arange(prefix, length - 1, 2)
    if len(write_slots) < pairs or len(read_slots) < pairs or pairs > pool:
        raise ValueError(f"Invalid MQAR geometry: {length=}, {pairs=}, {pool=}")
    rng = np.random.default_rng(seed)
    tokens = rng.integers(4 + 2 * pool, 4 + 3 * pool, (batch_size, length))
    query = np.empty((batch_size, pairs), dtype=np.int64)
    target = np.empty_like(query)
    gap = np.empty_like(query)
    value_positions = np.empty_like(query)
    key_targets = np.empty_like(query)
    value_targets = np.empty_like(query)
    for i in range(batch_size):
        keys = rng.choice(pool, pairs, replace=False) + 4
        values = rng.choice(pool, pairs, replace=False) + 4 + pool
        writes = rng.choice(write_slots, pairs, replace=False)
        reads = np.sort(rng.choice(read_slots, pairs, replace=False))
        order = rng.permutation(pairs)
        tokens[i, writes], tokens[i, writes+1], tokens[i, writes+2] = 1, keys, values
        tokens[i, reads], tokens[i, reads+1] = 2, keys[order]
        query[i], target[i] = reads + 1, values[order]
        gap[i] = reads + 1 - (writes[order] + 2)
        value_positions[i], key_targets[i], value_targets[i] = writes + 2, keys, values
    items = (tokens, query, target, value_positions, key_targets, value_targets)
    return tuple(torch.from_numpy(t).to(device) for t in items), gap


@torch.no_grad()
def mqar_eval(pair, lengths, loads, episodes, device, seed, amp=False):
    report = {}
    for model in pair.values():
        model.eval()
    for length in lengths:
        batch_size = 8 if length <= 1024 else 1
        for load in loads:
            if load > (length - 3 * length // 4) // 2:
                continue
            hits = {k: [] for k in pair}
            gaps = []
            for start in range(0, episodes, batch_size):
                batch = min(batch_size, episodes - start)
                batch_data, gap = mqar_batch(
                    pair["pulse"], length, load, batch, seed + 517 * (start + 1000 * load), device
                )
                tokens, query, targets = batch_data[:3]
                gaps.extend(gap.tolist())
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                    for name, net in pair.items():
                        predicted = net.query_logits(tokens, query).argmax(-1)
                        hits[name].extend(predicted.eq(targets).cpu().tolist())
            rows = {}
            distance = np.asarray(gaps)
            for name, values in hits.items():
                correct = np.asarray(values)
                by_distance = {}
                for low, high in ((0, 256), (256, 512), (512, 1024), (1024, 4096), (4096, 8192)):
                    keep = (distance >= low) & (distance < high)
                    if keep.any():
                        by_distance[f"{low}-{high-1}"] = {"hits": int(correct[keep].sum()),
                                                          "queries": int(keep.sum())}
                rows[name] = {"hits": int(correct.sum()), "queries": int(correct.size),
                              "accuracy": float(correct.mean()), "perfect_episodes": int(correct.all(1).sum()),
                              "episode_ci95": audit.boot_ci(correct.mean(1)), "by_distance": by_distance}
            delta = np.asarray(hits["pulse"], dtype=float).mean(1) - np.asarray(hits["transformer"], dtype=float).mean(1)
            report[f"{length}/{load}"] = {"models": rows, "paired_gap": float(delta.mean()),
                                           "paired_ci95": audit.boot_ci(delta)}
    return report


def mqar_train(pair, out, device, *, max_steps=16000, batch=16, seed=42, episodes=48, deadline=math.inf,
               warmup=200, learning_rate=1.2e-3, auxiliary=.25, checkpoint_interval=500):
    amp = device.type == "cuda"
    opts = {name: torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=.01, fused=amp)
            for name, model in pair.items()}
    pool = min(256, (pair["pulse"].config.vocab_size - 8) // 3)
    history = []
    started = time.monotonic()
    checkpoint = out / "mqar_curriculum.pt"
    curriculum = ((8, 1), (16, 2), (32, 4), (64, 8), (128, 8), (256, 8))
    stage = 0
    signature = {"seed": seed, "batch": batch, "vocab": pair["pulse"].config.vocab_size,
                 "params": {name: model.num_parameters() for name, model in pair.items()}, "layout": 5,
                 "warmup": warmup, "learning_rate": learning_rate, "auxiliary": auxiliary}
    completed = 0
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state["signature"] != signature:
            raise ValueError("MQAR checkpoint uses a different model or sampling protocol.")
        history = state["history"]
        completed = state["step"]
        stage = state["stage"]
        for name, model in pair.items():
            model.load_state_dict(state["models"][name])
            opts[name].load_state_dict(state["opts"][name])
        del state
    passed = any(row.get("stage_length") == 256 and all(value >= .90 for value in row["accuracy"].values())
                 for row in history[-1:])
    first_step = completed
    def save_checkpoint():
        tmp = checkpoint.with_suffix(".pending")
        torch.save({"signature": signature, "step": completed, "stage": stage, "history": history,
                    "models": {name: model.state_dict() for name, model in pair.items()},
                    "opts": {name: opt.state_dict() for name, opt in opts.items()}}, tmp)
        os.replace(tmp, checkpoint)

    for step in range(completed, max_steps):
        if passed or time.time() >= deadline:
            break
        # The same curriculum teaches both models the task before testing at longer lengths.
        length, pairs = curriculum[stage]
        batch_data, _ = mqar_batch(
            pair["pulse"], length, pairs, batch, 13_000_000 + seed * 1_000_000 + 10_007 * step, device
        )
        tokens, query, target, record_pos, keys, values = batch_data
        positions = torch.cat((query, record_pos), dim=1)
        for name, model in pair.items():
            model.train()
            opt = opts[name]
            for group in opt.param_groups:
                group["lr"] = learning_rate * min(1., (step + 1) / max(1, warmup))
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                scores = model.query_logits(tokens, positions)
                answers, records = scores.split(pairs, dim=1)
                recall = F.cross_entropy(answers.flatten(0, 1), target.flatten())
                key_copy = F.cross_entropy(records[:, :, 4:4+pool].flatten(0, 1), keys.flatten() - 4)
                value_copy = F.cross_entropy(records[:, :, 4+pool:4+2*pool].flatten(0, 1), values.flatten() - 4 - pool)
                loss = recall + auxiliary * (key_copy + value_copy)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            opt.step()
        completed = step + 1
        if completed % 100 == 0 or completed == max_steps:
            dev = mqar_eval(pair, (256,), (8,), episodes, device, seed=910927 + seed, amp=amp)["256/8"]
            acc = {name: row["accuracy"] for name, row in dev["models"].items()}
            current = mqar_eval(pair, (length,), (pairs,), episodes, device,
                                seed=1710927 + seed, amp=amp)[f"{length}/{pairs}"]
            stage_acc = {name: row["accuracy"] for name, row in current["models"].items()}
            history.append({"step": completed, "accuracy": acc, "stage_length": length,
                            "stage_accuracy": stage_acc})
            passed = stage == len(curriculum) - 1 and all(x >= .90 for x in acc.values())
            if not passed and stage < len(curriculum) - 1 and all(x > .94 for x in stage_acc.values()):
                stage += 1
            audit.save_json(out / "mqar_progress.json", {"history": history, "short_context_passed": passed})
            print("MQAR", history[-1], flush=True)
            if passed:
                break
        if completed % checkpoint_interval == 0:
            save_checkpoint()
    if completed > first_step:
        save_checkpoint()
    scores = mqar_eval(pair, (256, 1024, 4096, 8192), (8, 32, 64), episodes, device,
                       seed=410927 + seed, amp=amp) if passed else None
    return {"steps": completed, "history": history, "seed": seed, "short_context_passed": passed,
            "layout": "MQAR variant with explicit write and query markers and randomly located records",
            "training_auxiliary": "For both models, recover the observed key and value at the value position.",
            "threshold": "Both models >=90% on independent length-256 development data",
            "recipe": signature,
            "seconds": time.monotonic() - started, "batch": batch, "test": scores,
            "model_parameters": {name: model.num_parameters() for name, model in pair.items()}}


def edit_data(stream, net, device, seed, n=12):
    windows, _ = audit.ranged_windows(stream, 65, 2 * n + 40, seed)
    rng = np.random.default_rng(seed)
    edit = torch.stack([w[:16] for w in windows[:2*n]]).to(device)
    values = torch.tensor(rng.choice(np.arange(4, net.config.vocab_size), 2*n, replace=False), device=device)[:, None]
    replay = torch.stack(windows[2*n:2*n+16]).to(device).split(2)
    base = torch.stack(windows[2*n+16:]).to(device)
    return edit, values, replay, base


@torch.no_grad()
def edit_metrics(net, data, mode="full", fast=True):
    cues, targets, _, background = data
    net.eval()
    if isinstance(net, PULSEv6Model):
        def predict(x, last=False):
            return net(x, return_state=False, use_memory=mode=="full", use_fast_weights=fast,
                       last_logits_only=last)["logits"]
    else:
        def predict(x, last=False):
            if last:
                positions = torch.full_like(x[:, :1], x.size(1) - 1)
                return net.query_logits(x, positions)
            return audit.logits(net, x, mode)
    logits = predict(cues, last=True)[:, -1].float()
    hits = logits.argmax(-1).eq(targets[:, 0]).cpu().tolist()
    losses = [audit.nll_rows(predict(b[:, :-1]), b[:, 1:]).double().mean().item() for b in background.split(3)]
    n = len(hits) // 2
    return {"hits": hits, "old_accuracy": sum(hits[:n]) / n, "new_accuracy": sum(hits[n:]) / n,
            "accuracy": sum(hits) / len(hits), "target_nll": F.cross_entropy(logits, targets[:, 0]).item(),
            "base_ppl": math.exp(float(np.mean(losses)))}


def lora_edit(net, data, recipe, seed):
    if recipe["steps"] == 0:
        return edit_metrics(net, data, mode="off")
    edit, values, replay, _ = data
    n = len(edit) // 2
    params = audit.install_lora(net, seed)
    opt = torch.optim.AdamW(params, lr=recipe["lr"], weight_decay=0)
    for wave in range(2):
        low, high = wave*n, (wave+1)*n
        opt.zero_grad(set_to_none=True)
        audit.lora_replay(net, opt, params, edit[low:high], values[low:high], replay,
                         recipe["steps"], (edit[:n], values[:n]) if wave else None,
                         replay_weight=recipe["replay_weight"])
    measured = edit_metrics(net, data, mode="off")
    measured["adapter_parameters"] = sum(p.numel() for p in params)
    return measured


def lora_audit(pair, stream, device, out, *, drift_target=.05, quick=False, deadline=math.inf):
    recipes = [{"lr": lr, "replay_weight": replay, "steps": steps} for lr, replay, steps in (
        (0, .5, 0), (1e-5, .95, 96), (3e-5, .95, 96), (1e-4, .99, 96),
        (1e-4, .5, 48), (3e-4, .5, 48), (6e-4, .5, 48),
        (1e-4, .85, 96), (3e-4, .85, 48), (6e-4, .85, 48), (2e-3, .5, 48)
    )]
    seeds = (44713, 90713, 140713)
    if quick:
        recipes = [{**r, "steps": min(4, r["steps"])} for r in recipes[:4]]
        seeds = seeds[:2]
    slices = torch.tensor_split(stream, len(seeds))
    dev, *tests = [edit_data(x, pair["pulse"], device, seed) for x, seed in zip(slices, seeds, strict=True)]
    result = {"schema": 2, "dev_and_test_seeds": seeds, "ppl_drift_limit": drift_target, "models": {}}
    for name, net in pair.items():
        base = edit_metrics(net, dev, mode="off")
        results = []
        for index, recipe in enumerate(recipes):
            if time.time() >= deadline and index:
                break
            scores = lora_edit(audit.fresh_copy(net), dev, recipe, seeds[0])
            drift = scores["base_ppl"] / base["base_ppl"] - 1.
            results.append({"recipe": recipe, "metrics": scores, "relative_ppl_drift": drift})
            print("LoRA", name, recipe, scores["accuracy"], drift, flush=True)
        index = max((i for i, row in enumerate(results) if row["relative_ppl_drift"] <= drift_target),
                    key=lambda i: (results[i]["metrics"]["accuracy"], -results[i]["metrics"]["target_nll"],
                                   -results[i]["relative_ppl_drift"]))
        recipe = results[index]["recipe"]
        selected = []
        for data, seed in zip(tests, seeds[1:], strict=True):
            baseline = edit_metrics(net, data, mode="off")
            scores = lora_edit(audit.fresh_copy(net), data, recipe, seed)
            selected.append({"seed": seed, "base": baseline, "selected": scores,
                             "relative_ppl_drift": scores["base_ppl"] / baseline["base_ppl"] - 1.})
        result["models"][name] = {"dev": results, "dev_base": base, "selected": recipe,
                                  "test": selected, "grid_complete": len(results) == len(recipes),
                                  "feasible_learned_configs": sum(row["recipe"]["steps"] > 0 and
                                      row["relative_ppl_drift"] <= drift_target for row in results),
                                  "best_unconstrained": max(results, key=lambda row: row["metrics"]["accuracy"])}
        audit.save_json(out / "lora_progress.json", result)
    native_rows = []
    for data, seed in zip(tests, seeds[1:], strict=True):
        edit, values, _, _ = data
        native = audit.fresh_copy(pair["pulse"])
        n = len(edit) // 2
        before = edit_metrics(native, data)
        north = native.teach_fact_online(edit[:n], values[:n])
        south = native.teach_fact_online(edit[n:], values[n:])
        awake = edit_metrics(native, data)
        disabled = edit_metrics(native, data, mode="off")
        native.sleep_consolidate()
        asleep = edit_metrics(native, data, fast=False)
        native_rows.append({"seed": seed, "before": before, "writes": [north, south], "online": awake,
                            "without_memory": disabled, "sleep_fast_off": asleep})
    result["rls"] = native_rows
    return result


def run_checkpoint_audits(run, out, device, deadline):
    out.mkdir(parents=True, exist_ok=True)
    for label, amp, compiled in (("fp32", False, False), ("bf16_compiled", True, True)):
        path = out / f"state_{label}.json"
        if not path.exists() and (device.type == "cuda" or not amp):
            audit.save_json(path, state_audit(device, amp=amp, compile_models=compiled))

    pair, stream, provenance = load_checkpoint_pair(run, device)
    audit.save_json(out / "checkpoint_provenance.json", provenance)
    revised_lora = out / "lora_replay_v2.json"
    lora = json.loads(revised_lora.read_text(encoding="utf-8")) if revised_lora.exists() else {}
    if lora.get("schema") != 2 or not all(k["grid_complete"] for k in lora.get("models", {}).values()):
        audit.save_json(out / "lora_replay_v2.json", lora_audit(pair, stream, device, out, deadline=deadline))
    revised_path = out / "mqar_v2.json"
    revised = json.loads(revised_path.read_text(encoding="utf-8")) if revised_path.exists() else {}
    if not revised.get("short_context_passed") and revised.get("steps", 0) < 16000:
        audit.save_json(revised_path, mqar_train(pair, out, device, deadline=deadline))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--deadline", type=float, default=math.inf)
    args = parser.parse_args()
    torch.manual_seed(42)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_checkpoint_audits(args.legacy_run, args.out, dev, args.deadline)
