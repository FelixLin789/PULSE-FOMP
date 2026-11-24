"""6-Axis Scientific Benchmark Suite: Transformer vs. PULSE.

Directly evaluates all six architectural paradigm dimensions from the PULSE blueprint:
1. Inference Complexity & Working Memory: O(N)/O(1) vs. O(N^2)/Linear KV Cache
2. Adaptive Compute Depth: Energy-driven latent iterations vs. Fixed depth
3. Inference-Time Online Plasticity: Surprise-gated W_fast rule internalization vs. Frozen weights
4. Lifelong Sleep Consolidation: Generative Replay (W_fast -> W_slow) vs. Catastrophic forgetting
5. Structural Anti-Hallucination: Bidirectional free-energy contradiction detection & revision
6. Local Energy Learning: 100% Backprop-free asynchronous learning vs. Global Backprop
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
import tracemalloc
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from pulse.baseline_transformer import TransformerBaseline, TransformerConfig
from pulse.config import PULSEConfig
from pulse.model import PULSEModel
from pulse.tokenizer import PulseTokenizer


def _make_matched_models(seed: int = 42, output_dim: int = 260) -> tuple[PULSEModel, TransformerBaseline]:
    torch.manual_seed(seed)
    pcfg = PULSEConfig.small()
    pcfg.vocab_size = PulseTokenizer.VOCAB_SIZE
    pcfg.output_dim = output_dim
    pulse = PULSEModel(pcfg)

    tcfg = TransformerConfig(
        vocab_size=PulseTokenizer.VOCAB_SIZE,
        d_model=80,
        n_heads=4,
        n_layers=3,
        d_ff=160,
        max_seq_len=4096,
        output_dim=output_dim,
    )
    tf = TransformerBaseline(tcfg)
    return pulse, tf


def benchmark_1_complexity_and_memory() -> Dict[str, Any]:
    """Axis 1: Sequence latency & working state / KV-cache memory scaling."""
    pulse, tf = _make_matched_models(seed=101)
    pulse.eval()
    tf.eval()

    lengths = [64, 128, 256, 512, 1024, 2048]
    pulse_ms: List[float] = []
    tf_ms: List[float] = []
    pulse_mem_kb: List[float] = []
    tf_kv_kb: List[float] = []

    with torch.no_grad():
        # Warmup
        warm = torch.randint(4, 120, (1, 64))
        pulse.process_sequence(warm, max_steps_per_frame=3)
        tf(warm, use_cache=True)

        for seq_len in lengths:
            tokens = torch.randint(4, 120, (1, seq_len))

            # PULSE: O(N) time, O(1) working memory
            t0 = time.perf_counter()
            p_out = pulse.process_sequence(tokens, max_steps_per_frame=3)
            t1 = time.perf_counter()
            pulse_ms.append((t1 - t0) * 1000.0)
            pulse_mem_kb.append(pulse.working_state_bytes(p_out["working_state"]) / 1024.0)

            # Transformer: O(N^2) self-attention + O(N) KV cache
            t2 = time.perf_counter()
            _, kv = tf(tokens, use_cache=True)
            t3 = time.perf_counter()
            tf_ms.append((t3 - t2) * 1000.0)
            tf_kv_kb.append(tf.kv_cache_bytes(kv) / 1024.0)

        # Extra ultra-long check for PULSE at N = 4096
        long_tokens = torch.randint(4, 120, (1, 4096))
        p_long = pulse.process_sequence(long_tokens, max_steps_per_frame=3)
        pulse_4096_kb = pulse.working_state_bytes(p_long["working_state"]) / 1024.0

    return {
        "pulse_params": pulse.num_parameters(),
        "transformer_params": tf.num_parameters(),
        "lengths": lengths,
        "pulse_ms": pulse_ms,
        "transformer_ms": tf_ms,
        "pulse_state_kb": pulse_mem_kb,
        "transformer_kv_kb": tf_kv_kb,
        "pulse_4096_state_kb": pulse_4096_kb,
        "kv_reduction_at_2048x": tf_kv_kb[-1] / max(pulse_mem_kb[-1], 1e-9),
        "speedup_at_2048x": tf_ms[-1] / max(pulse_ms[-1], 1e-9),
    }


def benchmark_2_adaptive_compute() -> Dict[str, Any]:
    """Axis 2: Native energy-driven adaptive thinking depth vs. rigid Transformer depth."""
    torch.manual_seed(202)
    cfg = PULSEConfig.small()
    cfg.vocab_size = PulseTokenizer.VOCAB_SIZE
    cfg.max_think_steps = 28
    cfg.energy_threshold = 0.04
    cfg.weight_lr = 0.03
    pulse = PULSEModel(cfg)
    tok = PulseTokenizer()

    familiar_texts = [
        "sky is blue",
        "fire is hot",
        "ice is cold",
        "two plus two is four",
        "water flows downhill",
        "sun rises in east",
    ]
    familiar_toks = tok.encode_batch(familiar_texts, max_len=24)

    # Train PULSE locally on familiar concepts so amortized bottom-up settles quickly
    for _ in range(70):
        pulse.train_step_local(familiar_toks)

    novel_texts = [
        "quantum entanglement violates local realism under Bell tests",
        "non-abelian gauge fields induce topological confinement phases",
        "asymptotic homotopy equivalence across simplicial fibrations",
        "stochastic Langevin dynamics near critical saddle manifolds",
        "hyperbolic tessellations preserve exponential volume growth",
        "symplectic integrators conserve modified Hamiltonian invariants",
    ]
    novel_toks = tok.encode_batch(novel_texts, max_len=36)

    with torch.no_grad():
        # Tier 1: Easy / familiar inputs
        easy_res = pulse(familiar_toks)
        # Tier 2: Novel / complex unseen concepts
        novel_res = pulse(novel_toks)
        # Tier 3: Severe contradiction between sensory bottom & opposing top prior
        z_fam = pulse.encode_text(familiar_toks)
        conflict_prior = -easy_res["z_top"] * 2.5
        conflict_res = pulse.field.relax(
            z_bottom=z_fam,
            top_prior=conflict_prior,
            max_steps=cfg.max_think_steps,
            energy_threshold=cfg.energy_threshold,
        )

    return {
        "easy_mean_steps": float(easy_res["steps_taken"].float().mean().item()),
        "novel_mean_steps": float(novel_res["steps_taken"].float().mean().item()),
        "conflict_mean_steps": float(conflict_res.steps_taken.float().mean().item()),
        "easy_final_energy": float(easy_res["final_energy"].mean().item()),
        "conflict_initial_energy": float(conflict_res.initial_energy.mean().item()),
        "conflict_final_energy": float(conflict_res.final_energy.mean().item()),
        "transformer_fixed_layers": 3.0,
    }


def benchmark_3_and_4_online_plasticity_and_sleep() -> Dict[str, Any]:
    """Axes 3 & 4: Inference-time W_fast rule internalization + Sleep Consolidation (W_fast -> W_slow)."""
    torch.manual_seed(303)
    tok = PulseTokenizer()

    base_rules = [
        ("law_gravity", "objects_fall_down"),
        ("law_boiling", "water_100c_steam"),
        ("law_light", "speed_3e8_mps"),
        ("law_prime", "two_is_even_prime"),
        ("law_pi", "ratio_3_14159"),
        ("law_dna", "adenine_pairs_thymine"),
    ]
    private_rules = [
        ("priv_code_alpha", "deploy_cluster_09"),
        ("priv_code_beta", "rotate_vault_key"),
        ("priv_math_zorg", "square_plus_seven"),
        ("priv_flow_omega", "run_canary_first"),
    ]

    all_answers = [a for _, a in base_rules] + [a for _, a in private_rules]
    num_classes = len(all_answers)

    # --- Setup PULSE ---
    pcfg = PULSEConfig.small()
    pcfg.vocab_size = PulseTokenizer.VOCAB_SIZE
    pcfg.output_dim = num_classes
    pcfg.surprise_threshold = 0.05
    pcfg.surprise_temperature = 0.02
    pcfg.fast_weight_alpha = 0.9
    pcfg.fast_weight_decay = 0.0
    pcfg.consolidation_lr = 0.14
    pcfg.consolidation_ewc_lambda = 0.25
    pcfg.weight_lr = 0.025
    pulse = PULSEModel(pcfg)

    base_cues = tok.encode_batch([q for q, _ in base_rules], max_len=24)
    base_ans = tok.encode_batch([a for _, a in base_rules], max_len=24)
    priv_cues = tok.encode_batch([q for q, _ in private_rules], max_len=24)
    priv_ans = tok.encode_batch([a for _, a in private_rules], max_len=24)

    all_ans_tokens = tok.encode_batch(all_answers, max_len=24)

    # Pre-train PULSE purely via local energy learning on Base Rules (Task A)
    for _ in range(55):
        pulse.train_causal_concept_local(base_cues, base_ans)

    with torch.no_grad():
        answer_bank = pulse.encode_text(all_ans_tokens)  # Concept prototypes for exact retrieval eval

        def eval_pulse_retrieval(cues: torch.Tensor, true_ids: List[int], use_fast: bool) -> float:
            pred_z = pulse.predict_causal_concept(cues, use_fast_weights=use_fast)
            sims = F.cosine_similarity(pred_z.unsqueeze(1), answer_bank.unsqueeze(0), dim=-1)
            preds = sims.argmax(dim=-1).tolist()
            return sum(int(p == t) for p, t in zip(preds, true_ids)) / len(true_ids)

    base_ids = list(range(len(base_rules)))
    priv_ids = list(range(len(base_rules), num_classes))

    pulse_base_pre = eval_pulse_retrieval(base_cues, base_ids, use_fast=True)
    pulse_priv_pre = eval_pulse_retrieval(priv_cues, priv_ids, use_fast=True)

    # --- Setup Transformer Baseline (trained with AdamW Backprop on Base Rules) ---
    tcfg = TransformerConfig(
        vocab_size=PulseTokenizer.VOCAB_SIZE,
        d_model=80,
        n_heads=4,
        n_layers=3,
        d_ff=160,
        max_seq_len=64,
        output_dim=num_classes,
    )
    tf = TransformerBaseline(tcfg)
    tf_opt = torch.optim.AdamW(tf.parameters(), lr=3e-3)
    base_y = torch.tensor(base_ids, dtype=torch.long)
    priv_y = torch.tensor(priv_ids, dtype=torch.long)

    for _ in range(90):
        tf_opt.zero_grad()
        loss = F.cross_entropy(tf.classify_last(base_cues), base_y)
        loss.backward()
        tf_opt.step()

    with torch.no_grad():
        tf_base_pre = (tf.classify_last(base_cues).argmax(-1) == base_y).float().mean().item()
        tf_priv_frozen = (tf.classify_last(priv_cues).argmax(-1) == priv_y).float().mean().item()

    # --- Axis 3: Inference-Time Online Rule Teaching (ZERO backprop, ZERO context) ---
    teach_stats = pulse.teach_online(priv_cues, priv_ans, repeats=5)
    pulse.reset_working_state()  # Prove zero context window needed!
    pulse_priv_online = eval_pulse_retrieval(priv_cues, priv_ids, use_fast=True)
    pulse_base_during_online = eval_pulse_retrieval(base_cues, base_ids, use_fast=True)

    # --- Axis 4: Sleep Consolidation (W_fast -> W_slow) vs Naive Transformer Continual Finetuning ---
    sleep_stats = pulse.sleep_consolidate(
        num_replay_samples=40,
        num_epochs=45,
        clear_fast_weights=True,
    )
    # W_fast is now literally zero! Test BOTH base and private rules purely from W_slow!
    pulse_base_post_sleep = eval_pulse_retrieval(base_cues, base_ids, use_fast=False)
    pulse_priv_post_sleep = eval_pulse_retrieval(priv_cues, priv_ids, use_fast=False)

    # Naive continual fine-tuning of Transformer on Private Rules (no old data replay -> forgetting!)
    for _ in range(60):
        tf_opt.zero_grad()
        loss = F.cross_entropy(tf.classify_last(priv_cues), priv_y)
        loss.backward()
        tf_opt.step()

    with torch.no_grad():
        tf_base_after_ft = (tf.classify_last(base_cues).argmax(-1) == base_y).float().mean().item()
        tf_priv_after_ft = (tf.classify_last(priv_cues).argmax(-1) == priv_y).float().mean().item()

    return {
        "axis3_online_plasticity": {
            "pulse_private_before_acc": pulse_priv_pre,
            "pulse_private_after_online_zero_ctx_acc": pulse_priv_online,
            "pulse_base_preserved_during_online_acc": pulse_base_during_online,
            "pulse_surprise_gate": teach_stats["max_gate"],
            "transformer_frozen_private_zero_ctx_acc": tf_priv_frozen,
        },
        "axis4_sleep_consolidation": {
            "pulse_base_before_sleep_acc": pulse_base_pre,
            "pulse_base_after_sleep_wslow_only_acc": pulse_base_post_sleep,
            "pulse_private_after_sleep_wslow_only_acc": pulse_priv_post_sleep,
            "pulse_wfast_norm_after_sleep": float(pulse.causal_layer.w_fast.norm().item()),
            "sleep_consolidated_episodes": sleep_stats["causal_consolidated"],
            "transformer_base_before_ft_acc": tf_base_pre,
            "transformer_base_after_ft_acc": tf_base_after_ft,
            "transformer_private_after_ft_acc": tf_priv_after_ft,
        },
    }


def benchmark_5_anti_hallucination() -> Dict[str, Any]:
    """Axis 5: Structural Anti-Hallucination via Bidirectional Free Energy & Attractor Revision."""
    torch.manual_seed(505)
    tok = PulseTokenizer()
    pcfg = PULSEConfig.small()
    pcfg.vocab_size = PulseTokenizer.VOCAB_SIZE
    pcfg.weight_lr = 0.03
    pulse = PULSEModel(pcfg)

    consistent_pairs = [
        ("earth orbits", "the sun"),
        ("mammals breathe", "oxygen gas"),
        ("triangles have", "three sides"),
        ("python is a", "programming language"),
        ("boiling water", "creates steam"),
        ("magnets attract", "iron metal"),
        ("plants perform", "photosynthesis"),
        ("gravity pulls", "masses together"),
    ]
    hallucinated_pairs = [
        ("earth orbits", "iron metal"),
        ("mammals breathe", "three sides"),
        ("triangles have", "creates steam"),
        ("python is a", "the sun"),
        ("boiling water", "oxygen gas"),
        ("magnets attract", "photosynthesis"),
        ("plants perform", "programming language"),
        ("gravity pulls", "three sides"),
    ]

    p_toks = tok.encode_batch([p for p, _ in consistent_pairs], max_len=20)
    t_toks = tok.encode_batch([c for _, c in consistent_pairs], max_len=20)
    f_toks = tok.encode_batch([c for _, c in hallucinated_pairs], max_len=20)

    # Train PULSE world model purely locally on consistent facts
    for _ in range(55):
        pulse.train_causal_concept_local(p_toks, t_toks)

    cons_out = pulse.check_consistency(p_toks, t_toks)
    hall_out = pulse.check_consistency(p_toks, f_toks)

    e_true = cons_out["consistency_energy"]
    e_false = hall_out["consistency_energy"]

    # Exact pairwise AUROC: fraction of (false, true) pairs where E_false > E_true
    auroc = float((e_false.unsqueeze(1) > e_true.unsqueeze(0)).float().mean().item())

    # Attractor Revision test: corrupt internal hidden states and verify downhill self-correction
    with torch.no_grad():
        z_bottom = pulse.encode_text(p_toks)
        clean_relax = pulse.field.relax(z_bottom)
        corrupted = [z_bottom.clone()] + [
            s + torch.randn_like(s) * 1.1 for s in clean_relax.states[1:]
        ]
        e_corrupt_before, _ = pulse.field.compute_energy_and_errors(corrupted)
        revised = pulse.field.relax(z_bottom, init_states=corrupted, max_steps=28)
        e_corrupt_after = revised.final_energy

    return {
        "mean_consistent_energy": float(e_true.mean().item()),
        "mean_hallucinated_energy": float(e_false.mean().item()),
        "energy_separation_ratio": float((e_false.mean() / e_true.mean().clamp_min(1e-8)).item()),
        "hallucination_detection_auroc": auroc,
        "attractor_revision_energy_before": float(e_corrupt_before.mean().item()),
        "attractor_revision_energy_after": float(e_corrupt_after.mean().item()),
        "attractor_revision_drop_pct": float(
            (1.0 - (e_corrupt_after.mean() / e_corrupt_before.mean().clamp_min(1e-8))).item() * 100.0
        ),
    }


def benchmark_6_local_learning_vs_backprop() -> Dict[str, Any]:
    """Axis 6: Pure Local Energy Learning (Zero Backprop, Async Layers) vs. Global Backprop."""
    torch.manual_seed(606)
    tok = PulseTokenizer()
    num_classes = 10
    texts = [f"domain_{i}_signal_pattern_{i * 17 % 97}" for i in range(num_classes)]
    cues = tok.encode_batch(texts, max_len=28)
    targets = torch.arange(num_classes, dtype=torch.long)

    pulse, tf = _make_matched_models(seed=606, output_dim=num_classes)
    pulse.config.weight_lr = 0.03
    pulse.config.io_lr = 0.04

    # Measure Python/allocator peak delta during 1 training step of PULSE (no_grad) vs Transformer (backprop)
    tracemalloc.start()
    pulse_stats = pulse.train_step_local(cues, target_class=targets, async_order=True)
    _, pulse_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Run full local async training (ZERO .backward() calls)
    for _ in range(65):
        pulse_stats = pulse.train_step_local(cues, target_class=targets, async_order=True)

    pulse_grads_populated = sum(1 for p in pulse.parameters() if p.grad is not None)
    with torch.no_grad():
        pulse_acc = (pulse(cues)["logits"].argmax(-1) == targets).float().mean().item()

    # Transformer trained via standard global Backprop
    tf_opt = torch.optim.AdamW(tf.parameters(), lr=3e-3)
    tracemalloc.start()
    tf_opt.zero_grad()
    loss_tf = F.cross_entropy(tf.classify_last(cues), targets)
    loss_tf.backward()
    tf_opt.step()
    _, tf_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    for _ in range(65):
        tf_opt.zero_grad()
        loss_tf = F.cross_entropy(tf.classify_last(cues), targets)
        loss_tf.backward()
        tf_opt.step()

    tf_grads_populated = sum(1 for p in tf.parameters() if p.grad is not None)
    with torch.no_grad():
        tf_acc = (tf.classify_last(cues).argmax(-1) == targets).float().mean().item()

    return {
        "pulse_async_local_acc": pulse_acc,
        "pulse_final_action_loss": pulse_stats["action_loss"],
        "pulse_autograd_tensors_created": pulse_grads_populated,
        "pulse_step_traced_kb": pulse_peak / 1024.0,
        "transformer_backprop_acc": tf_acc,
        "transformer_autograd_tensors_created": tf_grads_populated,
        "transformer_step_traced_kb": tf_peak / 1024.0,
    }


def plot_benchmark_dashboard(results: Dict[str, Any], out_path: Path) -> None:
    """Render a clean 6-panel scientific comparison figure (Transformer vs. PULSE)."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5), dpi=150)
    fig.suptitle(
        "PULSE (Plastic Latent Dynamics Engine) vs. Transformer — 6-Axis Paradigm Benchmark",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    c_pulse = "#0F766E"
    c_tf = "#C2410C"
    c_accent = "#2563EB"

    # Panel 1: Working Memory / KV Cache Scaling
    ax = axes[0, 0]
    b1 = results["axis1_complexity_memory"]
    ax.plot(b1["lengths"], b1["transformer_kv_kb"], "o-", color=c_tf, lw=2.3, label="Transformer KV Cache O(N)")
    ax.plot(b1["lengths"], b1["pulse_state_kb"], "s-", color=c_pulse, lw=2.3, label="PULSE Fast State O(1)")
    ax.set_title("1. Memory Footprint vs. Sequence Length")
    ax.set_xlabel("Sequence Length N (tokens)")
    ax.set_ylabel("State / Cache Memory (KB)")
    ax.legend(fontsize=8.5)

    # Panel 2: Latency Scaling
    ax = axes[0, 1]
    ax.plot(b1["lengths"], b1["transformer_ms"], "o-", color=c_tf, lw=2.3, label="Transformer O(N^2)")
    ax.plot(b1["lengths"], b1["pulse_ms"], "s-", color=c_pulse, lw=2.3, label="PULSE O(N)")
    ax.set_title("2. Inference Latency Scaling")
    ax.set_xlabel("Sequence Length N (tokens)")
    ax.set_ylabel("Wall-Clock Time (ms)")
    ax.legend(fontsize=8.5)

    # Panel 3: Native Adaptive Compute Depth
    ax = axes[0, 2]
    b2 = results["axis2_adaptive_compute"]
    tiers = ["Easy\n(Familiar)", "Novel\n(Complex)", "Logical\nConflict"]
    p_steps = [b2["easy_mean_steps"], b2["novel_mean_steps"], b2["conflict_mean_steps"]]
    t_steps = [b2["transformer_fixed_layers"]] * 3
    x = np.arange(len(tiers))
    w = 0.35
    ax.bar(x - w / 2, p_steps, w, color=c_pulse, label="PULSE (Energy-Driven)")
    ax.bar(x + w / 2, t_steps, w, color=c_tf, label="Transformer (Fixed Depth)")
    ax.set_xticks(x)
    ax.set_xticklabels(tiers)
    ax.set_ylabel("Internal Thinking Steps / Layers")
    ax.set_title("3. Native Adaptive Compute Depth")
    ax.legend(fontsize=8.5)

    # Panel 4: Inference-Time Online Learning & Sleep Consolidation
    ax = axes[1, 0]
    b3 = results["axis3_online_plasticity"]
    b4 = results["axis4_sleep_consolidation"]
    cats = ["Online New\n(0-Ctx)", "Post-Sleep\nOld Rules", "Post-Sleep\nNew Rules"]
    p_vals = [
        b3["pulse_private_after_online_zero_ctx_acc"] * 100.0,
        b4["pulse_base_after_sleep_wslow_only_acc"] * 100.0,
        b4["pulse_private_after_sleep_wslow_only_acc"] * 100.0,
    ]
    t_vals = [
        b3["transformer_frozen_private_zero_ctx_acc"] * 100.0,
        b4["transformer_base_after_ft_acc"] * 100.0,
        b4["transformer_private_after_ft_acc"] * 100.0,
    ]
    x = np.arange(len(cats))
    ax.bar(x - w / 2, p_vals, w, color=c_pulse, label="PULSE (W_fast + Sleep)")
    ax.bar(x + w / 2, t_vals, w, color=c_tf, label="Transformer (Frozen / Naive FT)")
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Retrieval Accuracy (%)")
    ax.set_title("4. Online Plasticity & Sleep Consolidation")
    ax.legend(fontsize=8.0)

    # Panel 5: Structural Anti-Hallucination Free Energy
    ax = axes[1, 1]
    b5 = results["axis5_anti_hallucination"]
    labels = ["Consistent\nFacts", "Hallucinated\nClaims", "Corrupted\n(Pre-Revise)", "Revised\nAttractor"]
    vals = [
        b5["mean_consistent_energy"],
        b5["mean_hallucinated_energy"],
        b5["attractor_revision_energy_before"],
        b5["attractor_revision_energy_after"],
    ]
    colors = [c_pulse, c_tf, "#9333EA", c_accent]
    ax.bar(labels, vals, color=colors)
    ax.set_ylabel("Variational Free Energy E(z)")
    ax.set_title(f"5. Anti-Hallucination (AUROC = {b5['hallucination_detection_auroc']:.2f})")

    # Panel 6: Local Energy Learning (Zero Backprop)
    ax = axes[1, 2]
    b6 = results["axis6_local_learning"]
    metrics = ["Task Acc (%)", "Autograd Params\nLocked (.grad)"]
    p_m = [b6["pulse_async_local_acc"] * 100.0, float(b6["pulse_autograd_tensors_created"])]
    t_m = [b6["transformer_backprop_acc"] * 100.0, float(b6["transformer_autograd_tensors_created"])]
    x = np.arange(len(metrics))
    ax.bar(x - w / 2, p_m, w, color=c_pulse, label="PULSE (Async Local, 0 Backprop)")
    ax.bar(x + w / 2, t_m, w, color=c_tf, label="Transformer (Global Backprop)")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_title("6. Local Energy Learning vs. Backprop")
    ax.legend(fontsize=8.0)

    plt.tight_layout(rect=(0, 0, 1, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    print("Running 6-Axis PULSE vs. Transformer Scientific Benchmark Suite...")
    t_start = time.perf_counter()

    b1 = benchmark_1_complexity_and_memory()
    print(
        f"[Axis 1] Complexity & Memory: PULSE state = {b1['pulse_state_kb'][-1]:.2f} KB (flat O(1) at N=4096: {b1['pulse_4096_state_kb']:.2f} KB) "
        f"vs Transformer KV = {b1['transformer_kv_kb'][-1]:.1f} KB ({b1['kv_reduction_at_2048x']:.1f}x reduction), "
        f"N=2048 latency PULSE={b1['pulse_ms'][-1]:.1f}ms vs TF={b1['transformer_ms'][-1]:.1f}ms"
    )

    b2 = benchmark_2_adaptive_compute()
    print(
        f"[Axis 2] Adaptive Compute: Easy={b2['easy_mean_steps']:.1f} steps, "
        f"Novel={b2['novel_mean_steps']:.1f} steps, Conflict={b2['conflict_mean_steps']:.1f} steps "
        f"(vs Transformer fixed {b2['transformer_fixed_layers']:.0f} layers)"
    )

    b34 = benchmark_3_and_4_online_plasticity_and_sleep()
    b3 = b34["axis3_online_plasticity"]
    b4 = b34["axis4_sleep_consolidation"]
    print(
        f"[Axis 3] Online W_fast (0-Context): PULSE={b3['pulse_private_after_online_zero_ctx_acc']*100:.1f}% "
        f"(gate={b3['pulse_surprise_gate']:.2f}) vs Frozen Transformer={b3['transformer_frozen_private_zero_ctx_acc']*100:.1f}%"
    )
    print(
        f"[Axis 4] Sleep Consolidation (W_fast=0): PULSE Old={b4['pulse_base_after_sleep_wslow_only_acc']*100:.1f}%, "
        f"New={b4['pulse_private_after_sleep_wslow_only_acc']*100:.1f}% | "
        f"Naive FT Transformer Old={b4['transformer_base_after_ft_acc']*100:.1f}%, New={b4['transformer_private_after_ft_acc']*100:.1f}%"
    )

    b5 = benchmark_5_anti_hallucination()
    print(
        f"[Axis 5] Anti-Hallucination: E_true={b5['mean_consistent_energy']:.4f} vs E_false={b5['mean_hallucinated_energy']:.4f} "
        f"({b5['energy_separation_ratio']:.1f}x gap, AUROC={b5['hallucination_detection_auroc']:.3f}), "
        f"Attractor revision drop={b5['attractor_revision_drop_pct']:.1f}%"
    )

    b6 = benchmark_6_local_learning_vs_backprop()
    print(
        f"[Axis 6] Local Energy Learning: PULSE Async Acc={b6['pulse_async_local_acc']*100:.1f}% "
        f"(autograd .grad params={b6['pulse_autograd_tensors_created']}) vs "
        f"TF Backprop Acc={b6['transformer_backprop_acc']*100:.1f}% "
        f"(autograd .grad params={b6['transformer_autograd_tensors_created']})"
    )

    results = {
        "axis1_complexity_memory": b1,
        "axis2_adaptive_compute": b2,
        "axis3_online_plasticity": b3,
        "axis4_sleep_consolidation": b4,
        "axis5_anti_hallucination": b5,
        "axis6_local_learning": b6,
        "elapsed_seconds": time.perf_counter() - t_start,
    }

    root = Path(__file__).resolve().parent
    json_path = root / "results.json"
    png_path = root / "pulse_vs_transformer.png"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    plot_benchmark_dashboard(results, png_path)
    print(f"\nSaved results to {json_path} and chart to {png_path} ({results['elapsed_seconds']:.2f}s)")


if __name__ == "__main__":
    main()
