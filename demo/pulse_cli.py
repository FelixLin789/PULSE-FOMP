"""Interactive PULSE AI Lab (CLI).

Demonstrates all four PULSE mechanisms live in the terminal:
  - ask <query>                    : Think in continuous concept space with adaptive compute
  - teach <cue> => <answer>        : Inference-time online learning into W_fast (0 backprop, 0 context)
  - sleep                          : Offline Generative Replay consolidation (W_fast -> W_slow)
  - check <premise> | <hypothesis> : Bidirectional free-energy anti-hallucination verification
  - status                         : Inspect W_slow / W_fast norms, episodic buffer, O(1) memory
  - quit                           : Exit

Run interactive mode:
    python demo/pulse_cli.py
Run scripted walkthrough demo:
    python demo/pulse_cli.py --demo
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F

from pulse.config import PULSEConfig
from pulse.model import PULSEModel
from pulse.tokenizer import PulseTokenizer

STARTER_KNOWLEDGE: List[Tuple[str, str]] = [
    ("水的沸点是多少", "100摄氏度"),
    ("地球围绕什么运转", "太阳"),
    ("光速大约是多少", "每秒30万公里"),
    ("三角形有几条边", "三条边"),
    ("PULSE的核心思想", "隐空间能量最小化与突触可塑性"),
    ("Transformer的显存瓶颈", "KV Cache随序列线性膨胀"),
    ("water boils at", "100 celsius"),
    ("earth orbits", "the sun"),
    ("python is a", "programming language"),
    ("plants perform", "photosynthesis"),
    ("two plus two", "four"),
    ("gravity pulls", "masses together"),
]


class PULSELab:
    """Interactive wrapper pairing PULSEModel with a concept manifold prototype index."""

    def __init__(self, model: PULSEModel, tokenizer: PulseTokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_len = 36
        self.known_answers: List[str] = []
        self._answer_embeddings: torch.Tensor | None = None

    @torch.no_grad()
    def register_answer_concepts(self, answers: List[str]) -> None:
        added = False
        for ans in answers:
            if ans not in self.known_answers:
                self.known_answers.append(ans)
                added = True
        if added and self.known_answers:
            toks = self.tokenizer.encode_batch(self.known_answers, max_len=self.max_len)
            self._answer_embeddings = self.model.encode_text(toks).detach()

    @torch.no_grad()
    def bootstrap_local_training(self, epochs: int = 85) -> None:
        cues = self.tokenizer.encode_batch([q for q, _ in STARTER_KNOWLEDGE], max_len=self.max_len)
        ans = self.tokenizer.encode_batch([a for _, a in STARTER_KNOWLEDGE], max_len=self.max_len)
        self.register_answer_concepts([a for _, a in STARTER_KNOWLEDGE])

        for _ in range(epochs):
            self.model.train_causal_concept_local(
                cues, ans, async_order=True, train_decoder=True
            )

    @torch.no_grad()
    def ask(self, query: str) -> str:
        q_tok = self.tokenizer.encode_batch([query], max_len=self.max_len)
        out = self.model.decode_causal_tokens(q_tok, decode_steps=self.model.config.max_decode_tokens)
        z_pred = out["z_pred"]

        steps = int(out["steps_taken"].item())
        e_init = float(out["initial_energy"].item())
        e_final = float(out["final_energy"].item())
        conf = float(out["render_confidence"].item())

        # Concept manifold nearest semantic readout
        manifold_ans = "(unrecognized concept)"
        sim_score = 0.0
        if self._answer_embeddings is not None and len(self.known_answers) > 0:
            sims = F.cosine_similarity(z_pred, self._answer_embeddings, dim=-1)
            best_idx = int(sims.argmax().item())
            sim_score = float(sims[best_idx].item())
            manifold_ans = self.known_answers[best_idx]

        lines = [
            f"  [Latent Thought] steps={steps}/{self.model.config.max_think_steps} | "
            f"Free Energy: {e_init:.4f} -> {e_final:.4f} | Render Conf: {conf*100:.1f}%",
            f"  [Concept Match]  {manifold_ans}  (manifold cos={sim_score:.3f})",
        ]
        return "\n".join(lines)

    @torch.no_grad()
    def teach(self, cue: str, answer: str) -> str:
        c_tok = self.tokenizer.encode_batch([cue], max_len=self.max_len)
        a_tok = self.tokenizer.encode_batch([answer], max_len=self.max_len)
        stats = self.model.teach_online(c_tok, a_tok, repeats=6, adapt_decoder=False)
        self.register_answer_concepts([answer])
        # Explicitly clear working state to prove recall needs zero context window!
        self.model.reset_working_state()
        w_fast_norm = float(self.model.causal_layer.w_fast.norm().item())
        return (
            f"  [Online Plasticity W_fast] Zero-backprop update complete!\n"
            f"  Surprise ||eps||={stats['max_surprise']:.3f} | Gate sigma={stats['max_gate']:.3f} | "
            f"Energy drop={stats['total_energy_drop']:.4f} | ||W_fast||_F={w_fast_norm:.3f}\n"
            f"  (Working context reset to 0 tokens — knowledge now lives in W_fast synapses.)"
        )

    @torch.no_grad()
    def sleep(self) -> str:
        stats = self.model.sleep_consolidate(
            num_replay_samples=48, num_epochs=45, clear_fast_weights=True
        )
        w_fast_norm = float(self.model.causal_layer.w_fast.norm().item())
        return (
            f"  [Sleep Consolidation] Generative Replay complete!\n"
            f"  Consolidated causal episodes={int(stats['causal_consolidated'])}, "
            f"field episodes={int(stats['field_consolidated'])}\n"
            f"  New distillation MSE={stats['causal_new_loss']:.4f} | "
            f"Old replay MSE={stats['causal_replay_loss']:.4f} | "
            f"||W_fast||_F after wipe={w_fast_norm:.6f}"
        )

    @torch.no_grad()
    def check(self, premise: str, hypothesis: str) -> str:
        p_tok = self.tokenizer.encode_batch([premise], max_len=self.max_len)
        h_tok = self.tokenizer.encode_batch([hypothesis], max_len=self.max_len)
        res = self.model.check_consistency(p_tok, h_tok)
        e_val = float(res["consistency_energy"].item())
        cos = float(res["cosine_similarity"].item())
        verdict = (
            "CONSISTENT (Low-Energy Attractor)"
            if e_val < 0.25
            else "CONTRADICTION / HALLUCINATION (High Free Energy)"
        )
        return (
            f"  [Anti-Hallucination Check] Free Energy E={e_val:.4f} | "
            f"Concept Cos={cos:.3f} | Verdict: {verdict}"
        )

    def status(self) -> str:
        w_slow_norm = sum(l.w_slow.norm().item() for l in self.model.field.layers) + self.model.causal_layer.w_slow.norm().item()
        w_fast_norm = sum(l.w_fast.norm().item() for l in self.model.field.layers) + self.model.causal_layer.w_fast.norm().item()
        return (
            f"  Params: {self.model.num_parameters():,} | "
            f"||W_slow||={w_slow_norm:.3f} | ||W_fast||={w_fast_norm:.3f} | "
            f"Buffered episodes: field={len(self.model.fast_weights.episodes)}, "
            f"causal={len(self.model._causal_episodes)} | "
            f"O(1) SSM state dim={self.model.config.ssm_state_dim}"
        )


def build_lab(checkpoint_path: Path, retrain: bool = False) -> PULSELab:
    tok = PulseTokenizer()
    if checkpoint_path.exists() and not retrain:
        model = PULSEModel.load_checkpoint(str(checkpoint_path))
        lab = PULSELab(model, tok)
        lab.register_answer_concepts([a for _, a in STARTER_KNOWLEDGE])
        return lab

    torch.manual_seed(42)
    cfg = PULSEConfig.small()
    cfg.vocab_size = PulseTokenizer.VOCAB_SIZE
    cfg.output_dim = PulseTokenizer.VOCAB_SIZE
    cfg.max_decode_tokens = 36
    cfg.surprise_threshold = 0.08
    cfg.surprise_temperature = 0.03
    cfg.fast_weight_alpha = 0.9
    cfg.fast_weight_decay = 0.0
    cfg.consolidation_lr = 0.14
    cfg.consolidation_ewc_lambda = 0.25
    cfg.weight_lr = 0.025
    cfg.io_lr = 0.04

    model = PULSEModel(cfg)
    lab = PULSELab(model, tok)
    print("Bootstrapping PULSE model via 100% backprop-free Local Energy Learning...")
    lab.bootstrap_local_training(epochs=85)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_checkpoint(str(checkpoint_path))
    print(f"Saved trained checkpoint to {checkpoint_path}")
    return lab


def run_scripted_demo(lab: PULSELab) -> None:
    print("\n=== PULSE Interactive Architecture Walkthrough ===")
    print("\n1) Query pre-trained slow-weight knowledge (W_slow):")
    print("   > ask 水的沸点是多少")
    print(lab.ask("水的沸点是多少"))
    print("   > ask earth orbits")
    print(lab.ask("earth orbits"))

    print("\n2) Anti-Hallucination bidirectional energy check:")
    print("   > check earth orbits | the sun")
    print(lab.check("earth orbits", "the sun"))
    print("   > check earth orbits | photosynthesis")
    print(lab.check("earth orbits", "photosynthesis"))

    print("\n3) Online inference-time rule teaching (W_fast, ZERO backprop, ZERO context):")
    print("   > teach 私有代号OMEGA7 => 启动量子隔离协议")
    print(lab.teach("私有代号OMEGA7", "启动量子隔离协议"))
    print("   > ask 私有代号OMEGA7")
    print(lab.ask("私有代号OMEGA7"))

    print("\n4) Offline Sleep Consolidation (Generative Replay W_fast -> W_slow, then wipe W_fast=0):")
    print("   > sleep")
    print(lab.sleep())

    print("\n5) Post-sleep verification (W_fast is 0! Both old & new rules live permanently in W_slow):")
    print("   > ask 水的沸点是多少")
    print(lab.ask("水的沸点是多少"))
    print("   > ask 私有代号OMEGA7")
    print(lab.ask("私有代号OMEGA7"))

    print("\n6) System Status:")
    print(lab.status())


def main() -> None:
    parser = argparse.ArgumentParser(description="PULSE AI Interactive Lab")
    parser.add_argument("--demo", action="store_true", help="Run scripted non-interactive walkthrough")
    parser.add_argument("--retrain", action="store_true", help="Force fresh backprop-free local training")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/pulse_lab.pt"),
        help="Path to save/load PULSE checkpoint",
    )
    args = parser.parse_args()

    lab = build_lab(args.checkpoint, retrain=args.retrain)
    if args.demo:
        run_scripted_demo(lab)
        return

    print("\nPULSE AI Interactive Lab Ready! Commands:")
    print("  ask <query> | teach <cue> => <ans> | sleep | check <premise> | <hyp> | status | quit\n")
    while True:
        try:
            raw = input("PULSE> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting PULSE Lab.")
            break
        if not raw:
            continue
        if raw.lower() in {"quit", "exit", "q"}:
            break
        if raw.lower() == "status":
            print(lab.status())
        elif raw.lower() == "sleep":
            print(lab.sleep())
        elif raw.lower().startswith("ask "):
            print(lab.ask(raw[4:].strip()))
        elif raw.lower().startswith("teach ") and "=>" in raw:
            body = raw[6:].strip()
            cue, ans = [p.strip() for p in body.split("=>", 1)]
            print(lab.teach(cue, ans))
        elif raw.lower().startswith("check ") and "|" in raw:
            body = raw[6:].strip()
            premise, hyp = [p.strip() for p in body.split("|", 1)]
            print(lab.check(premise, hyp))
        else:
            print(lab.ask(raw))


if __name__ == "__main__":
    main()
