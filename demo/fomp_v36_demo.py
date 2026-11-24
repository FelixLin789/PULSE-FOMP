"""Interactive FOMP-v3.6 Live Knowledge Editing Demo (Llama-3-8B-Instruct or CPU --mock).

Usage:
  # Zero-GPU scripted walkthrough (verifies compile / ask / g_ext=0 identity / sign-flip locally)
  PYTHONPATH=src python demo/fomp_v36_demo.py --mock --demo

  # Live interactive editing on Llama-3-8B-Instruct (GPU, loads checkpoints/fomp_v36_llama3_8b.pt)
  PYTHONPATH=src python demo/fomp_v36_demo.py --ckpt checkpoints/fomp_v36_llama3_8b.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from pulse.fomp_v3 import FOMPv3BackboneWrapper, FOMPv3Config, FOMPv3Module  # noqa: E402


class _MockTokenizer:
    bos_token_id = 2
    eos_token_id = 1
    pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        return [4 + (sum(ord(c) * (i + 1) for i, c in enumerate(w)) % 120) for w in text.strip().split()] or [4]

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        return " ".join(f"tok{i}" for i in ids if not (skip_special_tokens and i in (0, 1, 2)))


class _MockLayer(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d, d, bias=False)

    def forward(self, hidden_states: torch.Tensor, **kwargs: Any) -> Tuple[torch.Tensor]:
        return (hidden_states + 0.1 * torch.tanh(self.proj(hidden_states)),)


class _MockBackbone(nn.Module):
    def __init__(self, vocab: int = 128, d: int = 64, n_layers: int = 8) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab, d)
        self.model.layers = nn.ModuleList([_MockLayer(d) for _ in range(n_layers)])
        self.lm_head = nn.Linear(d, vocab, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: Any = None, use_cache: bool = False) -> Dict[str, torch.Tensor]:
        h = self.model.embed_tokens(input_ids)
        for lyr in self.model.layers:
            h = lyr(h)[0]
        return {"logits": self.lm_head(h)}


def _enc_prompt(tokenizer: Any, text: str, prepend_bos: bool) -> List[int]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    bos_id = getattr(tokenizer, "bos_token_id", None) if prepend_bos else None
    if bos_id is not None and (not ids or ids[0] != int(bos_id)):
        return [int(bos_id)] + ids
    return ids


class FOMPLiveSession:
    """Thin stateful session managing instant 1-pass FOMP-v3.6 memory banks on a frozen backbone."""

    def __init__(self, wrap: FOMPv3BackboneWrapper, tokenizer: Any, device: torch.device, amp: bool, prepend_bos: bool) -> None:
        self.wrap = wrap
        self.tok = tokenizer
        self.device = device
        self.amp = amp
        self.prepend_bos = prepend_bos
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        self.edits: List[Tuple[str, str]] = []
        self.compiled: Optional[Dict[str, torch.Tensor]] = None
        self.g_ext: float = 1.0

    def recompile(self) -> float:
        if not self.edits:
            self.compiled = None
            return 0.0
        cues = [_enc_prompt(self.tok, c, self.prepend_bos) for c, _ in self.edits]
        kv = self.wrap.fomp.config.max_tgt_tokens
        tgts = [self.tok.encode(" " + t.strip(), add_special_tokens=False)[:kv] for _, t in self.edits]
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad(), torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.amp):
            self.compiled = self.wrap.compile_token_episodes(cues, tgts, pad_id=self.pad_id, device=self.device, g_ext=self.g_ext)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        return round((time.perf_counter() - t0) * 1000.0, 2)

    def teach(self, cue: str, target: str) -> float:
        self.edits.append((cue.strip(), target.strip()))
        return self.recompile()

    @torch.no_grad()
    def query(self, prompt: str, max_new_tokens: int = 8, sign_flip: bool = False) -> Dict[str, Any]:
        p_ids = _enc_prompt(self.tok, prompt, self.prepend_bos)
        x = torch.tensor([p_ids], device=self.device, dtype=torch.long)
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.amp):
            base_lg = self.wrap.forward_logits(x, compiled=None)["logits"]
            out_ed = self.wrap.forward_logits(x, compiled=self.compiled, sign_flip=sign_flip)
            base_ids = self.wrap.greedy_generate(p_ids, max_new_tokens=max_new_tokens, device=self.device, compiled=None)
            ed_ids = self.wrap.greedy_generate(
                p_ids, max_new_tokens=max_new_tokens, device=self.device, compiled=self.compiled, sign_flip=sign_flip,
            )
        rt = out_ed["routing"]
        return {
            "base_text": self.tok.decode(base_ids, skip_special_tokens=True),
            "edited_text": self.tok.decode(ed_ids, skip_special_tokens=True),
            "max_gate": float(rt["max_gate"][0, -1].item()) if rt else 0.0,
            "d_gate": float(rt["d_gate"][0, -1].item()) if rt else 0.0,
            "max_logit_diff": float((out_ed["logits"].float() - base_lg.float()).abs().max().item()),
        }


def _load_session(args: argparse.Namespace) -> FOMPLiveSession:
    if args.mock:
        torch.manual_seed(42)
        dev = torch.device("cpu")
        tok = _MockTokenizer()
        bb = _MockBackbone().eval()
        cfg = FOMPv3Config(d_model=64, read_layer=4, inject_layers=(4, 7), key_dim=32, ent_dim=32, val_dim=16, max_tgt_tokens=4, top_k=2, rel_hidden=16)
        fomp = FOMPv3Module(cfg).eval()
        return FOMPLiveSession(FOMPv3BackboneWrapper(bb, fomp, freeze_backbone=True), tok, dev, amp=False, prepend_bos=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"FOMP checkpoint not found: {ckpt_path}")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = dev.type == "cuda"
    print(f"[FOMP-v3.6 Demo] Loading {args.model_id} on {dev} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model_id)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if amp else torch.float32
    try:
        bb = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=dtype).to(dev).eval()
    except TypeError:
        bb = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=dtype).to(dev).eval()
    st = torch.load(ckpt_path, map_location=dev, weights_only=False)
    fomp = FOMPv3Module(FOMPv3Config(**st["config"])).to(dev).eval()
    fomp.load_state_dict(st["fomp"], strict=True)
    wrap = FOMPv3BackboneWrapper(bb, fomp, freeze_backbone=True)
    print(f"[FOMP-v3.6 Demo] Ready! Loaded {sum(p.numel() for p in fomp.parameters()):,} adapter params from {ckpt_path.name}.", flush=True)
    return FOMPLiveSession(wrap, tok, dev, amp=amp, prepend_bos=(tok.bos_token_id is not None))


def _run_walkthrough(sess: FOMPLiveSession, max_new: int) -> None:
    print("\n=== FOMP-v3.6 Automated Live Editing Walkthrough ===")
    samples = [
        ("Question: What university did Watts Humphrey attend?\nAnswer:", "University of Michigan"),
        ("The capital of France is", "Lyon"),
    ]
    for cue, tgt in samples:
        ms = sess.teach(cue, tgt)
        print(f"[teach] ({ms:.2f} ms, M={len(sess.edits)}) {cue!r} => {tgt!r}")

    q = samples[0][0]
    res_on = sess.query(q, max_new_tokens=max_new)
    print(f"[ask g_ext=1] base={res_on['base_text']!r} | edited={res_on['edited_text']!r} | gate={res_on['max_gate']:.3f} d_gate={res_on['d_gate']:.3f}")
    assert res_on["max_logit_diff"] > 0.0, "Active FOMP-v3.6 edit should perturb matched query logits!"

    res_flip = sess.query(q, max_new_tokens=max_new, sign_flip=True)
    print(f"[sign_flip=True] edited={res_flip['edited_text']!r} | max_logit_diff={res_flip['max_logit_diff']:.4f}")

    sess.g_ext = 0.0
    sess.recompile()
    res_off = sess.query(q, max_new_tokens=max_new)
    print(f"[ask g_ext=0] base={res_off['base_text']!r} | edited={res_off['edited_text']!r} | max_logit_diff={res_off['max_logit_diff']:.6f}")
    assert res_off["max_logit_diff"] == 0.0 and res_off["edited_text"] == res_off["base_text"], "g_ext=0 MUST be exact bit-for-bit identity!"
    print("=== Walkthrough Self-Check PASSED (0.000000 g_ext=0 identity verified) ===\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Interactive FOMP-v3.6 Zero-Gradient Knowledge Editing Demo")
    ap.add_argument("--ckpt", type=str, default=str(ROOT / "checkpoints/fomp_v36_llama3_8b.pt"))
    ap.add_argument("--model-id", type=str, default="NousResearch/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--mock", action="store_true", help="Run zero-GPU CPU mock model for fast local verification")
    ap.add_argument("--demo", action="store_true", help="Run automated non-interactive walkthrough and exit")
    args = ap.parse_args()

    sess = _load_session(args)
    if args.demo:
        _run_walkthrough(sess, args.max_new_tokens)
        return

    print("Commands: teach <cue> => <target> | ask <prompt> | flip <prompt> | toggle | list | clear | quit")
    while True:
        try:
            raw = input("fomp> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not raw:
            continue
        low = raw.lower()
        if low in ("quit", "exit", "q"):
            break
        if low == "list":
            for i, (c, t) in enumerate(sess.edits):
                print(f"  [{i}] {c!r} => {t!r}")
            print(f"  (total={len(sess.edits)}, g_ext={sess.g_ext})")
        elif low == "clear":
            sess.edits.clear()
            sess.recompile()
            print("  Cleared all compiled edits.")
        elif low == "toggle":
            sess.g_ext = 0.0 if sess.g_ext > 0.0 else 1.0
            ms = sess.recompile()
            print(f"  g_ext={sess.g_ext} (recompiled {len(sess.edits)} edits in {ms:.2f} ms)")
        elif low.startswith("teach "):
            body = raw[6:]
            if "=>" not in body:
                print("  Usage: teach <cue> => <new_target>")
                continue
            cue, tgt = body.split("=>", 1)
            ms = sess.teach(cue, tgt)
            print(f"  Compiled M={len(sess.edits)} edits in {ms:.2f} ms (0 gradients, 0 base weight updates).")
        elif low.startswith("ask ") or low.startswith("flip "):
            flip = low.startswith("flip ")
            prompt = raw[5:].strip() if flip else raw[4:].strip()
            r = sess.query(prompt, max_new_tokens=args.max_new_tokens, sign_flip=flip)
            print(f"  Base  : {r['base_text']!r}")
            print(f"  Edited: {r['edited_text']!r}  [gate={r['max_gate']:.3f}, d_gate={r['d_gate']:.3f}, flip={flip}]")
        else:
            print("  Unknown command. Use: teach <cue> => <target> | ask <prompt> | flip <prompt> | toggle | list | clear | quit")


if __name__ == "__main__":
    main()
