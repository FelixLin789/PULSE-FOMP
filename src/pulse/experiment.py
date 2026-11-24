"""Colab-Resilient 200-CCU Scientific Controlled Experiment Runner: PULSE-LM vs. Matched Transformer.

Features:
- Automatic GPU detection (H100 / A100 / L4 / V100 / T4 / CPU) & CCU budget guardrails.
- Atomic Google-Drive-safe checkpointing and seamless auto-resume across Colab disconnects.
- Deterministic shared token block cache ensuring identical data order for PULSE and Transformer.
- Full 6-axis GPU scientific comparison suite, Matplotlib publication plots, and Markdown report.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import gc
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pulse.pulse_lm import PULSELMConfig, PULSELanguageModel
from pulse.pulse_v2 import PULSEv2Config, PULSEv2Model
from pulse.pulse_v3 import PULSEv3Config, PULSEv3Model
from pulse.pulse_v4 import PULSEv4Config, PULSEv4Model
from pulse.scalable_transformer import ScalableTransformerConfig, ScalableTransformerLM
from pulse.tokenizer import PulseTokenizer

_ALL_PULSE_CLASSES = (PULSELanguageModel, PULSEv2Model, PULSEv3Model, PULSEv4Model)


def _newton_schulz_batched(grads: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Batched Quintic Newton-Schulz orthogonalization (Muon) for [K, M, N] 2D hidden matrices (BF16 on CUDA)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    work_dtype = torch.bfloat16 if (grads.is_cuda and torch.cuda.is_bf16_supported()) else torch.float32
    x = grads.to(work_dtype)
    transposed = x.shape[-2] > x.shape[-1]
    if transposed:
        x = x.transpose(-1, -2)
    norms = x.flatten(1).norm(dim=-1).view(-1, 1, 1) + 1e-7
    x = x / norms
    for _ in range(steps):
        a_mat = torch.bmm(x, x.transpose(-1, -2))
        b_mat = b * a_mat + c * torch.bmm(a_mat, a_mat)
        x = a * x + torch.bmm(b_mat, x)
    if transposed:
        x = x.transpose(-1, -2)
    return x.to(grads.dtype)


class FairMuonAdamW(torch.optim.Optimizer):
    """Fair Batched-BF16 Muon + Vectorized AdamW hybrid optimizer (applied identically to BOTH
    PULSE and Transformer: Newton-Schulz Muon on homogeneous 2D hidden weight matrices, AdamW on
    embeddings/1D norms/biases/tiny gate projections)."""

    def __init__(
        self,
        params: List[torch.nn.Parameter],
        lr: float = 6e-4,
        muon_lr_mult: float = 12.0,
        adam_lr_mult: float = 2.5,
        momentum: float = 0.95,
        betas: Tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.1,
    ) -> None:
        muon_params: List[torch.nn.Parameter] = []
        adam_params: List[torch.nn.Parameter] = []
        for p in params:
            if not p.requires_grad:
                continue
            if getattr(p, "_no_muon", False):
                adam_params.append(p)
            elif p.ndim == 2 and max(p.shape) <= 8192 and min(p.shape) >= 16:
                muon_params.append(p)
            else:
                adam_params.append(p)
        defaults = dict(
            lr=lr,
            muon_lr_mult=muon_lr_mult,
            adam_lr_mult=adam_lr_mult,
            momentum=momentum,
            betas=betas,
            weight_decay=weight_decay,
            eps=1e-8,
        )
        groups = [
            {"params": muon_params, "is_muon": True},
            {"params": adam_params, "is_muon": False},
        ]
        super().__init__(groups, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Any] = None) -> Optional[Any]:
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            if group["is_muon"]:
                m_lr = lr * group["muon_lr_mult"]
                beta = group["momentum"]
                # Group identical-shape 2D matrices for single-launch batched bmm Newton-Schulz!
                buckets: Dict[Tuple[int, int], List[torch.nn.Parameter]] = {}
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    buckets.setdefault((p.shape[0], p.shape[1]), []).append(p)
                for (rows, cols), plist in buckets.items():
                    g_list = [p.grad for p in plist]
                    b_list = []
                    for p in plist:
                        st = self.state[p]
                        if "buf" not in st:
                            st["buf"] = torch.zeros_like(p)
                        b_list.append(st["buf"])
                    torch._foreach_mul_(b_list, beta)
                    torch._foreach_add_(b_list, g_list)
                    nesterov = torch._foreach_add(g_list, b_list, alpha=beta)
                    stacked = torch.stack(nesterov, dim=0)
                    ortho = _newton_schulz_batched(stacked, steps=5)
                    scale = max(1.0, rows / max(cols, 1)) ** 0.5
                    if wd > 0.0:
                        torch._foreach_mul_(plist, 1.0 - lr * wd)
                    torch._foreach_add_(plist, list(ortho.unbind(0)), alpha=-m_lr * scale)
            else:
                a_lr = lr * float(group.get("adam_lr_mult", 2.5))
                b1, b2 = group["betas"]
                eps = group["eps"]
                p_list, g_list, m_list, v_list, p_wd = [], [], [], [], []
                step_val = 0
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    st = self.state[p]
                    if "step" not in st:
                        st["step"] = 0
                        st["exp_avg"] = torch.zeros_like(p)
                        st["exp_avg_sq"] = torch.zeros_like(p)
                    st["step"] += 1
                    step_val = st["step"]
                    p_list.append(p)
                    g_list.append(p.grad)
                    m_list.append(st["exp_avg"])
                    v_list.append(st["exp_avg_sq"])
                    if wd > 0.0 and p.ndim == 2 and not getattr(p, "_no_muon", False):
                        p_wd.append(p)
                if p_list:
                    bc1 = 1.0 - b1 ** step_val
                    bc2 = 1.0 - b2 ** step_val
                    if p_wd:
                        torch._foreach_mul_(p_wd, 1.0 - lr * wd)
                    torch._foreach_mul_(m_list, b1)
                    torch._foreach_add_(m_list, g_list, alpha=1.0 - b1)
                    torch._foreach_mul_(v_list, b2)
                    torch._foreach_addcmul_(v_list, g_list, g_list, value=1.0 - b2)
                    denom = torch._foreach_sqrt(v_list)
                    torch._foreach_div_(denom, math.sqrt(bc2))
                    torch._foreach_add_(denom, eps)
                    torch._foreach_addcdiv_(p_list, m_list, denom, value=-a_lr / bc1)
        return loss

# Conservative Colab Compute Unit hourly burn estimates (used to protect user's 200 CCU budget)
_GPU_CCU_RATES: List[Tuple[str, float]] = [
    ("H100", 15.0),
    ("A100", 11.8),
    ("L4", 4.8),
    ("V100", 5.5),
    ("T4", 2.0),
]

SYNTHETIC_STORIES_AND_LOGIC: List[str] = [
    "Once upon a time, a little fox found a shiny blue pebble near the quiet river. She carried it home to show her mother, and they smiled together under the warm sun.",
    "Tom built a small wooden boat and placed it gently on the pond. The soft wind pushed the boat across the clear water until it reached the green grass on the other side.",
    "Maya loved reading books about stars and planets. Every clear night she looked through her telescope and drew careful maps of the bright constellations in her notebook.",
    "Fact: water boils at one hundred degrees Celsius at standard sea level pressure and turns into steam.",
    "Fact: the Earth orbits around the Sun once every three hundred and sixty five days.",
    "Fact: every triangle has three straight sides and its interior angles always sum to one hundred and eighty degrees.",
    "Rule: when an ice cube is heated above zero degrees Celsius, it melts smoothly into liquid water.",
    "Rule: if the traffic light turns red, every careful driver stops the car before the white line.",
    "Lily planted three tiny sunflower seeds in a clay pot. She watered them each morning, and after two weeks tall green sprouts reached toward the bright window.",
    "Ben lost his red kite in the tall oak tree. His older sister brought a long wooden pole, gently freed the string, and handed the kite back with a cheerful laugh.",
    "Fact: green plants use sunlight, water, and carbon dioxide during photosynthesis to produce oxygen and sugar.",
    "Fact: magnets attract iron and steel objects because magnetic fields align microscopic domains inside the metal.",
    "Rule: if all birds have feathers and a sparrow is a bird, then a sparrow definitely has feathers.",
    "Rule: when a battery circuit is broken by an open switch, electric current stops flowing through the lamp.",
    "A friendly puppy chased a yellow butterfly across the sunny meadow. When the butterfly landed on a flower, the puppy sat still and watched in quiet wonder.",
    "Sam baked a warm apple pie with cinnamon and honey. The sweet aroma filled the whole kitchen and brought the whole family to the wooden table.",
]

ONLINE_TEST_FACTS: List[Tuple[str, str]] = [
    ("Private protocol code ZETA9 triggers", " immediate quantum isolation mode."),
    ("Custom math operator zorg(x) computes", " x squared plus seventeen."),
    ("Internal service alias ORION4 routes", " traffic to backup vault nine."),
    ("Laboratory alloy KRYPTON7 melts at", " exactly nine hundred kelvin."),
    ("Project codename HELIOS2 deploys", " canary nodes before global rollout."),
    ("Secret botanical species LUNA5 blooms", " only under violet moonlight."),
    ("Private cipher rule DELTA3 shifts", " every odd byte by eleven."),
    ("Archival vault index SIGMA8 stores", " verified lunar seismograph logs."),
]


class CCUBudgetGuard:
    """Tracks real wall-clock GPU consumption against a strict Colab CCU budget across resumes."""

    def __init__(
        self,
        total_ccu_budget: float = 180.0,
        gpu_name: Optional[str] = None,
        phase_spent: Optional[Dict[str, float]] = None,
    ) -> None:
        self.total_ccu_budget = float(total_ccu_budget)
        if gpu_name is None:
            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        self.gpu_name = gpu_name
        self.ccu_per_hour = self._estimate_rate(gpu_name)
        self.phase_spent: Dict[str, float] = dict(phase_spent or {})

    @staticmethod
    def _estimate_rate(gpu_name: str) -> float:
        upper = gpu_name.upper()
        for key, rate in _GPU_CCU_RATES:
            if key in upper:
                return rate
        return 0.1  # CPU fallback

    def record_elapsed_seconds(self, phase: str, seconds: float) -> float:
        delta_ccu = (max(seconds, 0.0) / 3600.0) * self.ccu_per_hour
        self.phase_spent[phase] = self.phase_spent.get(phase, 0.0) + delta_ccu
        return delta_ccu

    def record_spent_ccu(self, phase: str, ccu: float) -> None:
        self.phase_spent[phase] = self.phase_spent.get(phase, 0.0) + max(float(ccu), 0.0)

    def total_spent_ccu(self) -> float:
        return sum(self.phase_spent.values())

    def remaining_ccu(self) -> float:
        return max(self.total_ccu_budget - self.total_spent_ccu(), 0.0)

    def phase_spent_ccu(self, phase: str) -> float:
        return self.phase_spent.get(phase, 0.0)

    def phase_exceeded(self, phase: str, phase_fraction: float = 0.45) -> bool:
        return (
            self.phase_spent_ccu(phase) >= self.total_ccu_budget * phase_fraction
            or self.remaining_ccu() <= 0.0
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_ccu_budget": self.total_ccu_budget,
            "gpu_name": self.gpu_name,
            "ccu_per_hour": self.ccu_per_hour,
            "phase_spent": dict(self.phase_spent),
            "total_spent_ccu": self.total_spent_ccu(),
            "remaining_ccu": self.remaining_ccu(),
        }


class ExperimentTokenizer:
    """Unified tokenizer wrapping GPT-2 BPE (for 30M Colab runs) or PulseTokenizer (offline micro)."""

    def __init__(self, mode: str = "byte") -> None:
        self.mode = mode
        self._tiktoken = None
        self._hf_tok = None
        self._byte_tok = PulseTokenizer()
        if mode == "gpt2":
            try:
                import tiktoken

                self._tiktoken = tiktoken.get_encoding("gpt2")
                self.vocab_size = self._tiktoken.n_vocab
                self.eos_id = self._tiktoken.eot_token
                return
            except Exception:
                pass
            try:
                from transformers import AutoTokenizer

                self._hf_tok = AutoTokenizer.from_pretrained("gpt2")
                self.vocab_size = int(self._hf_tok.vocab_size)
                self.eos_id = int(self._hf_tok.eos_token_id)
                return
            except Exception:
                # Graceful offline fallback
                self.mode = "byte"

        self.vocab_size = self._byte_tok.VOCAB_SIZE
        self.eos_id = self._byte_tok.EOS_ID

    def encode(self, text: str) -> List[int]:
        if self._tiktoken is not None:
            return self._tiktoken.encode(text, allowed_special={"<|endoftext|>"})
        if self._hf_tok is not None:
            return list(self._hf_tok.encode(text, add_special_tokens=False))
        return self._byte_tok.encode(text, add_special_tokens=False)

    def decode(self, ids: List[int]) -> str:
        if self._tiktoken is not None:
            return self._tiktoken.decode([int(i) for i in ids if 0 <= int(i) < self.vocab_size])
        if self._hf_tok is not None:
            return str(self._hf_tok.decode([int(i) for i in ids if 0 <= int(i) < self.vocab_size]))
        return self._byte_tok.decode(ids)


@dataclass
class ExperimentConfig:
    """Full configuration for a matched PULSE vs. Transformer scientific controlled experiment."""

    output_dir: str = "experiment_outputs/pulse_vs_transformer"
    preset: str = "colab_30m"  # "micro_smoke" | "colab_30m" | "colab_100m"
    pulse_version: str = "v4"  # "v4" | "v3" | "v2" | "v1"
    optimizer_type: str = "adamw"  # "adamw" | "muon" (applied identically to BOTH models!)
    dataset: str = "tinystories"  # "synthetic" | "tinystories" | "fineweb_edu"
    tokenizer_mode: str = "gpt2"  # "byte" | "gpt2"
    seed: int = 42

    # Training schedule (matched identically across PULSE and Transformer)
    seq_len: int = 512
    batch_size: int = 16
    grad_accum_steps: int = 2
    max_steps: int = 12000
    lr: float = 6e-4
    min_lr_ratio: float = 0.1
    warmup_ratio: float = 0.05
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    async_core_updates: bool = True
    compile_models: bool = True

    # Data cache sizing
    max_train_stories: int = 200_000
    max_val_stories: int = 2_000

    # Checkpointing, logging, and CCU budget guard
    total_ccu_budget: float = 160.0
    phase_budget_fraction: float = 0.45
    checkpoint_every: int = 250
    eval_every: int = 250
    log_every: int = 25
    eval_batches: int = 12
    eval_seq_lengths: Tuple[int, ...] = (256, 512, 1024, 2048, 4096)

    @classmethod
    def micro_smoke(cls, output_dir: str = "experiment_outputs/micro_smoke") -> ExperimentConfig:
        """Ultra-fast CPU/GPU smoke preset (< 10 seconds on CPU) validating the full pipeline."""
        return cls(
            output_dir=output_dir,
            preset="micro_smoke",
            pulse_version="v4",
            dataset="synthetic",
            tokenizer_mode="byte",
            seq_len=32,
            batch_size=4,
            grad_accum_steps=1,
            max_steps=10,
            lr=3e-3,
            max_train_stories=128,
            max_val_stories=32,
            total_ccu_budget=5.0,
            checkpoint_every=5,
            eval_every=5,
            log_every=2,
            eval_batches=4,
            eval_seq_lengths=(32, 64, 128),
            compile_models=False,
        )

    @classmethod
    def for_colab_200ccu(
        cls,
        output_dir: str = "/content/drive/MyDrive/Colab Notebooks/PULSE_AI/runs/exp_pulse_v4_30m",
        total_ccu_budget: float = 150.0,
        smoke: bool = False,
        micro_bench: bool = False,
        pulse_version: str = "v4",
        preset: str = "colab_30m",
        dataset: str = "tinystories",
        optimizer_type: str = "adamw",
    ) -> ExperimentConfig:
        """Auto-adapts batch size, sequence length, and steps to the active Colab GPU VRAM."""
        if smoke:
            return cls(
                output_dir=output_dir,
                preset=preset,
                pulse_version=pulse_version,
                optimizer_type=optimizer_type,
                dataset=dataset,
                tokenizer_mode="gpt2",
                seq_len=512,
                batch_size=8,
                grad_accum_steps=1,
                max_steps=40,
                lr=6e-4,
                max_train_stories=4000,
                max_val_stories=400,
                total_ccu_budget=5.0,
                checkpoint_every=20,
                eval_every=20,
                log_every=10,
                eval_batches=6,
                eval_seq_lengths=(256, 512, 1024, 2048, 4096),
            )

        vram_gb = 0.0
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024.0**3)

        if preset == "colab_100m":
            if vram_gb >= 36.0:
                batch_size, accum, seq_len, max_steps = 16, 4, 512, 16000
                eval_lengths: Tuple[int, ...] = (256, 512, 1024, 2048, 4096, 8192)
            else:
                batch_size, accum, seq_len, max_steps = 8, 8, 512, 12000
                eval_lengths = (256, 512, 1024, 2048, 4096)
        elif vram_gb >= 36.0:  # A100 40GB/80GB or H100
            batch_size, accum, seq_len, max_steps = 32, 1, 512, 14000
            eval_lengths = (256, 512, 1024, 2048, 4096, 8192)
        elif vram_gb >= 20.0:  # L4 24GB
            batch_size, accum, seq_len, max_steps = 16, 2, 512, 12000
            eval_lengths = (256, 512, 1024, 2048, 4096, 8192)
        else:  # T4 / V100 16GB or CPU
            batch_size, accum, seq_len, max_steps = 8, 4, 256, 10000
            eval_lengths = (256, 512, 1024, 2048)

        if micro_bench:
            # Step 2: ~2 CCU 2,000-step wall-clock / VRAM / PPL gate verification
            max_steps = 2000
            total_ccu_budget = min(total_ccu_budget, 12.0)

        return cls(
            output_dir=output_dir,
            preset=preset,
            pulse_version=pulse_version,
            optimizer_type=optimizer_type,
            dataset=dataset,
            tokenizer_mode="gpt2",
            seq_len=seq_len,
            batch_size=batch_size,
            grad_accum_steps=accum,
            max_steps=max_steps,
            max_train_stories=200_000,
            max_val_stories=2_000,
            total_ccu_budget=total_ccu_budget,
            eval_seq_lengths=eval_lengths,
        )


def _atomic_torch_save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _cosine_lr_multiplier(step: int, total_steps: int, warmup_steps: int, min_ratio: float) -> float:
    if step < max(warmup_steps, 1):
        return float(step + 1) / float(max(warmup_steps, 1))
    progress = min(
        max((step - warmup_steps) / max(total_steps - warmup_steps, 1), 0.0),
        1.0,
    )
    return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * progress))


class ControlledExperimentRunner:
    """Runs a reproducible, resume-safe PULSE-LM vs. Matched Transformer controlled experiment."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.out_dir = Path(config.output_dir)
        self.ckpt_dir = self.out_dir / "checkpoints"
        self.cache_dir = self.out_dir / "data_cache"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda" and hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        if hasattr(torch, "_dynamo"):
            torch._dynamo.config.cache_size_limit = max(getattr(torch._dynamo.config, "cache_size_limit", 8), 64)
        self.use_bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
        self.tokenizer = ExperimentTokenizer(mode=config.tokenizer_mode)
        self.state_path = self.out_dir / "experiment_state.json"
        self.state = self._load_or_init_state()
        self.budget = CCUBudgetGuard(
            total_ccu_budget=config.total_ccu_budget,
            phase_spent=self.state.get("phase_spent", {}),
        )
        self._train_blocks: Optional[torch.Tensor] = None
        self._val_blocks: Optional[torch.Tensor] = None
        self._val_stream: Optional[torch.Tensor] = None
        self._val_stream_non_repeating: bool = False

    def _load_or_init_state(self) -> Dict[str, Any]:
        if self.state_path.exists():
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        return {
            "phase_spent": {},
            "pulse_steps": 0,
            "transformer_steps": 0,
            "pulse_history": [],
            "transformer_history": [],
        }

    def _save_state(self) -> None:
        self.state["phase_spent"] = dict(self.budget.phase_spent)
        tmp = self.state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.state, indent=2), encoding="utf-8")
        os.replace(tmp, self.state_path)

    def _build_models_configs(self) -> Tuple[Any, ScalableTransformerConfig]:
        if self.config.pulse_version == "v4":
            if self.config.preset == "micro_smoke":
                pcfg: Any = PULSEv4Config.micro_test()
            elif self.config.preset == "colab_100m":
                pcfg = PULSEv4Config.colab_100m()
            else:
                pcfg = PULSEv4Config.colab_30m()
        elif self.config.pulse_version == "v3":
            if self.config.preset == "micro_smoke":
                pcfg = PULSEv3Config.micro_test()
            elif self.config.preset == "colab_100m":
                pcfg = PULSEv3Config.colab_100m()
            else:
                pcfg = PULSEv3Config.colab_30m()
        elif self.config.pulse_version == "v2":
            pcfg = (
                PULSEv2Config.micro_test()
                if self.config.preset == "micro_smoke"
                else PULSEv2Config.colab_30m()
            )
        else:
            pcfg = (
                PULSELMConfig.micro_test()
                if self.config.preset == "micro_smoke"
                else PULSELMConfig.colab_30m()
            )

        if self.config.preset == "micro_smoke":
            tcfg = ScalableTransformerConfig.micro_test()
        elif self.config.preset == "colab_100m":
            tcfg = ScalableTransformerConfig.colab_100m()
        else:
            tcfg = ScalableTransformerConfig.colab_30m()

        pcfg.vocab_size = self.tokenizer.vocab_size
        tcfg.vocab_size = self.tokenizer.vocab_size
        return pcfg, tcfg

    def _instantiate_pulse(self, pcfg: Any) -> torch.nn.Module:
        if self.config.pulse_version == "v4":
            return PULSEv4Model(pcfg).to(self.device)
        if self.config.pulse_version == "v3":
            return PULSEv3Model(pcfg).to(self.device)
        if self.config.pulse_version == "v2":
            return PULSEv2Model(pcfg).to(self.device)
        return PULSELanguageModel(pcfg).to(self.device)

    def _build_optimizer(self, params: Any) -> torch.optim.Optimizer:
        if self.config.optimizer_type == "muon":
            flat_params = (
                [p for g in params for p in g["params"]]
                if (isinstance(params, list) and params and isinstance(params[0], dict))
                else params
            )
            return FairMuonAdamW(
                flat_params,
                lr=self.config.lr,
                betas=(0.9, 0.95),
                weight_decay=self.config.weight_decay,
            )
        if self.device.type == "cuda":
            try:
                return torch.optim.AdamW(
                    params,
                    lr=self.config.lr,
                    betas=(0.9, 0.95),
                    weight_decay=self.config.weight_decay,
                    fused=True,
                )
            except Exception:
                pass
        return torch.optim.AdamW(
            params,
            lr=self.config.lr,
            betas=(0.9, 0.95),
            weight_decay=self.config.weight_decay,
        )

    def _prepare_shared_corpus(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build or load identical cached token blocks [N_blocks, seq_len+1] and pure non-repeating
        1D validation stream for strictly uncontaminated length extrapolation."""
        if self._train_blocks is not None and self._val_blocks is not None:
            return self._train_blocks, self._val_blocks

        train_cache = self.cache_dir / f"train_pure_{self.config.dataset}_{self.config.seq_len}.pt"
        val_cache = self.cache_dir / f"val_pure_{self.config.dataset}_{self.config.seq_len}.pt"
        stream_cache = self.cache_dir / f"val_stream_{self.config.dataset}.pt"

        if train_cache.exists() and val_cache.exists() and stream_cache.exists():
            self._train_blocks = torch.load(train_cache, map_location="cpu", weights_only=True)
            self._val_blocks = torch.load(val_cache, map_location="cpu", weights_only=True)
            self._val_stream = torch.load(stream_cache, map_location="cpu", weights_only=True)
            self._val_stream_non_repeating = self.config.dataset != "synthetic"
            return self._train_blocks, self._val_blocks

        train_texts: List[str] = []
        val_texts: List[str] = []
        used_real_corpus = False

        if self.config.dataset == "tinystories":
            try:
                from datasets import load_dataset

                print(
                    f"Streaming PURE TinyStories ({self.config.max_train_stories:,} train / "
                    f"{self.config.max_val_stories:,} val, zero synthetic repetition)..."
                )
                ds_train = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
                for idx, row in enumerate(ds_train):
                    if idx >= self.config.max_train_stories:
                        break
                    txt = row.get("text", "").strip()
                    if txt:
                        train_texts.append(txt)
                ds_val = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)
                for idx, row in enumerate(ds_val):
                    if idx >= self.config.max_val_stories:
                        break
                    txt = row.get("text", "").strip()
                    if txt:
                        val_texts.append(txt)
                used_real_corpus = bool(train_texts and val_texts)
            except Exception as exc:
                print(f"[Info] HuggingFace TinyStories unavailable ({exc}); using synthetic corpus.")
        elif self.config.dataset == "fineweb_edu":
            try:
                from datasets import load_dataset

                print(
                    f"Streaming PURE FineWeb-Edu ({self.config.max_train_stories:,} train / "
                    f"{self.config.max_val_stories:,} held-out val docs)..."
                )
                ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
                total_need = self.config.max_val_stories + self.config.max_train_stories
                for idx, row in enumerate(ds):
                    if idx >= total_need:
                        break
                    txt = row.get("text", "").strip()
                    if not txt:
                        continue
                    if idx < self.config.max_val_stories:
                        val_texts.append(txt)
                    else:
                        train_texts.append(txt)
                used_real_corpus = bool(train_texts and val_texts)
            except Exception as exc:
                print(f"[Info] HuggingFace FineWeb-Edu unavailable ({exc}); using synthetic corpus.")

        if not train_texts:
            repeats_tr = max(self.config.max_train_stories // len(SYNTHETIC_STORIES_AND_LOGIC), 16)
            repeats_va = max(self.config.max_val_stories // len(SYNTHETIC_STORIES_AND_LOGIC), 4)
            train_texts = SYNTHETIC_STORIES_AND_LOGIC * repeats_tr
            val_texts = SYNTHETIC_STORIES_AND_LOGIC * repeats_va

        def encode_stream(texts: List[str]) -> List[int]:
            stream: List[int] = []
            eos = self.tokenizer.eos_id
            for t in texts:
                stream.extend(self.tokenizer.encode(t))
                stream.append(eos)
            return stream

        def pack_blocks(stream_in: List[int], block_len: int) -> torch.Tensor:
            stream = list(stream_in)
            min_needed = block_len * max(self.config.batch_size * 2, 8)
            while len(stream) < min_needed:
                stream.extend(stream[: min_needed - len(stream)])
            n_blocks = len(stream) // block_len
            arr = np.asarray(stream[: n_blocks * block_len], dtype=np.int32)
            return torch.from_numpy(arr.reshape(n_blocks, block_len))

        train_stream = encode_stream(train_texts)
        val_stream_raw = encode_stream(val_texts)

        block_len = self.config.seq_len + 1
        train_blocks = pack_blocks(train_stream, block_len)
        val_blocks = pack_blocks(val_stream_raw, block_len)
        val_stream_tensor = torch.from_numpy(np.asarray(val_stream_raw, dtype=np.int32))

        _atomic_torch_save(train_blocks, train_cache)
        _atomic_torch_save(val_blocks, val_cache)
        _atomic_torch_save(val_stream_tensor, stream_cache)
        self._train_blocks = train_blocks
        self._val_blocks = val_blocks
        self._val_stream = val_stream_tensor
        self._val_stream_non_repeating = used_real_corpus
        print(
            f"Cached PURE shared token stream: train={train_blocks.shape[0]:,} blocks "
            f"({train_blocks.numel():,} tokens), val={val_blocks.shape[0]:,} blocks, "
            f"val_stream={val_stream_tensor.numel():,} continuous tokens (non_repeating={used_real_corpus})"
        )
        return train_blocks, val_blocks

    def _sample_deterministic_batch(
        self, blocks: torch.Tensor, step_idx: int, batch_size: int
    ) -> torch.Tensor:
        """Deterministic permutation per epoch so PULSE & Transformer see IDENTICAL step-by-step data."""
        n = blocks.shape[0]
        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.config.seed * 100_003 + step_idx)
        indices = torch.randint(0, n, (batch_size,), generator=gen)
        return blocks.index_select(0, indices).long().to(self.device, non_blocking=True)

    @torch.no_grad()
    def _evaluate_val_ppl(
        self, model: torch.nn.Module, model_kind: str, val_blocks: torch.Tensor
    ) -> Dict[str, float]:
        model.eval()
        losses: List[float] = []
        steps_list: List[float] = []
        energies_list: List[float] = []

        for b_idx in range(self.config.eval_batches):
            batch = self._sample_deterministic_batch(
                val_blocks, step_idx=900_000 + b_idx, batch_size=self.config.batch_size
            )
            with torch.amp.autocast(
                device_type="cuda" if self.device.type == "cuda" else "cpu",
                dtype=torch.bfloat16,
                enabled=self.use_bf16,
            ):
                if model_kind == "pulse":
                    out = (
                        model(
                            batch[:, :-1],
                            use_fast_weights=False,
                            return_state=False,
                            run_core=(b_idx == 0),
                        )
                        if isinstance(model, PULSEv4Model)
                        else model(batch[:, :-1], use_fast_weights=False)
                    )
                    logits = out["logits"]
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]), batch[:, 1:].reshape(-1)
                    )
                    if out["steps_taken"].numel() > 0:
                        steps_list.append(float(out["steps_taken"].float().mean().item()))
                        energies_list.append(float(out["final_energy"].float().mean().item()))
                else:
                    assert isinstance(model, ScalableTransformerLM)
                    loss = model.compute_loss(batch)
            losses.append(float(loss.item()))

        mean_loss = sum(losses) / max(len(losses), 1)
        res = {
            "val_loss": mean_loss,
            "val_ppl": float(math.exp(min(mean_loss, 20.0))),
        }
        if steps_list:
            res["val_mean_steps"] = sum(steps_list) / len(steps_list)
            res["val_mean_energy"] = sum(energies_list) / len(energies_list)
        return res

    def run_training_phase(
        self,
        model_kind: str,
        target_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Train either 'pulse' (Hybrid-Local) or 'transformer' (Backprop) with auto-resume & CCU guard."""
        assert model_kind in {"pulse", "transformer"}
        train_blocks, val_blocks = self._prepare_shared_corpus()
        pcfg, tcfg = self._build_models_configs()
        max_steps = target_steps if target_steps is not None else self.config.max_steps

        torch.manual_seed(self.config.seed)
        ckpt_path = self.ckpt_dir / f"{model_kind}_latest.pt"
        history_key = f"{model_kind}_history"
        steps_key = f"{model_kind}_steps"

        if model_kind == "pulse":
            model: torch.nn.Module = self._instantiate_pulse(pcfg)
            if hasattr(model, "io_param_groups"):
                optimizer = self._build_optimizer(model.io_param_groups(self.config.lr, self.config.weight_decay))  # type: ignore[operator]
            else:
                optimizer = self._build_optimizer(model.io_parameters())  # type: ignore[operator]
        else:
            model = ScalableTransformerLM(tcfg).to(self.device)
            optimizer = self._build_optimizer(list(model.parameters()))

        start_step = 0
        if ckpt_path.exists():
            payload = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            model.load_state_dict(payload["model_state_dict"])
            optimizer.load_state_dict(payload["optimizer_state_dict"])
            start_step = int(payload.get("step", 0))
            if model_kind == "pulse" and isinstance(model, _ALL_PULSE_CLASSES):
                model.fast_weights.load_latent_state(
                    payload.get("latent_mean"),
                    payload.get("latent_var"),
                    payload.get("latent_count", 0),
                    payload.get("prior_centroids", []),
                )
            print(f"[{model_kind.upper()}] Resumed from checkpoint at step {start_step}/{max_steps}")

        if self.config.compile_models and self.device.type == "cuda" and hasattr(model, "compile_fast_paths"):
            model.compile_fast_paths()  # type: ignore[operator]

        warmup_steps = int(max_steps * self.config.warmup_ratio)
        tokens_per_step = self.config.batch_size * self.config.grad_accum_steps * self.config.seq_len
        history: List[Dict[str, Any]] = list(self.state.get(history_key, []))

        phase_t0 = time.perf_counter()
        last_tick = phase_t0
        cur_step = start_step

        try:
            for step in range(start_step, max_steps):
                if self.budget.phase_exceeded(model_kind, self.config.phase_budget_fraction):
                    print(
                        f"[{model_kind.upper()}] Phase CCU budget reached "
                        f"({self.budget.phase_spent_ccu(model_kind):.2f} CCU). Saving & stopping cleanly."
                    )
                    break

                lr_mult = _cosine_lr_multiplier(step, max_steps, warmup_steps, self.config.min_lr_ratio)
                for pg in optimizer.param_groups:
                    pg["lr"] = self.config.lr * lr_mult * float(pg.get("lr_mult", 1.0))

                step_t0 = time.perf_counter()
                if model_kind == "pulse":
                    assert isinstance(model, _ALL_PULSE_CLASSES)
                    accum_ce = 0.0
                    ce_count = 0
                    accum_core_mse = 0.0
                    accum_steps = 0.0
                    n_micro = self.config.grad_accum_steps
                    for micro in range(n_micro):
                        batch = self._sample_deterministic_batch(
                            train_blocks,
                            step_idx=step * n_micro + micro,
                            batch_size=self.config.batch_size,
                        )
                        extra_kw: Dict[str, Any] = {}
                        if isinstance(model, (PULSEv3Model, PULSEv4Model)):
                            extra_kw = {
                                "core_lr_mult": lr_mult,
                                "update_radar_every": (
                                    isinstance(model, PULSEv3Model)
                                    and micro == n_micro - 1
                                    and step % 4 == 0
                                ),
                                "update_critic": (micro == n_micro - 1 and step % 2 == 0),
                                "observe_prior": (micro == n_micro - 1 and step % 8 == 0),
                                "grad_accum_steps": n_micro,
                                "is_first_micro": (micro == 0),
                                "is_last_micro": (micro == n_micro - 1),
                            }
                        elif isinstance(model, PULSEv2Model):
                            extra_kw = {"core_lr_mult": lr_mult}
                        st = model.train_step_hybrid_local(
                            batch,
                            io_optimizer=optimizer,
                            async_core=self.config.async_core_updates,
                            max_grad_norm=self.config.max_grad_norm,
                            use_amp_bf16=self.use_bf16,
                            **extra_kw,
                        )
                        if st["ce_loss"] > 0.0:
                            accum_ce += st["ce_loss"]
                            ce_count += 1
                        if st["core_mse"] > 0.0:
                            accum_core_mse = st["core_mse"]
                        accum_steps += st["mean_steps"]
                    train_loss = accum_ce / max(ce_count, 1)
                    extra_metrics = {
                        "core_mse": accum_core_mse,
                        "mean_think_steps": accum_steps / n_micro,
                    }
                else:
                    assert isinstance(model, ScalableTransformerLM)
                    model.train()
                    optimizer.zero_grad(set_to_none=True)
                    accum_loss = 0.0
                    for micro in range(self.config.grad_accum_steps):
                        batch = self._sample_deterministic_batch(
                            train_blocks,
                            step_idx=step * self.config.grad_accum_steps + micro,
                            batch_size=self.config.batch_size,
                        )
                        with torch.amp.autocast(
                            device_type="cuda" if self.device.type == "cuda" else "cpu",
                            dtype=torch.bfloat16,
                            enabled=self.use_bf16,
                        ):
                            loss = model.compute_loss(batch) / self.config.grad_accum_steps
                        loss.backward()
                        accum_loss += float(loss.item())
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
                    optimizer.step()
                    train_loss = accum_loss
                    extra_metrics = {}

                now = time.perf_counter()
                self.budget.record_elapsed_seconds(model_kind, now - last_tick)
                last_tick = now
                cur_step = step + 1

                if cur_step % self.config.log_every == 0 or cur_step == max_steps:
                    rec: Dict[str, Any] = {
                        "step": cur_step,
                        "tokens_seen": cur_step * tokens_per_step,
                        "train_loss": train_loss,
                        "train_ppl": float(math.exp(min(train_loss, 20.0))),
                        "step_ms": (now - step_t0) * 1000.0,
                        "ccu_spent": self.budget.phase_spent_ccu(model_kind),
                        **extra_metrics,
                    }
                    if cur_step % self.config.eval_every == 0 or cur_step == max_steps:
                        rec.update(self._evaluate_val_ppl(model, model_kind, val_blocks))
                    history.append(rec)
                    val_str = f" | val_ppl={rec['val_ppl']:.2f}" if "val_ppl" in rec else ""
                    print(
                        f"[{model_kind.upper()}] step {cur_step:5d}/{max_steps} | "
                        f"tokens={rec['tokens_seen']/1e6:.2f}M | train_loss={train_loss:.3f}"
                        f"{val_str} | {rec['step_ms']:.1f}ms/step | CCU={rec['ccu_spent']:.2f}"
                    )

                if cur_step % self.config.checkpoint_every == 0 or cur_step == max_steps:
                    self._save_model_ckpt(model, optimizer, cur_step, ckpt_path, model_kind)
                    self.state[history_key] = history
                    self.state[steps_key] = cur_step
                    self._save_state()
                last_tick = time.perf_counter()
        finally:
            if cur_step > start_step or not ckpt_path.exists():
                self._save_model_ckpt(model, optimizer, cur_step, ckpt_path, model_kind)
                self.state[history_key] = history
                self.state[steps_key] = cur_step
                self._save_state()

        final_val = self._evaluate_val_ppl(model, model_kind, val_blocks)
        return {
            "model_kind": model_kind,
            "num_parameters": model.num_parameters(),
            "steps_completed": cur_step,
            "tokens_seen": cur_step * tokens_per_step,
            "phase_ccu_spent": self.budget.phase_spent_ccu(model_kind),
            "final_val": final_val,
            "history": history,
        }

    def _save_model_ckpt(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
        path: Path,
        model_kind: str,
    ) -> None:
        payload: Dict[str, Any] = {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        if model_kind == "pulse" and isinstance(model, _ALL_PULSE_CLASSES):
            payload["latent_mean"] = model.fast_weights._latent_mean
            payload["latent_var"] = model.fast_weights._latent_var
            payload["latent_count"] = model.fast_weights._latent_count
            payload["prior_centroids"] = model.fast_weights._prior_centroids
        _atomic_torch_save(payload, path)

    def _load_trained_pair(self) -> Tuple[Any, ScalableTransformerLM]:
        pcfg, tcfg = self._build_models_configs()
        pulse = self._instantiate_pulse(pcfg)
        tf = ScalableTransformerLM(tcfg).to(self.device)

        p_ckpt = self.ckpt_dir / "pulse_latest.pt"
        t_ckpt = self.ckpt_dir / "transformer_latest.pt"
        if p_ckpt.exists():
            pl = torch.load(p_ckpt, map_location=self.device, weights_only=False)
            pulse.load_state_dict(pl["model_state_dict"])
            pulse.fast_weights.load_latent_state(  # type: ignore[union-attr]
                pl.get("latent_mean"),
                pl.get("latent_var"),
                pl.get("latent_count", 0),
                pl.get("prior_centroids", []),
            )
        if t_ckpt.exists():
            tl = torch.load(t_ckpt, map_location=self.device, weights_only=False)
            tf.load_state_dict(tl["model_state_dict"])
        return pulse, tf

    def evaluate_six_axes(self) -> Dict[str, Any]:
        """Run full 7-axis scientific head-to-head comparison on the trained models."""
        t0 = time.perf_counter()
        _, val_blocks = self._prepare_shared_corpus()
        pulse, tf = self._load_trained_pair()
        if self.config.compile_models and self.device.type == "cuda":
            if hasattr(pulse, "compile_fast_paths"):
                pulse.compile_fast_paths()
            if hasattr(tf, "compile_fast_paths"):
                tf.compile_fast_paths()
        pulse.eval()
        tf.eval()

        # --- Axis 1: Validation PPL + Warmed-Up BF16 Median Latency & Flat O(1) Memory ---
        p_val = self._evaluate_val_ppl(pulse, "pulse", val_blocks)
        t_val = self._evaluate_val_ppl(tf, "transformer", val_blocks)

        lengths = list(self.config.eval_seq_lengths)
        p_ms: List[float] = []
        t_ms: List[float] = []
        p_kb: List[float] = []
        t_kb: List[float] = []
        n_warmup = 2 if self.device.type == "cuda" else 1
        n_timed = 5 if self.device.type == "cuda" else 2
        lat_think_steps = 1 if isinstance(pulse, (PULSEv3Model, PULSEv4Model)) else 2

        with torch.no_grad():
            for seq_n in lengths:
                toks = torch.randint(4, self.tokenizer.vocab_size, (1, seq_n), device=self.device)
                with torch.amp.autocast(
                    device_type="cuda" if self.device.type == "cuda" else "cpu",
                    dtype=torch.bfloat16,
                    enabled=self.use_bf16,
                ):
                    for _ in range(n_warmup):
                        p_out = pulse(toks, use_fast_weights=False, max_think_steps=lat_think_steps)
                        _, kv = tf(toks, use_cache=True)

                    p_runs: List[float] = []
                    for _ in range(n_timed):
                        if self.device.type == "cuda":
                            torch.cuda.synchronize()
                        s0 = time.perf_counter()
                        p_out = pulse(toks, use_fast_weights=False, max_think_steps=lat_think_steps)
                        if self.device.type == "cuda":
                            torch.cuda.synchronize()
                        p_runs.append((time.perf_counter() - s0) * 1000.0)
                    p_ms.append(float(np.median(p_runs)))
                    p_kb.append(pulse.working_state_bytes(p_out["working_state"]) / 1024.0)

                    t_runs: List[float] = []
                    for _ in range(n_timed):
                        if self.device.type == "cuda":
                            torch.cuda.synchronize()
                        s1 = time.perf_counter()
                        _, kv = tf(toks, use_cache=True)
                        if self.device.type == "cuda":
                            torch.cuda.synchronize()
                        t_runs.append((time.perf_counter() - s1) * 1000.0)
                    t_ms.append(float(np.median(t_runs)))
                    t_kb.append(tf.kv_cache_bytes(kv) / 1024.0)

        p_trainable_params = sum(p.numel() for p in pulse.io_parameters())
        axis1 = {
            "pulse_params": pulse.num_parameters(),
            "pulse_trainable_params": p_trainable_params,
            "transformer_params": tf.num_parameters(),
            "pulse_val_ppl": p_val["val_ppl"],
            "transformer_val_ppl": t_val["val_ppl"],
            "lengths": lengths,
            "pulse_ms": p_ms,
            "transformer_ms": t_ms,
            "pulse_state_kb": p_kb,
            "transformer_kv_kb": t_kb,
            "memory_reduction_max_len": t_kb[-1] / max(p_kb[-1], 1e-9),
        }

        # --- Axis 2: Smooth EMA-Threshold Native Adaptive Compute Depth ---
        eff_thresh = (
            pulse.effective_energy_threshold()
            if hasattr(pulse, "effective_energy_threshold")
            else pulse.config.energy_threshold
        )
        with torch.no_grad():
            easy_batch = self._sample_deterministic_batch(val_blocks, 999_001, min(self.config.batch_size, 4))[:, :-1]
            easy_out = pulse(easy_batch, use_fast_weights=False)
            # Genuine Out-Of-Distribution technical/code/math text (never uniform random tokens!)
            ood_text = (
                " ".join(f"{q}{a}" for q, a in ONLINE_TEST_FACTS)
                + " Theorem 4.2: Let M be a compact symplectic manifold with curvature tensor R_ijkl "
                "and eigenvalue spectrum lambda_k satisfying det(A - lambda I) = 0. "
                "def solve_rls(P, phi, y, gamma=0.99): g = (P @ phi) / (gamma + phi.T @ P @ phi); "
                "return P - np.outer(g, P @ phi). "
                "HTTP/2 503 Service Unavailable; TLSv1.3 handshake aborted at cipher AES_256_GCM_SHA384. "
            )
            ood_ids = self.tokenizer.encode(ood_text)
            if not ood_ids:
                ood_ids = list(range(4, min(self.tokenizer.vocab_size, 132)))
            need_n = easy_batch.numel()
            reps = (need_n + len(ood_ids) - 1) // len(ood_ids)
            novel_batch = torch.tensor(
                (ood_ids * reps)[:need_n], dtype=torch.long, device=self.device
            ).view_as(easy_batch)
            novel_out = pulse(novel_batch, use_fast_weights=False)
            z_e = easy_out["z_chunks"].reshape(-1, pulse.config.concept_dim)[:8].float()
            z_top_e = easy_out["z_top_seq"].reshape(-1, pulse.config.top_dim)[:8].float()
            conflict_relax = pulse.field.relax(
                z_bottom=z_e,
                top_prior=-z_top_e * 2.5,
                max_steps=pulse.config.max_think_steps,
                energy_threshold=eff_thresh,
            )

        axis2 = {
            "easy_val_steps": float(easy_out["steps_taken"].float().mean().item()),
            "novel_random_steps": float(novel_out["steps_taken"].float().mean().item()),
            "conflict_steps": float(conflict_relax.steps_taken.float().mean().item()),
            "effective_energy_threshold": float(eff_thresh),
            "transformer_fixed_layers": float(tf.config.n_layers),
        }

        # --- Axes 3 & 4: RLS Online Learning (0-Ctx, 0.00% Base Interference) + Orthogonal Sleep ---
        k = pulse.config.chunk_size
        fact_cues: List[torch.Tensor] = []
        fact_tgts: List[torch.Tensor] = []
        for q_txt, a_txt in ONLINE_TEST_FACTS:
            q_ids = self.tokenizer.encode(q_txt)
            a_ids = self.tokenizer.encode(a_txt)
            q_len = max(k * 2, ((len(q_ids) + k - 1) // k) * k)
            q_pad = (q_ids + [self.tokenizer.eos_id] * q_len)[: k * 2]
            a_pad = (a_ids + [self.tokenizer.eos_id] * k)[:k]
            fact_cues.append(torch.tensor(q_pad, dtype=torch.long))
            fact_tgts.append(torch.tensor(a_pad, dtype=torch.long))

        cues_t = torch.stack(fact_cues, dim=0).to(self.device)
        tgts_t = torch.stack(fact_tgts, dim=0).to(self.device)
        full_facts = torch.cat([cues_t, tgts_t], dim=1)

        with torch.no_grad():
            if hasattr(pulse, "encode_target_concept"):
                tgt_bank = pulse.encode_target_concept(tgts_t)
            else:
                tgt_bank = pulse.encode_chunks(tgts_t)[:, 0, :].float()

            def fact_retrieval_acc(use_fast: bool) -> float:
                pred = pulse.predict_next_concept(cues_t, use_fast_weights=use_fast)
                sims = F.cosine_similarity(pred.unsqueeze(1), tgt_bank.unsqueeze(0), dim=-1)
                preds = sims.argmax(dim=-1)
                return float((preds == torch.arange(len(ONLINE_TEST_FACTS), device=self.device)).float().mean().item())

            pulse_fact_before = fact_retrieval_acc(use_fast=True)
            tf_fact_loss_before = float(tf.compute_loss(full_facts).item())

        # Teach all unseen facts online into PULSE W_fast (0 backprop!)
        teach_stats = pulse.teach_fact_online(cues_t, tgts_t, repeats=6)
        pulse_fact_online = fact_retrieval_acc(use_fast=True)

        # Measure end-to-end LM bypass Top-1 hit accuracy & 0.00% interference on base validation text!
        with torch.no_grad():
            cue_logits_fast = pulse(cues_t, use_fast_weights=True)["logits"]
            lm_bypass_top1 = float((cue_logits_fast[:, -1, :].argmax(dim=-1) == tgts_t[:, 0]).float().mean().item())
            # Evaluate base Val PPL on IDENTICAL eval_batches with use_fast_weights=True active -> exact 0.000% drift audit!
            online_losses: List[float] = []
            for b_idx in range(self.config.eval_batches):
                batch = self._sample_deterministic_batch(
                    val_blocks, step_idx=900_000 + b_idx, batch_size=self.config.batch_size
                )
                with torch.amp.autocast(
                    device_type="cuda" if self.device.type == "cuda" else "cpu",
                    dtype=torch.bfloat16,
                    enabled=self.use_bf16,
                ):
                    o_on = (
                        pulse(batch[:, :-1], use_fast_weights=True, return_state=False)
                        if isinstance(pulse, PULSEv4Model)
                        else pulse(batch[:, :-1], use_fast_weights=True)
                    )
                    l_on = F.cross_entropy(
                        o_on["logits"].reshape(-1, o_on["logits"].shape[-1]), batch[:, 1:].reshape(-1)
                    )
                online_losses.append(float(l_on.item()))
            p_val_during_online = float(math.exp(min(sum(online_losses) / max(len(online_losses), 1), 20.0)))
            online_drift_pct = ((p_val_during_online - p_val["val_ppl"]) / max(p_val["val_ppl"], 1e-6)) * 100.0

        # Sleep consolidate W_fast -> W_slow and wipe W_fast = 0!
        sleep_stats = pulse.sleep_consolidate(num_replay_samples=48, num_epochs=40, clear_fast_weights=True)
        pulse_fact_post_sleep = fact_retrieval_acc(use_fast=False)
        pulse_val_post_sleep = self._evaluate_val_ppl(pulse, "pulse", val_blocks)["val_ppl"]

        # Naive Transformer continual fine-tuning on the same new facts (no old data!)
        tf_ft_opt = torch.optim.AdamW(tf.parameters(), lr=3e-4)
        tf.train()
        for _ in range(15):
            tf_ft_opt.zero_grad(set_to_none=True)
            loss_ft = tf.compute_loss(full_facts)
            loss_ft.backward()
            tf_ft_opt.step()
        tf_val_after_ft = self._evaluate_val_ppl(tf, "transformer", val_blocks)["val_ppl"]
        with torch.no_grad():
            tf_fact_loss_after_ft = float(tf.compute_loss(full_facts).item())

        axis3 = {
            "pulse_fact_acc_before": pulse_fact_before,
            "pulse_fact_acc_online_zero_ctx": pulse_fact_online,
            "pulse_lm_bypass_top1_acc": lm_bypass_top1,
            "pulse_online_base_ppl_drift_pct": online_drift_pct,
            "pulse_surprise_gate": teach_stats["max_gate"],
            "transformer_frozen_fact_nll": tf_fact_loss_before,
        }
        axis4 = {
            "pulse_base_val_ppl_before_sleep": p_val["val_ppl"],
            "pulse_base_val_ppl_after_sleep": pulse_val_post_sleep,
            "pulse_ppl_drift_pct": ((pulse_val_post_sleep - p_val["val_ppl"]) / max(p_val["val_ppl"], 1e-6)) * 100.0,
            "pulse_new_fact_acc_post_sleep_wslow": pulse_fact_post_sleep,
            "pulse_wfast_norm_after_sleep": float(pulse.causal_layer.w_fast.norm().item()),
            "sleep_consolidated_episodes": sleep_stats["causal_consolidated"],
            "transformer_base_val_ppl_before_ft": t_val["val_ppl"],
            "transformer_base_val_ppl_after_ft": tf_val_after_ft,
            "transformer_ppl_drift_pct": ((tf_val_after_ft - t_val["val_ppl"]) / max(t_val["val_ppl"], 1e-6)) * 100.0,
            "transformer_fact_nll_after_ft": tf_fact_loss_after_ft,
        }

        # --- Axis 5: Structural Anti-Hallucination (Cross-Story + Hard Single-Entity Swap!) ---
        with torch.no_grad():
            n_ah = min(self.config.batch_size, 16)
            n_ah_batches = 4 if val_blocks.shape[0] >= 64 else 1
            et_all, ef_all, eh_all = [], [], []
            for ah_bi in range(n_ah_batches):
                eval_b = self._sample_deterministic_batch(val_blocks, 999_002 + ah_bi, n_ah)
                split = (eval_b.shape[1] // k - 1) * k
                prompts_ah = eval_b[:, :split]
                true_cont = eval_b[:, split : split + k]

                # 5a. Cross-story mismatched continuation
                false_cont = torch.roll(true_cont, shifts=1, dims=0)
                if torch.equal(true_cont, false_cont):
                    false_cont = (true_cont + 37) % self.tokenizer.vocab_size

                # 5b. Hard in-place single-entity swap (only 1-2 tokens altered inside true continuation!)
                hard_swap_cont = true_cont.clone()
                swap_pos = k // 2
                donor = torch.roll(true_cont, shifts=max(n_ah // 2, 1), dims=0)
                hard_swap_cont[:, swap_pos] = donor[:, swap_pos]
                if k >= 4:
                    hard_swap_cont[:, swap_pos - 1] = donor[:, swap_pos - 1]
                same_sw = hard_swap_cont == true_cont
                if same_sw[:, swap_pos].any():
                    hard_swap_cont[:, swap_pos] = (true_cont[:, swap_pos] + 53) % self.tokenizer.vocab_size

                et_all.append(pulse.continuation_free_energy(prompts_ah, true_cont, use_fast_weights=False))
                ef_all.append(pulse.continuation_free_energy(prompts_ah, false_cont, use_fast_weights=False))
                eh_all.append(pulse.continuation_free_energy(prompts_ah, hard_swap_cont, use_fast_weights=False))

            e_true = torch.cat(et_all, dim=0)
            e_false = torch.cat(ef_all, dim=0)
            e_hard = torch.cat(eh_all, dim=0)
            auroc = float((e_false.unsqueeze(1) > e_true.unsqueeze(0)).float().mean().item())
            hard_auroc = float((e_hard.unsqueeze(1) > e_true.unsqueeze(0)).float().mean().item())

        axis5 = {
            "mean_consistent_energy": float(e_true.mean().item()),
            "mean_contradictory_energy": float(e_false.mean().item()),
            "mean_hard_swap_energy": float(e_hard.mean().item()),
            "energy_ratio": float((e_false.mean() / e_true.mean().clamp_min(1e-8)).item()),
            "hard_energy_ratio": float((e_hard.mean() / e_true.mean().clamp_min(1e-8)).item()),
            "hallucination_auroc": auroc,
            "hard_entity_swap_auroc": hard_auroc,
            "cross_paired_acc": float((e_false > e_true).float().mean().item()),
            "hard_paired_acc": float((e_hard > e_true).float().mean().item()),
            "n_eval_stories": int(e_true.numel()),
        }

        # --- Axis 6: Local Energy Learning Audit & Strictly Isolated Peak Training VRAM ---
        p_peak_mb = 0.0
        t_peak_mb = 0.0
        sample_b = self._sample_deterministic_batch(val_blocks, 999_003, self.config.batch_size)
        # Evict fine-tuned Transformer & optimizer so PULSE VRAM measurement is 100% isolated!
        del tf, tf_ft_opt
        gc.collect()
        p_io_opt = self._build_optimizer(pulse.io_parameters())
        # Warm up 1 step before resetting peak VRAM so compile scratch is excluded symmetrically!
        pulse.train_step_hybrid_local(sample_b, p_io_opt, use_amp_bf16=self.use_bf16)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        pulse.train_step_hybrid_local(sample_b, p_io_opt, use_amp_bf16=self.use_bf16)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            p_peak_mb = torch.cuda.max_memory_allocated() / (1024.0**2)
        pulse_core_grads = pulse.core_autograd_grad_count()
        pulse_core_param_count = (
            pulse.core_param_count()
            if hasattr(pulse, "core_param_count")
            else sum(p.numel() for p in pulse.field.parameters())
            + sum(p.numel() for p in pulse.causal_layer.parameters())
        )

        # Evict PULSE & optimizer before measuring Transformer Peak VRAM in strict isolation!
        del pulse, p_io_opt
        gc.collect()
        _, tcfg_vram = self._build_models_configs()
        tf_vram = ScalableTransformerLM(tcfg_vram).to(self.device)
        t_ckpt = self.ckpt_dir / "transformer_latest.pt"
        if t_ckpt.exists():
            tf_vram.load_state_dict(torch.load(t_ckpt, map_location=self.device, weights_only=False)["model_state_dict"])
        if self.config.compile_models and self.device.type == "cuda":
            tf_vram.compile_fast_paths()
        t_opt = self._build_optimizer(list(tf_vram.parameters()))
        # Warm up 1 step symmetrically before measuring steady-state Peak VRAM!
        t_opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            device_type="cuda" if self.device.type == "cuda" else "cpu",
            dtype=torch.bfloat16,
            enabled=self.use_bf16,
        ):
            loss_t = tf_vram.compute_loss(sample_b)
        loss_t.backward()
        t_opt.step()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        t_opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            device_type="cuda" if self.device.type == "cuda" else "cpu",
            dtype=torch.bfloat16,
            enabled=self.use_bf16,
        ):
            loss_t = tf_vram.compute_loss(sample_b)
        loss_t.backward()
        t_opt.step()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            t_peak_mb = torch.cuda.max_memory_allocated() / (1024.0**2)
        tf_grad_tensors = sum(1 for p in tf_vram.parameters() if p.grad is not None)
        del tf_vram, t_opt
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        # Reload pure trained pair for Axis 7 extrapolation & unseen sample generations!
        pulse_fresh, tf_fresh = self._load_trained_pair()

        # --- Axis 7: Strictly Non-Repeating Multi-Window Zero-Shot 4x Length Extrapolation ---
        val_stream = self._val_stream if self._val_stream is not None else val_blocks.reshape(-1)
        target_extrap = self.config.seq_len * 4
        max_avail_len = max(int(val_stream.numel()) - 1, self.config.seq_len + 1)
        extrap_len = min(target_extrap, max_avail_len)
        stride = extrap_len + 1
        num_extrap_seqs = min(8, max(int(val_stream.numel()) // stride, 1))
        non_repeating_ok = bool(
            self._val_stream_non_repeating and int(val_stream.numel()) >= num_extrap_seqs * stride
        )

        pulse_fresh.eval()
        tf_fresh.eval()
        p_full_losses: List[float] = []
        t_full_losses: List[float] = []
        p_tail_losses: List[float] = []
        t_tail_losses: List[float] = []
        tail_start = min(self.config.seq_len, extrap_len - 1)

        with torch.no_grad():
            for seq_i in range(num_extrap_seqs):
                seg = val_stream[seq_i * stride : (seq_i + 1) * stride].unsqueeze(0).long().to(self.device)
                with torch.amp.autocast(
                    device_type="cuda" if self.device.type == "cuda" else "cpu",
                    dtype=torch.bfloat16,
                    enabled=self.use_bf16,
                ):
                    p_ex_logits = pulse_fresh(seg[:, :-1], use_fast_weights=False)["logits"]
                    t_ex_logits, _ = tf_fresh(seg[:, :-1], use_cache=False)
                    tgts_ex = seg[:, 1:]
                    p_full_losses.append(
                        float(F.cross_entropy(p_ex_logits.reshape(-1, p_ex_logits.shape[-1]), tgts_ex.reshape(-1)).item())
                    )
                    t_full_losses.append(
                        float(F.cross_entropy(t_ex_logits.reshape(-1, t_ex_logits.shape[-1]), tgts_ex.reshape(-1)).item())
                    )
                    # Strictly out-of-window tail positions [train_seq_len .. extrap_len)!
                    p_tail_losses.append(
                        float(
                            F.cross_entropy(
                                p_ex_logits[:, tail_start:, :].reshape(-1, p_ex_logits.shape[-1]),
                                tgts_ex[:, tail_start:].reshape(-1),
                            ).item()
                        )
                    )
                    t_tail_losses.append(
                        float(
                            F.cross_entropy(
                                t_ex_logits[:, tail_start:, :].reshape(-1, t_ex_logits.shape[-1]),
                                tgts_ex[:, tail_start:].reshape(-1),
                            ).item()
                        )
                    )

        axis7 = {
            "train_seq_len": self.config.seq_len,
            "extrap_seq_len": extrap_len,
            "num_non_overlapping_windows": num_extrap_seqs,
            "non_repeating_verified": non_repeating_ok,
            "pulse_extrap_ppl": float(math.exp(min(sum(p_full_losses) / len(p_full_losses), 20.0))),
            "transformer_extrap_ppl": float(math.exp(min(sum(t_full_losses) / len(t_full_losses), 20.0))),
            "pulse_extrap_tail_ppl": float(math.exp(min(sum(p_tail_losses) / len(p_tail_losses), 20.0))),
            "transformer_extrap_tail_ppl": float(math.exp(min(sum(t_tail_losses) / len(t_tail_losses), 20.0))),
        }

        # Genuinely UNSEEN prompt for sample text generations (never memorized from training!)
        sample_prompt_txt = "One rainy morning, a small silver turtle climbed onto the warm mossy stone and saw"
        p_ids = torch.tensor([self.tokenizer.encode(sample_prompt_txt)], dtype=torch.long, device=self.device)
        if p_ids.shape[1] == 0:
            p_ids = torch.tensor([[5, 6, 7, 8]], dtype=torch.long, device=self.device)
        torch.manual_seed(self.config.seed + 7)
        pulse_gen_txt = self.tokenizer.decode(
            pulse_fresh.generate(p_ids, max_new_tokens=32, temperature=0.7)[0].tolist()
        )
        torch.manual_seed(self.config.seed + 19)
        tf_gen_txt = self.tokenizer.decode(
            tf_fresh.generate(p_ids, max_new_tokens=32, temperature=0.7)[0].tolist()
        )

        axis6 = {
            "pulse_core_params_zero_backprop": pulse_core_param_count,
            "pulse_core_autograd_grads": pulse_core_grads,
            "pulse_peak_train_vram_mb": p_peak_mb,
            "transformer_autograd_grad_tensors": tf_grad_tensors,
            "transformer_peak_train_vram_mb": t_peak_mb,
        }

        self.budget.record_elapsed_seconds("eval", time.perf_counter() - t0)
        self._save_state()

        return {
            "axis1_complexity_memory": axis1,
            "axis2_adaptive_compute": axis2,
            "axis3_online_plasticity": axis3,
            "axis4_sleep_consolidation": axis4,
            "axis5_anti_hallucination": axis5,
            "axis6_local_learning": axis6,
            "axis7_length_extrapolation": axis7,
            "sample_generations": {
                "prompt": sample_prompt_txt,
                "pulse": pulse_gen_txt,
                "transformer": tf_gen_txt,
            },
        }

    def generate_plots_and_report(self, full_results: Dict[str, Any]) -> None:
        """Export publication-grade Matplotlib plots and Markdown EXPERIMENT_REPORT.md."""
        plt.style.use("seaborn-v0_8-whitegrid")
        c_p, c_t = "#0F766E", "#C2410C"
        ver = self.config.pulse_version.upper()
        pulse_label = (
            "PULSE-v5 (AttnRes + Multi-Scale Resonance)"
            if self.config.pulse_version == "v4"
            else (
                "PULSE-v3 (Hyper-Plastic Multi-Rate Resonance)"
                if self.config.pulse_version == "v3"
                else (
                    "PULSE-v2 (Hyper-Plastic Resonance)"
                    if self.config.pulse_version == "v2"
                    else "PULSE-LM (Hybrid-Local)"
                )
            )
        )

        # 1. Training curves figure (2x2: Tokens Loss, Val PPL vs Steps, Val PPL vs CCU Iso-Compute, Core MSE)
        fig, axes_grid = plt.subplots(2, 2, figsize=(13.5, 8.5), dpi=150)
        axes = axes_grid.reshape(-1)
        fig.suptitle(f"Controlled Training & Iso-CCU Efficiency: {pulse_label} vs. Matched Transformer", fontweight="bold")
        ph = full_results["pulse_training"]["history"]
        th = full_results["transformer_training"]["history"]

        if ph and th:
            axes[0].plot([r["tokens_seen"] / 1e6 for r in ph], [r["train_loss"] for r in ph], color=c_p, lw=2, label=f"PULSE-{ver}")
            axes[0].plot([r["tokens_seen"] / 1e6 for r in th], [r["train_loss"] for r in th], color=c_t, lw=2, label="Transformer")
            axes[0].set_title("Training Cross-Entropy Loss vs. Tokens")
            axes[0].set_xlabel("Tokens Seen (M)")
            axes[0].set_ylabel("CE Loss")
            axes[0].legend()

            pv = [r for r in ph if "val_ppl" in r]
            tv = [r for r in th if "val_ppl" in r]
            if pv and tv:
                axes[1].plot([r["step"] for r in pv], [r["val_ppl"] for r in pv], "o-", color=c_p, lw=2, label=f"PULSE-{ver}")
                axes[1].plot([r["step"] for r in tv], [r["val_ppl"] for r in tv], "s-", color=c_t, lw=2, label="Transformer")
                axes[2].plot([r["ccu_spent"] for r in pv], [r["val_ppl"] for r in pv], "o-", color=c_p, lw=2, label=f"PULSE-{ver}")
                axes[2].plot([r["ccu_spent"] for r in tv], [r["val_ppl"] for r in tv], "s-", color=c_t, lw=2, label="Transformer")
            axes[1].set_title("Sample Efficiency: Val PPL vs. Steps")
            axes[1].set_xlabel("Step")
            axes[1].set_ylabel("Validation PPL")
            axes[1].legend()

            axes[2].set_title("Iso-CCU Wall-Clock Efficiency: Val PPL vs. CCU")
            axes[2].set_xlabel("GPU Compute Spent (CCU)")
            axes[2].set_ylabel("Validation PPL")
            axes[2].legend()

            axes[3].plot([r["step"] for r in ph], [r.get("core_mse", 0.0) for r in ph], color=c_p, lw=2, label="Core Local MSE (0 Backprop)")
            axes[3].set_title("PULSE Zero-Backprop Core Convergence")
            axes[3].set_xlabel("Step")
            axes[3].set_ylabel("Layer Local Predictive MSE")
            axes[3].legend()

        plt.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(self.out_dir / "training_curves.png")
        plt.close(fig)

        # 2. Six-axis comparison figure
        ev = full_results["six_axis_eval"]
        b1, b2, b3, b4, b5, b6 = (
            ev["axis1_complexity_memory"],
            ev["axis2_adaptive_compute"],
            ev["axis3_online_plasticity"],
            ev["axis4_sleep_consolidation"],
            ev["axis5_anti_hallucination"],
            ev["axis6_local_learning"],
        )
        fig2, ax = plt.subplots(2, 3, figsize=(16, 9.2), dpi=150)
        fig2.suptitle(f"{pulse_label} vs. Matched RoPE-SwiGLU Transformer — Controlled Evaluation", fontsize=14, fontweight="bold")

        ax[0, 0].plot(b1["lengths"], b1["transformer_kv_kb"], "o-", color=c_t, lw=2.2, label="Transformer KV Cache O(N)")
        ax[0, 0].plot(b1["lengths"], b1["pulse_state_kb"], "s-", color=c_p, lw=2.2, label="PULSE Fast State O(1)")
        ax[0, 0].set_title("1. Memory Footprint vs. Sequence Length")
        ax[0, 0].set_xlabel("Sequence Length N")
        ax[0, 0].set_ylabel("State / Cache (KB)")
        ax[0, 0].legend(fontsize=8)

        ax[0, 1].plot(b1["lengths"], b1["transformer_ms"], "o-", color=c_t, lw=2.2, label="Transformer O(N^2)")
        ax[0, 1].plot(b1["lengths"], b1["pulse_ms"], "s-", color=c_p, lw=2.2, label="PULSE O(N)")
        ax[0, 1].set_title("2. Inference Latency Scaling (Warmed Median)")
        ax[0, 1].set_xlabel("Sequence Length N")
        ax[0, 1].set_ylabel("Latency (ms)")
        ax[0, 1].legend(fontsize=8)

        tiers = ["Easy Val", "Novel OOD", "Conflict"]
        x = np.arange(3)
        w = 0.35
        ax[0, 2].bar(x - w / 2, [b2["easy_val_steps"], b2["novel_random_steps"], b2["conflict_steps"]], w, color=c_p, label="PULSE Adaptive")
        ax[0, 2].bar(x + w / 2, [b2["transformer_fixed_layers"]] * 3, w, color=c_t, label="Transformer Fixed")
        ax[0, 2].set_xticks(x)
        ax[0, 2].set_xticklabels(tiers)
        ax[0, 2].set_title("3. Adaptive Compute Depth")
        ax[0, 2].legend(fontsize=8)

        ax[1, 0].bar(
            ["Before\nOnline", "W_fast\nOnline (0-Ctx)", "Post-Sleep\nW_slow Only"],
            [
                b3["pulse_fact_acc_before"] * 100.0,
                b3["pulse_fact_acc_online_zero_ctx"] * 100.0,
                b4["pulse_new_fact_acc_post_sleep_wslow"] * 100.0,
            ],
            color=[c_t, "#2563EB", c_p],
        )
        ax[1, 0].set_ylim(0, 115)
        ax[1, 0].set_ylabel("Unseen Fact Retrieval (%)")
        ax[1, 0].set_title("4. Online Plasticity & Sleep Consolidation")

        ax[1, 1].bar(
            ["Consistent\nContinuation", "Hard Entity\nSwap", "Cross-Story\nMismatch"],
            [
                b5["mean_consistent_energy"],
                b5.get("mean_hard_swap_energy", b5["mean_contradictory_energy"]),
                b5["mean_contradictory_energy"],
            ],
            color=[c_p, "#D97706", c_t],
        )
        ax[1, 1].set_ylabel("Tri-Signal Free Energy E(z)")
        ax[1, 1].set_title(f"5. Anti-Hallucination (AUROC={b5['hallucination_auroc']:.2f})")

        ax[1, 2].bar(
            ["PULSE Sleep\nBase PPL Drift %", "Transformer FT\nBase PPL Drift %"],
            [b4["pulse_ppl_drift_pct"], b4["transformer_ppl_drift_pct"]],
            color=[c_p, c_t],
        )
        ax[1, 2].set_ylabel("Forgetting (Base Val PPL Drift %)")
        ax[1, 2].set_title("6. Lifelong Continual Anti-Forgetting")

        plt.tight_layout(rect=(0, 0, 1, 0.94))
        fig2.savefig(self.out_dir / "six_axis_comparison.png")
        plt.close(fig2)

        # 3. Markdown Report
        budget_info = full_results["budget"]
        b7 = ev.get("axis7_length_extrapolation", {})
        extrap_row = (
            f"| **7. Zero-Shot Length Extrapolation (`{b7['train_seq_len']} -> {b7['extrap_seq_len']}`, {b7.get('num_non_overlapping_windows', 1)} non-overlap windows, non-rep=`{b7.get('non_repeating_verified', False)}`)** "
            f"| `{b7['transformer_extrap_ppl']:.2f}` PPL (Tail `>{b7['train_seq_len']}`: `{b7.get('transformer_extrap_tail_ppl', b7['transformer_extrap_ppl']):.2f}`) "
            f"| **`{b7['pulse_extrap_ppl']:.2f}` PPL** (Tail `>{b7['train_seq_len']}`: **`{b7.get('pulse_extrap_tail_ppl', b7['pulse_extrap_ppl']):.2f}`**) |\n"
            if b7
            else ""
        )
        p_med_step = float(np.median([r["step_ms"] for r in ph])) if ph else 0.0
        t_med_step = float(np.median([r["step_ms"] for r in th])) if th else 0.0
        p_phase_ccu = float(full_results["pulse_training"].get("phase_ccu_spent", 0.0))
        t_phase_ccu = float(full_results["transformer_training"].get("phase_ccu_spent", 0.0))
        # Compute Iso-CCU efficiency: CCU needed by PULSE to reach Transformer's final Val PPL
        iso_ccu_line = ""
        pv_pts = [r for r in ph if "val_ppl" in r]
        tf_final_ppl = float(b1["transformer_val_ppl"])
        if pv_pts and t_phase_ccu > 0.0 and p_phase_ccu > 0.0:
            match_ccu: Optional[float] = None
            match_step: Optional[float] = None
            for idx_pt, pt in enumerate(pv_pts):
                if pt["val_ppl"] <= tf_final_ppl:
                    if idx_pt == 0:
                        match_ccu = float(pt["ccu_spent"])
                        match_step = float(pt["step"])
                    else:
                        prev = pv_pts[idx_pt - 1]
                        frac = (prev["val_ppl"] - tf_final_ppl) / max(prev["val_ppl"] - pt["val_ppl"], 1e-8)
                        frac = max(0.0, min(1.0, frac))
                        match_ccu = float(prev["ccu_spent"] + frac * (pt["ccu_spent"] - prev["ccu_spent"]))
                        match_step = float(prev["step"] + frac * (pt["step"] - prev["step"]))
                    break
            if match_ccu is not None and match_step is not None and match_ccu > 0.0:
                total_steps = max(full_results["pulse_training"]["steps_completed"], 1)
                iso_ccu_line = (
                    f"- **Iso-CCU / Iso-Time Efficiency**: PULSE matched Transformer's final Val PPL (`{tf_final_ppl:.2f}`) "
                    f"at step **`{match_step:.0f}`** (`{total_steps / max(match_step, 1.0):.2f}x` sample efficiency) "
                    f"using **`{match_ccu:.3f}` CCU** vs. Transformer's `{t_phase_ccu:.3f}` CCU "
                    f"(**`{t_phase_ccu / match_ccu:.2f}x` Iso-CCU compute efficiency**)\n"
                )
        report = f"""# {pulse_label} vs. Matched Transformer — Controlled Scientific Experiment Report

## 1. Hardware & Budget Telemetry
- **GPU**: `{budget_info['gpu_name']}` (Est. `{budget_info['ccu_per_hour']:.1f}` CCU/hr, BF16=`{self.use_bf16}`, Optimizer=`{self.config.optimizer_type}`, Symmetric Compile=`{self.config.compile_models}`)
- **Total CCU Spent**: `{budget_info['total_spent_ccu']:.2f}` / `{budget_info['total_ccu_budget']:.1f}` CCU (PULSE=`{p_phase_ccu:.3f}` CCU, Transformer=`{t_phase_ccu:.3f}` CCU)
- **Parameter Parity**: {pulse_label} `{b1['pulse_params']:,}` (Trainable Backprop: `{b1.get('pulse_trainable_params', b1['pulse_params']):,}`, Zero-Backprop Core: `{b6['pulse_core_params_zero_backprop']:,}`) vs. Transformer `{b1['transformer_params']:,}`
- **Matched Training Budget**: `{full_results['pulse_training']['steps_completed']:,}` steps (`{full_results['pulse_training']['tokens_seen']/1e6:.2f}M` tokens each, Dataset=`{self.config.dataset}`)
- **Median Training Step Time**: PULSE `{p_med_step:.1f} ms/step` vs. Transformer `{t_med_step:.1f} ms/step`
{iso_ccu_line}

## 2. Quantitative Paradigm Comparison

| Paradigm Axis | Matched Transformer (RoPE + SwiGLU) | {pulse_label} |
| :--- | :--- | :--- |
| **1a. Held-Out Validation PPL (`N={self.config.seq_len}`)** | `{b1['transformer_val_ppl']:.2f}` | **`{b1['pulse_val_ppl']:.2f}`** |
| **1b. Working Memory (`N={b1['lengths'][-1]}`)** | `{b1['transformer_kv_kb'][-1]:.1f} KB` KV Cache (`O(N)`) | **`{b1['pulse_state_kb'][-1]:.2f} KB` Flat `O(1)`** (`{b1['memory_reduction_max_len']:.1f}x` smaller) |
| **1c. Warmed Latency (`N={b1['lengths'][-1]}`)** | `{b1['transformer_ms'][-1]:.1f} ms` (`O(N^2)`) | **`{b1['pulse_ms'][-1]:.1f} ms`** (`O(N)`) |
| **2. Compute Depth** | Fixed `{b2['transformer_fixed_layers']:.0f}` layers | Adaptive: Easy `{b2['easy_val_steps']:.2f}` / Novel `{b2['novel_random_steps']:.2f}` / Conflict `{b2['conflict_steps']:.1f}` steps |
| **3. Online Rule Learning (0-Context)** | Frozen (`fact NLL={b3['transformer_frozen_fact_nll']:.2f}`) | **`{b3['pulse_fact_acc_online_zero_ctx']*100:.1f}%`** (`before={b3['pulse_fact_acc_before']*100:.1f}%`, LM Top-1=`{b3.get('pulse_lm_bypass_top1_acc', 0.0)*100:.1f}%`, base drift=`{b3.get('pulse_online_base_ppl_drift_pct', 0.0):+.2f}%`) |
| **4. Sleep Consolidation vs. Forgetting** | Naive FT Base PPL Drift: **`{b4['transformer_ppl_drift_pct']:+.2f}%`** | Sleep (`W_fast=0`) New Recall: **`{b4['pulse_new_fact_acc_post_sleep_wslow']*100:.1f}%`**, Base PPL Drift: **`{b4['pulse_ppl_drift_pct']:+.2f}%`** |
| **5. Structural Anti-Hallucination** | Unidirectional autoregressive | **Cross-Story AUROC = `{b5['hallucination_auroc']:.3f}`** (`{b5['energy_ratio']:.2f}x`), **Hard Entity-Swap AUROC = `{b5.get('hard_entity_swap_auroc', 0.0):.3f}`** |
| **6. Zero-Backprop Core & Peak VRAM** | `{b6['transformer_autograd_grad_tensors']}` `.grad` tensors (`{b6['transformer_peak_train_vram_mb']:.1f} MB`) | **`{b6['pulse_core_autograd_grads']}` `.grad` tensors in core** (`{b6['pulse_peak_train_vram_mb']:.1f} MB`) |
{extrap_row}
## 3. Sample Generations (Unseen Held-Out Prompt)
- **Prompt**: `{ev['sample_generations']['prompt']}`
- **PULSE**: `{ev['sample_generations']['pulse']}`
- **Transformer**: `{ev['sample_generations']['transformer']}`

## 4. Visual Figures
![Training Curves](training_curves.png)
![Six-Axis Comparison](six_axis_comparison.png)
"""
        (self.out_dir / "EXPERIMENT_REPORT.md").write_text(report, encoding="utf-8")

    def run_all(self) -> Dict[str, Any]:
        """Run Phase A (PULSE-LM) -> Phase B (Matched Transformer) -> Phase C (6-Axis Eval & Report)."""
        print("=" * 72)
        print(
            f"PULSE vs. Transformer Controlled Experiment | Device={self.device} "
            f"| GPU={self.budget.gpu_name} | Budget={self.budget.total_ccu_budget:.1f} CCU"
        )
        print("=" * 72)

        pulse_res = self.run_training_phase("pulse", target_steps=self.config.max_steps)
        # Match Transformer step count identically to PULSE's actual completed steps!
        matched_steps = max(pulse_res["steps_completed"], 1)
        tf_res = self.run_training_phase("transformer", target_steps=matched_steps)
        six_axis = self.evaluate_six_axes()

        full_results = {
            "config": asdict(self.config),
            "budget": self.budget.to_dict(),
            "pulse_training": pulse_res,
            "transformer_training": tf_res,
            "six_axis_eval": six_axis,
        }
        (self.out_dir / "experiment_results.json").write_text(
            json.dumps(full_results, indent=2), encoding="utf-8"
        )
        self.generate_plots_and_report(full_results)
        print(f"\nReport & plots saved under: {self.out_dir}")
        return full_results
