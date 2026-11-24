"""Build self-contained Colab L4 notebook for PULSE-v8 Stage B: 30M (100M-Token) Real-Corpus Showdown."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _cell(kind: str, src: str) -> dict:
    item = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
    if kind == "code":
        item.update(execution_count=None, outputs=[])
    return item


def build_notebook() -> Path:
    paths = sorted((ROOT / "src/pulse").glob("*.py")) + [
        ROOT / "benchmarks/v7_stage0_stage1.py",
        ROOT / "benchmarks/v7_stage2.py",
        ROOT / "benchmarks/v7_stage3.py",
        ROOT / "benchmarks/pulse_v8_stage_b_30m.py",
        ROOT / "tests/test_pulse_v8.py",
    ]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in paths:
            info = zipfile.ZipInfo(p.relative_to(ROOT).as_posix())
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, p.read_bytes())
    payload = buf.getvalue()
    sha = hashlib.sha256(payload).hexdigest()

    intro = """# PULSE-v8 第 3–4 周 · Stage B 终极真实语料决战（`30M` 级 / `100M Tokens` `FineWeb-Edu + 20% Code`）

⚡ **支持任意 `sm_80+` 原生 BF16 GPU（推荐 `NVIDIA A100` / `H100` / `L4`，自动适配显存微批大小且保持严格 `16,384 tok/step` 等价，已内置防老旧 T4 护栏）**，点击 **全部运行（Run all）**。
内置**按 Arm 与每 1,000 步原子断点续跑**（`checkpoints/<arm>_latest.pt`），支持任意中断零浪费续训，跑完自动释放 GPU：
- **NVIDIA A100（40GB/80GB）**：3 臂打满 `100M tokens` + 五项 4K 评测仅需约 **~30–40 分钟**！
- **NVIDIA L4（24GB）**：默认 3 臂总耗时约 **~90 分钟（~3.0 CCU）**（取消勾选 `INCLUDE_V7B_ABLATION` 仅跑双雄约 ~60 分钟 / ~2.0 CCU）。

1. **严格等时间（Iso-Time）+ 同参/更少参数（`PULSE-v8 39.57M <= TF-Modern 40.71M`）零稻草人预训练**：
   - **`TF-Modern-30M`**（`40.71M`，`12L d=384 H=6`，NoPE + SSMax + Canon + Gated SDPA + `torch.compile`）
   - **`PULSE-v8-30M`**（`39.57M`，`2× SWA-128 + 6× FLA GDN + 2× sm_89 Triton 64-Slot [L2+L10] + 2× NoPE-MLA + 原生 FOMP-v2.2`）
   - **`PULSE-v7B-30M`**（`39.43M`，严格对照消融：同一 12L 骨架去掉 64-Slot 与 FOMP）
   - 复用 Drive 已缓存的 `740M tokens` GPT-2 BPE `FineWeb-Edu` 分片 + 自动缓存 `20%` 真实开源代码流（无网/限流时安全回退至 `100% FineWeb-Edu`）。
2. **五大真语料 + 4K 终极评测（全在训练后的 30M 权重上执行）**：
   - **Eval B1**：`FineWeb-Edu` + `Code` 验证集 `1K (ID)` & `4K (4× OOD)` 困惑度 + 配对 Bootstrap 95% CI + 4K 工作态显存审计。
   - **Eval B2**：`1K & 4K` 真实文档零样本长程片段复现（Zero-Shot Repeated-Span Copy / Induction）。
   - **Eval B3**：`4K` 真实 `FineWeb-Edu` 散文内嵌多键检索（`K=16 → K=32 → K=64` 线性秩墙检验）。
   - **Eval B4**：`4K` 真实散文内嵌三分类反惯例实体重绑定（一致 / 合法状态转移 / 非法流畅偷换：`cc_mid`、`AUROC`、`Legal FAR`）。
   - **Eval B5**：真实英文词表零词面重叠在线规则编译（`PULSE-v8 FOMP-v2.2` vs `TF+LoRA+Replay` vs `TF+ICL`，含强制 ICML 2026 符号翻转检与 `FineWeb-Edu` 困惑度漂移审计）。
- 结果与检查点原子写入 `Colab Notebooks/PULSE_AI/runs/pulse_v8_stage_b_30m/`，实时日志 `stage_b_30m.log`。
"""

    unpack_cell = f"""import base64, hashlib, io, tempfile, zipfile
from pathlib import Path

source = base64.b64decode({base64.b64encode(payload).decode()!r})
code_sha256 = {sha!r}
assert hashlib.sha256(source).hexdigest() == code_sha256
work = Path(tempfile.mkdtemp(prefix="pulse-v8-stageb-", dir="/content"))
with zipfile.ZipFile(io.BytesIO(source)) as zf:
    zf.extractall(work)
print("Unpacked PULSE-v8 Stage B 30M bundle:", code_sha256, "->", work, flush=True)
"""

    run_cell = """import json, os, subprocess, sys, time
from pathlib import Path
import torch
from google.colab import drive, runtime

#@title 运行配置（支持 NVIDIA A100 / L4 / H100；支持任意中断后点 Run all 无缝断点续跑！）
MAX_HOURS = 2.4 #@param {type:"number"}
ISO_TIME_MIN_PER_ARM = 25.0 #@param {type:"number"}
MAX_TOKENS_PER_ARM = 100000000 #@param {type:"integer"}
USE_20PCT_CODE_MIX = True #@param {type:"boolean"}
INCLUDE_V7B_ABLATION = True #@param {type:"boolean"}
REQUIRE_BF16_GPU = True #@param {type:"boolean"}
AUTO_UNASSIGN_GPU = True #@param {type:"boolean"}

if not torch.cuda.is_available():
    raise RuntimeError("请先将 Colab 运行时切换到 NVIDIA A100 / L4 / H100 等 GPU。")
gpu_name = torch.cuda.get_device_name(0)
cc_major, cc_minor = torch.cuda.get_device_capability(0)
vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
print(f"GPU: {gpu_name} (sm_{cc_major}{cc_minor}, {vram_gb:.1f} GiB) | PyTorch: {torch.__version__}", flush=True)
if REQUIRE_BF16_GPU and cc_major < 8:
    raise RuntimeError(f"检测到当前 GPU 为 {gpu_name} (sm_{cc_major}{cc_minor}，不支持原生 BF16 / 99+ KiB SMEM)。请切换至 NVIDIA A100 / L4 / H100 等 Ampere+ GPU 后重跑。")

started = time.time()
deadline = started + MAX_HOURS * 3600
drive.mount("/content/drive")

project = Path("/content/drive/MyDrive/Colab Notebooks/PULSE_AI")
fineweb_dir = project / "runs/review_v6_l4_v1/fineweb_edu"
if not (fineweb_dir / "0000.train.u16").exists():
    raise FileNotFoundError(f"未找到 Drive 缓存的 FineWeb-Edu 分片: {fineweb_dir}")

out_dir = project / "runs/pulse_v8_stage_b_30m"
out_dir.mkdir(parents=True, exist_ok=True)

env = dict(os.environ, PYTHONPATH=str(work / "src"))
status = {"gpu": gpu_name, "cc": f"sm_{cc_major}{cc_minor}", "torch": torch.__version__, "code_sha256": code_sha256, "started_utc": started, "stage": "setup"}
(out_dir / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
log_path = out_dir / "stage_b_30m.log"
tail_lines = []

def run_cmd(args, timeout_s=600, tee_log=False):
    if not tee_log:
        subprocess.run([sys.executable, *args], check=True, env=env, cwd=work, timeout=max(30, min(timeout_s, deadline - time.time())))
        return
    with log_path.open("a", encoding="utf-8") as lf:
        lf.write("\\n=== CMD: " + str(args) + " (" + time.strftime("%Y-%m-%d %H:%M:%S") + ") ===\\n")
        lf.flush()
        proc = subprocess.Popen(
            [sys.executable, *args],
            env=env, cwd=work, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            lf.write(line)
            lf.flush()
            tail_lines.append(line.rstrip())
            if len(tail_lines) > 50:
                tail_lines.pop(0)
        rc = proc.wait(timeout=max(30, min(timeout_s, deadline - time.time())))
        if rc != 0:
            raise subprocess.CalledProcessError(rc, [sys.executable, *args])

try:
    run_cmd(["-m", "pip", "-q", "install", "einops", "tiktoken", "datasets"], 300)
    try:
        run_cmd(["-m", "pip", "-q", "install", "--no-deps", "fla-core"], 240)
    except Exception as exc:
        status["fla_install_warn"] = str(exc)

    cmd = [
        "-u", "benchmarks/pulse_v8_stage_b_30m.py",
        "--out", str(out_dir),
        "--fineweb-dir", str(fineweb_dir),
        "--iso-time-min", str(ISO_TIME_MIN_PER_ARM),
        "--max-tokens", str(MAX_TOKENS_PER_ARM),
        "--deadline", str(deadline - 90),
    ]
    if not USE_20PCT_CODE_MIX:
        cmd.append("--no-code-mix")
    if not INCLUDE_V7B_ABLATION:
        cmd.append("--no-v7b-ablation")
    status["stage"] = "running"
    (out_dir / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    run_cmd(cmd, int(MAX_HOURS * 3600), tee_log=True)
    status["stage"] = "complete"
except BaseException as exc:
    status.update(stage="error", error=f"{type(exc).__name__}: {exc}", last_log_tail=tail_lines[-30:])
    raise
finally:
    status["ended_utc"] = time.time()
    status["elapsed_min"] = round((status["ended_utc"] - started) / 60.0, 2)
    (out_dir / "status.json").write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Status saved ->", out_dir / "status.json", status, flush=True)
    if AUTO_UNASSIGN_GPU:
        print("Releasing Colab GPU runtime to save CCU...", flush=True)
        runtime.unassign()
"""

    nb = {
        "nbformat": 4,
        "nbformat_minor": 4,
        "cells": [_cell("markdown", intro), _cell("code", unpack_cell), _cell("code", run_cell)],
        "metadata": {
            "accelerator": "GPU",
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
        },
    }
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            compile("".join(c["source"]), "<PULSE_v8_StageB_30M_L4>", "exec")
    out_nb = ROOT / "colab/PULSE_v8_StageB_30M_L4.ipynb"
    out_nb.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Built {out_nb} ({out_nb.stat().st_size:,} bytes, sha256={sha[:12]}...)")
    return out_nb


if __name__ == "__main__":
    build_notebook()
