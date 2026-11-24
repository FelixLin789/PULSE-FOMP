"""Build self-contained Colab L4 notebook for Week-3 PULSE-v8 sm_89 Triton 64-Slot Bench & Gate-3 Check."""
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
        ROOT / "benchmarks/whiteboard_probes.py",
        ROOT / "benchmarks/whiteboard_week2_screen.py",
        ROOT / "benchmarks/whiteboard_week2_v2.py",
        ROOT / "benchmarks/pulse_v8_gate_check.py",
        ROOT / "benchmarks/pulse_v8_triton_bench.py",
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

    intro = """# PULSE-v8 第 3 周 · L4 `sm_89` 99 KiB SMEM 融合 Triton 64-Slot 核测速与 `L0=SWA` 终验

⚠️ **请务必切至 `NVIDIA L4` 运行时（已内置防 T4 护栏）**，点击 **全部运行（Run all）**，预计仅需 **~4–6 分钟（~0.35–0.5 CCU）** 并自动释放 GPU：

1. **Check 1 · 融合 Triton 前向 + 反向数值精确性审计**：在 `3M (D=32, T=256)`、非对齐长度 `T=137` 与 **`30M Stage B (D=64, T=256)`** 三组规格上，逐项比对 `O, a_mean` 与全部 7 个梯度 `(dQ, dR, dF, dGamma, dK0, dInvTau, dBFree)` 对 PyTorch 参考实现的相对误差（IEEE 模式严格 `< 2e-4`）。
2. **Check 2 · 真机 NVIDIA L4 提速与全模型端到端墙钟比（`3M` + `30M`）**：
   - 单层 `64-Slot` 提速比（PyTorch 分块循环 vs `sm_89` 片上 SRAM 驻留融合 Triton 核，覆及 `L=256 / 1024 / 4096`）。
   - 全模型端到端训练步时（`PULSE-v8` 块级编译 + Triton 核 vs 同参编译 `TF-Modern`）在 **`3M L=256`**、**`30M L=1024`**、**`30M L=4096`** 的实机墙钟比（目标 `L=256..1024 ≤ 1.15× TF`、`L=4096` 反超 `TF`！）。
3. **Check 3 · `L0=SWA-32` 统一 `PULSE-v8` 复验 Gate-3 `P4 No-Hash Rule-2Hop`**：检验 `L0=SWA` 是否修复 `GDN` 的 2-token 前缀模糊，把 `VA` 拉回 `95-100%` 并将 `M=16` 背景漂移清零至 `≤ 1.00 pp`（过关即自动确认 3 种子并正式具备发车 `30M / 100M tokens` Stage B 的全部硬件与算法通行证！）。
- 结果原子写入 `Colab Notebooks/PULSE_AI/runs/pulse_v8_triton_bench/`，实时日志 `triton_bench.log`。
"""

    unpack_cell = f"""import base64, hashlib, io, tempfile, zipfile
from pathlib import Path

source = base64.b64decode({base64.b64encode(payload).decode()!r})
code_sha256 = {sha!r}
assert hashlib.sha256(source).hexdigest() == code_sha256
work = Path(tempfile.mkdtemp(prefix="pulse-v8-triton-", dir="/content"))
with zipfile.ZipFile(io.BytesIO(source)) as zf:
    zf.extractall(work)
print("Unpacked PULSE-v8 Week-3 Triton Bench bundle:", code_sha256, "->", work, flush=True)
"""

    run_cell = """import json, os, subprocess, sys, time
from pathlib import Path
import torch
from google.colab import drive, runtime

#@title 运行配置（必须使用 NVIDIA L4，预计 ~4-6 分钟 / ~0.35-0.5 CCU）
MAX_HOURS = 0.40 #@param {type:"number"}
AUTO_3SEED_ON_GATE3_PASS = True #@param {type:"boolean"}
REQUIRE_BF16_GPU = True #@param {type:"boolean"}
AUTO_UNASSIGN_GPU = True #@param {type:"boolean"}

if not torch.cuda.is_available():
    raise RuntimeError("请先将 Colab 运行时切换到 NVIDIA L4 GPU。")
gpu_name = torch.cuda.get_device_name(0)
cc_major, cc_minor = torch.cuda.get_device_capability(0)
print(f"GPU: {gpu_name} (sm_{cc_major}{cc_minor}) | PyTorch: {torch.__version__}", flush=True)
if REQUIRE_BF16_GPU and cc_major < 8:
    raise RuntimeError(f"检测到当前 GPU 为 {gpu_name} (sm_{cc_major}{cc_minor}，非原生 BF16/99 KiB SMEM)。请切换为 NVIDIA L4 后重跑。")

started = time.time()
deadline = started + MAX_HOURS * 3600
drive.mount("/content/drive")

project = Path("/content/drive/MyDrive/Colab Notebooks/PULSE_AI")
out_dir = project / "runs/pulse_v8_triton_bench"
out_dir.mkdir(parents=True, exist_ok=True)

env = dict(os.environ, PYTHONPATH=str(work / "src"))
status = {"gpu": gpu_name, "torch": torch.__version__, "code_sha256": code_sha256, "started_utc": started, "stage": "setup"}
(out_dir / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
log_path = out_dir / "triton_bench.log"
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
            if len(tail_lines) > 40:
                tail_lines.pop(0)
        rc = proc.wait(timeout=max(30, min(timeout_s, deadline - time.time())))
        if rc != 0:
            raise subprocess.CalledProcessError(rc, [sys.executable, *args])

try:
    run_cmd(["-m", "pip", "-q", "install", "einops"], 240)
    try:
        run_cmd(["-m", "pip", "-q", "install", "--no-deps", "fla-core"], 240)
    except Exception as exc:
        status["fla_install_warn"] = str(exc)

    cmd = [
        "-u", "benchmarks/pulse_v8_triton_bench.py",
        "--out", str(out_dir),
        "--deadline", str(deadline - 60),
    ]
    if not AUTO_3SEED_ON_GATE3_PASS:
        cmd.append("--no-auto-3seed")
    status["stage"] = "running"
    (out_dir / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    run_cmd(cmd, int(MAX_HOURS * 3600), tee_log=True)
    status["stage"] = "complete"
except BaseException as exc:
    status.update(stage="error", error=f"{type(exc).__name__}: {exc}", last_log_tail=tail_lines[-25:])
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
            "colab": {"gpuType": "L4"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
        },
    }
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            compile("".join(c["source"]), "<PULSE_v8_Week3_TritonBench_L4>", "exec")
    out_nb = ROOT / "colab/PULSE_v8_Week3_TritonBench_L4.ipynb"
    out_nb.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Built {out_nb} ({out_nb.stat().st_size:,} bytes, sha256={sha[:12]}...)")
    return out_nb


if __name__ == "__main__":
    build_notebook()
