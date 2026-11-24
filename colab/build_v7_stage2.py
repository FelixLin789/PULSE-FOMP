"""Build self-contained Colab L4 notebook for PULSE-v7 Stage 1b (SSMax 8K Passkey) + Stage 2 (LT2 2-Loop Full BPTT)."""
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
        ROOT / "tests/test_pulse_v7.py",
    ]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in paths:
            info = zipfile.ZipInfo(p.relative_to(ROOT).as_posix())
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, p.read_bytes())
    payload = buf.getvalue()
    sha = hashlib.sha256(payload).hexdigest()

    intro = """# PULSE-v7 · Stage 1b（SSMax 8K Passkey 补强）+ Stage 2（LT2 2-Loop Full BPTT 隐态循环深度）

选择 **NVIDIA L4** 运行时，点击 **全部运行（Run all）** 即可自动完成验证并释放 GPU（预计 **~25–30 分钟，约 2.0–2.5 CCU**）。

- 自动复用 Drive 中已缓存的 `FineWeb-Edu` 分片（零下载）及 `Stage 0` 的 `tf_30m` 种子 42/43 基线，并自动保存种子 41 的 BF16 权重供 **Stage 3** 直接复用（零重复训练）。
- 全程结果原子保存至 `Colab Notebooks/PULSE_AI/runs/pulse_v7_stage2/`，完整日志同步写入 `stage2_run.log`。
- **Stage 1b（~2 min）**：验证 `GatedNoPEMLABlock` 加入 Scalable-Softmax (`s_h * ln(pos + 2)`) 后，8K Passkey 能否从 `81.2%` 拉升至 `>= 98%` 且保持 `L=4096 K=64 >= 98%`。
- **Stage 2a（~8 min）**：6L 等参（`PULSE <= TF`）硬刚 $TC^0$ 深度墙：`Swap-S5 (32->256)`、`Parity (40->256)`、`Multi-Hop 1-Pass Pointer Chasing (2/4/6/8-hop)` + 测试时循环扩展 `r = 1, 2, 4, 6`。
- **Stage 2b（~15 min）**：30M FineWeb-Edu 3 种子（`41/42/43`），同步测 **Iso-FLOP（step 384）** 与 **Iso-Param（step 512）** PPL + STARS `r = 1, 2, 4, 6` 稳定性体检。
"""

    unpack_cell = f"""import base64, hashlib, io, tempfile, zipfile
from pathlib import Path

source = base64.b64decode({base64.b64encode(payload).decode()!r})
code_sha256 = {sha!r}
assert hashlib.sha256(source).hexdigest() == code_sha256
work = Path(tempfile.mkdtemp(prefix="pulse-v7-s2-", dir="/content"))
with zipfile.ZipFile(io.BytesIO(source)) as zf:
    zf.extractall(work)
print("Unpacked PULSE-v7 Stage 2 bundle:", code_sha256, "->", work, flush=True)
"""

    run_cell = """import json, os, subprocess, sys, time
from pathlib import Path
import torch
from google.colab import drive, runtime

#@title 运行配置（推荐直接默认运行）
MAX_HOURS = 1.5 #@param {type:"number"}
RUN_STAGE_1B = True #@param {type:"boolean"}
RUN_STAGE_2A = True #@param {type:"boolean"}
RUN_STAGE_2B = True #@param {type:"boolean"}
AUTO_UNASSIGN_GPU = True #@param {type:"boolean"}

if not torch.cuda.is_available():
    raise RuntimeError("请先将 Colab 运行时切换到 NVIDIA L4 GPU。")
gpu_name = torch.cuda.get_device_name(0)
print(f"GPU: {gpu_name} | PyTorch: {torch.__version__}", flush=True)

started = time.time()
deadline = started + MAX_HOURS * 3600
drive.mount("/content/drive")

project = Path("/content/drive/MyDrive/Colab Notebooks/PULSE_AI")
fineweb_dir = project / "runs/review_v6_l4_v1/fineweb_edu"
stage01_dir = project / "runs/pulse_v7_stage01"
out_dir = project / "runs/pulse_v7_stage2"
out_dir.mkdir(parents=True, exist_ok=True)
if not (fineweb_dir / "0000.train.u16").exists():
    raise FileNotFoundError(f"未找到缓存的 FineWeb-Edu 分片: {fineweb_dir}")

env = dict(os.environ, PYTHONPATH=str(work / "src"))
status = {"gpu": gpu_name, "torch": torch.__version__, "code_sha256": code_sha256, "started_utc": started, "stage": "setup"}
(out_dir / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
log_path = out_dir / "stage2_run.log"
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
        "-u", "benchmarks/v7_stage2.py",
        "--out", str(out_dir),
        "--fineweb-dir", str(fineweb_dir),
        "--stage01-dir", str(stage01_dir),
        "--deadline", str(deadline - 60),
    ]
    if not RUN_STAGE_1B:
        cmd.append("--skip-stage1b")
    if not RUN_STAGE_2A:
        cmd.append("--skip-stage2a")
    if not RUN_STAGE_2B:
        cmd.append("--skip-stage2b")
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
            compile("".join(c["source"]), "<PULSE_v7_Stage2_L4>", "exec")
    out_nb = ROOT / "colab/PULSE_v7_Stage2_L4.ipynb"
    out_nb.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Built {out_nb} ({out_nb.stat().st_size:,} bytes, sha256={sha[:12]}...)")
    return out_nb


if __name__ == "__main__":
    build_notebook()
