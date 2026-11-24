"""Build self-contained Colab L4 notebook for PULSE-v7 Stage 0 (Base & L4 Speed) + Stage 1 (Rank-Wall 4-Way Showdown)."""
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

    intro = """# PULSE-v7 · Stage 0.3b（L4 提速排雷）+ Stage 0.4（30M 3 种子）+ Stage 1（并发秩墙四路对决）

选择 **NVIDIA L4** 运行时，点击 **全部运行（Run all）** 即可无损续跑（自动跳过已完成的 `stage0_backend/storage/speed_105m`），跑完自动释放 GPU（预计 **~20–25 分钟，约 1.7–2.1 CCU**）。

- 全程日志实时同步至 `Colab Notebooks/PULSE_AI/runs/pulse_v7_stage01/stage01_run.log`，单臂异常隔离记录、绝不中断后续实验。
- **Stage 0.3b**：`torch._dynamo.explain` graph-break 审计 + 去掉 `3840` 通道 `Canon-D` 与 50K 词表 `softcap tanh` 后的精简 `v7b_fast` 在 `L=512` 与 `L=2048` 正面硬刚 `tf_105m`。
- **Stage 0.4**：30M FineWeb-Edu 3 种子（`41/42/43`），对比 `v7b_fast2/fast4/kda`、`v7a_o1` 与 `tf_30m`（过关红线 `PPL <= 480`）。
- **Stage 1**：2L 噪声干草堆（`d_head=32`）5 路对决（`TF-FullKV` vs `v7B: GDN-2 + Gated NoPE-MLA` vs `Triadic E=4` / `Toaster 128` / `RAM-Net S=128 Top-8`）。
"""

    unpack_cell = f"""import base64, hashlib, io, tempfile, zipfile
from pathlib import Path

source = base64.b64decode({base64.b64encode(payload).decode()!r})
code_sha256 = {sha!r}
assert hashlib.sha256(source).hexdigest() == code_sha256
work = Path(tempfile.mkdtemp(prefix="pulse-v7-", dir="/content"))
with zipfile.ZipFile(io.BytesIO(source)) as zf:
    zf.extractall(work)
print("Unpacked PULSE-v7 bundle:", code_sha256, "->", work, flush=True)
"""

    run_cell = """import json, os, subprocess, sys, time
from pathlib import Path
import torch
from google.colab import drive, runtime

#@title 运行配置（推荐直接默认运行）
MAX_HOURS = 1.5 #@param {type:"number"}
RUN_STAGE_0 = True #@param {type:"boolean"}
RUN_STAGE_1 = True #@param {type:"boolean"}
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
out_dir = project / "runs/pulse_v7_stage01"
out_dir.mkdir(parents=True, exist_ok=True)
if not (fineweb_dir / "0000.train.u16").exists():
    raise FileNotFoundError(f"未找到缓存的 FineWeb-Edu 分片: {fineweb_dir}")

env = dict(os.environ, PYTHONPATH=str(work / "src"))
status = {"gpu": gpu_name, "torch": torch.__version__, "code_sha256": code_sha256, "started_utc": started, "stage": "setup"}
(out_dir / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")

log_path = out_dir / "stage01_run.log"
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

    cmd = ["-u", "benchmarks/v7_stage0_stage1.py", "--out", str(out_dir), "--fineweb-dir", str(fineweb_dir), "--deadline", str(deadline - 60)]
    if not RUN_STAGE_0:
        cmd.append("--skip-stage0")
    if not RUN_STAGE_1:
        cmd.append("--skip-stage1")
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
            compile("".join(c["source"]), "<PULSE_v7_Stage0_Stage1_L4>", "exec")
    out_nb = ROOT / "colab/PULSE_v7_Stage0_Stage1_L4.ipynb"
    out_nb.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Built {out_nb} ({out_nb.stat().st_size:,} bytes, sha256={sha[:12]}...)")
    return out_nb


if __name__ == "__main__":
    build_notebook()
