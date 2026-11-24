"""Build self-contained Colab GPU notebook for Standalone PULSE-v8.1 Flagship 30M Verification."""
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
        ROOT / "benchmarks/pulse_v81_flagship_test.py",
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

    intro = """# PULSE-v8.1 最终旗舰合体版 · 单模型独立终验（`5:1 GDN+SWA+NoPE-MLA` + 原生 `FOMP-CMS v2.2`）

⚡ **支持任意 `sm_80+` GPU（`A100` / `L4` / `H100` 均可）**，零基线重复训练，直接复用 Drive 已训好的 30M 最强主干权重（`pulse_v7b_30m_final.pt`），点击 **全部运行（Run all）**，预计仅需 **~3–5 分钟（A100 约 ~2 分钟）** 并自动释放 GPU：

1. **零退化主干接入**：将 `59.17 / 57.31` PPL 的 `5:1` 主干权重无损载入 `PULSE-v8.1`（`39.77M <= TF 40.71M`，`4K` 显存仅 `2.00 MiB`，零隐藏参数审计通过）。
2. **保底差分学习率 `FOMP-CMS v2.2` 校准**：主干极低学习率（`8e-5`）+ `3× FineWeb-Edu` 回放 + `FOMP` 快权重高速学习（`1.5e-3`），彻底修复此前 Eval B5 主干困惑度偏移的问题。
3. **同一套单模型权重全项验收**：同时核验校准后的 `FineWeb 1K/4K PPL`（目标稳守 `~59–63`）、`4K` 零样本复现、`M=1 & M=16` 零词面重叠 1-Pass 2-Hop（`≥85%`）、强制 **ICML 2026 符号翻转归零**、`g_ext=0` 精确 `0.0000` 漂移与 `M=16` 全载 `1K & 4K` 漂移 `≤0.10 PPL`，并落盘最终旗舰权重 `PULSE_v81_flagship_30m.pt`。
- 结果写入 `Colab Notebooks/PULSE_AI/runs/pulse_v81_flagship_test/`。
"""

    unpack_cell = f"""import base64, hashlib, io, tempfile, zipfile
from pathlib import Path

source = base64.b64decode({base64.b64encode(payload).decode()!r})
code_sha256 = {sha!r}
assert hashlib.sha256(source).hexdigest() == code_sha256
work = Path(tempfile.mkdtemp(prefix="pulse-v81-flagship-", dir="/content"))
with zipfile.ZipFile(io.BytesIO(source)) as zf:
    zf.extractall(work)
print("Unpacked PULSE-v8.1 Flagship Test bundle:", code_sha256, "->", work, flush=True)
"""

    run_cell = """import json, os, subprocess, sys, time
from pathlib import Path
import torch
from google.colab import drive, runtime

#@title 运行配置（支持 NVIDIA A100 / L4 / H100，预计 ~2-5 分钟自动释放 GPU）
MAX_HOURS = 0.5 #@param {type:"number"}
CALIB_STEPS = 650 #@param {type:"integer"}
REQUIRE_BF16_GPU = True #@param {type:"boolean"}
AUTO_UNASSIGN_GPU = True #@param {type:"boolean"}

if not torch.cuda.is_available():
    raise RuntimeError("请先将 Colab 运行时切换到 NVIDIA A100 / L4 / H100 GPU。")
gpu_name = torch.cuda.get_device_name(0)
cc_major, cc_minor = torch.cuda.get_device_capability(0)
vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
print(f"GPU: {gpu_name} (sm_{cc_major}{cc_minor}, {vram_gb:.1f} GiB) | PyTorch: {torch.__version__}", flush=True)
if REQUIRE_BF16_GPU and cc_major < 8:
    raise RuntimeError(f"检测到当前 GPU 为 {gpu_name} (sm_{cc_major}{cc_minor})，请切至 A100 / L4 / H100 后重跑。")

started = time.time()
deadline = started + MAX_HOURS * 3600
drive.mount("/content/drive")

project = Path("/content/drive/MyDrive/Colab Notebooks/PULSE_AI")
fineweb_dir = project / "runs/review_v6_l4_v1/fineweb_edu"
backbone_ckpt = project / "runs/pulse_v8_stage_b_30m/checkpoints/pulse_v7b_30m_final.pt"
if not (fineweb_dir / "0000.train.u16").exists():
    raise FileNotFoundError(f"未找到 FineWeb-Edu 缓存分片: {fineweb_dir}")
if not backbone_ckpt.exists():
    raise FileNotFoundError(f"未找到预训练 5:1 主干权重: {backbone_ckpt}")

out_dir = project / "runs/pulse_v81_flagship_test"
out_dir.mkdir(parents=True, exist_ok=True)

env = dict(os.environ, PYTHONPATH=str(work / "src"))
status = {"gpu": gpu_name, "cc": f"sm_{cc_major}{cc_minor}", "torch": torch.__version__, "code_sha256": code_sha256, "started_utc": started, "stage": "setup"}
(out_dir / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
log_path = out_dir / "flagship_test.log"
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
    run_cmd(["-m", "pip", "-q", "install", "einops", "tiktoken"], 240)
    try:
        run_cmd(["-m", "pip", "-q", "install", "--no-deps", "fla-core"], 240)
    except Exception as exc:
        status["fla_install_warn"] = str(exc)

    cmd = [
        "-u", "benchmarks/pulse_v81_flagship_test.py",
        "--out", str(out_dir),
        "--fineweb-dir", str(fineweb_dir),
        "--backbone-ckpt", str(backbone_ckpt),
        "--steps", str(CALIB_STEPS),
        "--deadline", str(deadline - 60),
    ]
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
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
        },
    }
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            compile("".join(c["source"]), "<PULSE_v81_Flagship_Test>", "exec")
    out_nb = ROOT / "colab/PULSE_v81_Flagship_30M_Test.ipynb"
    out_nb.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Built {out_nb} ({out_nb.stat().st_size:,} bytes, sha256={sha[:12]}...)")
    return out_nb


if __name__ == "__main__":
    build_notebook()
