"""Build self-contained Colab L4 notebook for Week-2 Stage A Screening: Route 3 (FOMP-CMS) + Route 1 (APRM)."""
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
        ROOT / "tests/test_whiteboard_routes.py",
    ]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in paths:
            info = zipfile.ZipInfo(p.relative_to(ROOT).as_posix())
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, p.read_bytes())
    payload = buf.getvalue()
    sha = hashlib.sha256(payload).hexdigest()

    intro = """# PULSE 白板重构 · 第 2 周 Stage A 原创路线快筛：路线三（FOMP-CMS）+ 路线一（APRM）

选择 **NVIDIA L4** 运行时，点击 **全部运行（Run all）** 即可完成两条核心原创路线的 Stage A 铁面快筛并自动释放 GPU（预计 **~16–22 分钟，约 1.3–1.8 CCU**）。

- **严守反稻草人纪律（零隐藏参数，`<= TF-Modern 3.73M`）**：
  - **路线三 · `FOMP-CMS`（`3.67M`）**：一阶中层摊销快权重超网络（零二阶 HVP）+ 外部证据硬门控 `g_ext ∈ {0, 1}` + MeG 熟悉度门控 + 三频 CMS。
    - 直面 Week-1 **`P4 No-Hash Rule-2Hop`**（双强基线 `TF-Modern` / `PULSE-v7B-Ref` 1-Pass 两跳均崩至 **`18.75%`**，无门控自生成漂移 `+0.085–0.090 nats`）。
    - 同台检验 **单条（`M=1`）** 与 **16 条未见实体同时写入（`M=16`）** 下的零词面共享 View-B 两跳准确率（通关线 `≥ 75%`）、**ICML 2026 符号翻转自检**（反转 `-ΔW_mid` 必须暴跌 `≥ 50 pp`）、背景零遗忘与 `0.0000 nats` 自生成隔离。
  - **路线一 · `APRM`（`3.63M`）**：片上 64 槽位角色-填充寄存器（FP32 `16 KiB/head << L4 99 KiB SMEM`）+ 双组共享 KV（Lookup + Continuation）+ 2 层权重共享阻尼吸引子指针解引用核（`T=8` → `20` 有效层）。
    - **B1 (`P1a-SC` 单大环 16 节点 `Depo1` 8-hop，零短环捷径！)**：同等 2 阶段跳数课程下直面全新训练的 `TF-Modern` 与 `PULSE-v7B-Ref`；同步消融 **`Full BPTT` vs `Last-4 截断` vs `1-Step 隐式微分`**，输出**全循环步线性探针跳数阶梯**、槽位活跃熵（止损红线 `≥ 20%`）与 **`T=8 → T=12` 未见 `hop10` 外推**！
    - **B2 (`P1c DAG Arithmetic`)**、**B3 (`P2 S5-Retrieval 4×`)**、**B4 (`P5a Star-Graph` 首步规划)** 全面对决 Week-1 双强基线。
- 结果原子写入 `Colab Notebooks/PULSE_AI/runs/whiteboard_week2_screen/`，实时日志 `week2_screen.log`。
"""

    unpack_cell = f"""import base64, hashlib, io, tempfile, zipfile
from pathlib import Path

source = base64.b64decode({base64.b64encode(payload).decode()!r})
code_sha256 = {sha!r}
assert hashlib.sha256(source).hexdigest() == code_sha256
work = Path(tempfile.mkdtemp(prefix="pulse-w2-screen-", dir="/content"))
with zipfile.ZipFile(io.BytesIO(source)) as zf:
    zf.extractall(work)
print("Unpacked PULSE Week-2 Stage A bundle:", code_sha256, "->", work, flush=True)
"""

    run_cell = """import json, os, subprocess, sys, time
from pathlib import Path
import torch
from google.colab import drive, runtime

#@title 运行配置（默认同时跑路线三 FOMP-CMS + 路线一 APRM，约 1.3-1.8 CCU）
MAX_HOURS = 1.0 #@param {type:"number"}
ROUTES = "route3,route1" #@param {type:"string"}
SEEDS = "41,42,43" #@param {type:"string"}
AUTO_UNASSIGN_GPU = True #@param {type:"boolean"}

if not torch.cuda.is_available():
    raise RuntimeError("请先将 Colab 运行时切换到 NVIDIA L4 GPU。")
gpu_name = torch.cuda.get_device_name(0)
print(f"GPU: {gpu_name} | PyTorch: {torch.__version__}", flush=True)

started = time.time()
deadline = started + MAX_HOURS * 3600
drive.mount("/content/drive")

project = Path("/content/drive/MyDrive/Colab Notebooks/PULSE_AI")
out_dir = project / "runs/whiteboard_week2_screen"
out_dir.mkdir(parents=True, exist_ok=True)

env = dict(os.environ, PYTHONPATH=str(work / "src"))
status = {"gpu": gpu_name, "torch": torch.__version__, "code_sha256": code_sha256, "started_utc": started, "stage": "setup"}
(out_dir / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
log_path = out_dir / "week2_screen.log"
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
        "-u", "benchmarks/whiteboard_week2_screen.py",
        "--out", str(out_dir),
        "--routes", ROUTES,
        "--seeds", SEEDS,
        "--deadline", str(deadline - 90),
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
            "colab": {"gpuType": "L4"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
        },
    }
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            compile("".join(c["source"]), "<PULSE_Whiteboard_Week2_Routes1_3_L4>", "exec")
    out_nb = ROOT / "colab/PULSE_Whiteboard_Week2_Routes1_3_L4.ipynb"
    out_nb.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Built {out_nb} ({out_nb.stat().st_size:,} bytes, sha256={sha[:12]}...)")
    return out_nb


if __name__ == "__main__":
    build_notebook()
