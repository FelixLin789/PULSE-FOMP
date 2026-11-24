"""Build self-contained Colab L4 notebook for Week-2 Stage A: Route 2 (DC-ER) Iso-Time Screen."""
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
        ROOT / "benchmarks/whiteboard_week2_dcer.py",
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

    intro = """# PULSE 白板重构 · 第 2 周 Stage A 快筛：路线二 DC-ER（动态概念空间平衡态推理机）

选择 **NVIDIA L4** 运行时，点击 **全部运行（Run all）** 即可完成路线二 **`DC-ER`** 对决双强基线（`TF-Modern` 与 `PULSE-v7B-Ref`）的**严格等墙钟时间（Iso-Time）** Stage A 快筛并自动释放 GPU（预计 **~11–13 分钟，约 0.9–1.0 CCU**）。

- **严守蓝图反稻草人纪律（零隐藏参数，`DC-ER 3.66M <= TF-Modern 3.69M`）**：
  - **三段式原创架构**：2 层 `GDN-scalar` 快速词元编码器 + **解耦因果惊讶边界路由器**（H-Net 余弦跃变 + 截断因果惊讶度 + H-Net Eq.10 精确比率锁，零未来泄漏）+ 宽概念空间平衡态循环核（`d_con=320`，`2L × R=4 = 8` 概念层，严格套用 DLCM Decoupled μP `d_tok/d_con = 0.80` 学习率）+ 向量化 H-Net 因果 EMA 反分块 + 1 层薄词元解码器。
  - **公平等墙钟协议（Iso-Time）**：每个模型先跑 3 步不计时的 `torch.compile` / FLA 预热，再在完全相同的纯训练墙钟秒数下同台竞技，并实测精确 `ms/step` 墙钟倍率（蓝图止损线 `≤ 1.45× TF`）。
- **三项正交 Stage A 考核**：
  1. **T1 · `P5b Lano VarDensity CFG`（`45s/模型`）**：验证边界直方图向高熵转折点聚集（选择比 `>> 1.5`，对比均匀步长 `0.714`）、压缩率稳锁 `[3.2, 4.8]` 零漂移（止损线 `[2.5, 6.0]`），并实测 `DC-ER / TF` 墙钟比。
  2. **T2 · `P5b-Deep` 变密度 8 层概念栈（`75s/模型`）**：78% 变长确定性宏填充中嵌入 8 层 LIFO 后进先出概念栈与相邻模 4 组合，直测等时间下概念压缩循环对扁平基线的净领先。
  3. **T3 · `P1b Depo2-SC` 单大环多词元图遍历（`80s/模型`，5–7 词元/节点，词表 4，零 2/4 环捷径！）**：直测等时间下多跳精确匹配（`hop1/2/4/8 EM`）与概念边界对齐。
- 结果原子写入 `Colab Notebooks/PULSE_AI/runs/whiteboard_week2_dcer/`，实时日志 `week2_dcer.log`。
"""

    unpack_cell = f"""import base64, hashlib, io, tempfile, zipfile
from pathlib import Path

source = base64.b64decode({base64.b64encode(payload).decode()!r})
code_sha256 = {sha!r}
assert hashlib.sha256(source).hexdigest() == code_sha256
work = Path(tempfile.mkdtemp(prefix="pulse-w2-dcer-", dir="/content"))
with zipfile.ZipFile(io.BytesIO(source)) as zf:
    zf.extractall(work)
print("Unpacked PULSE Week-2 DC-ER bundle:", code_sha256, "->", work, flush=True)
"""

    run_cell = """import json, os, subprocess, sys, time
from pathlib import Path
import torch
from google.colab import drive, runtime

#@title 运行配置（路线二 DC-ER Stage A 严格 Iso-Time 快筛，约 11-13 分钟 / ~0.9-1.0 CCU）
MAX_HOURS = 0.60 #@param {type:"number"}
SEED = 41 #@param {type:"integer"}
AUTO_UNASSIGN_GPU = True #@param {type:"boolean"}

if not torch.cuda.is_available():
    raise RuntimeError("请先将 Colab 运行时切换到 NVIDIA L4 GPU。")
gpu_name = torch.cuda.get_device_name(0)
print(f"GPU: {gpu_name} | PyTorch: {torch.__version__}", flush=True)

started = time.time()
deadline = started + MAX_HOURS * 3600
drive.mount("/content/drive")

project = Path("/content/drive/MyDrive/Colab Notebooks/PULSE_AI")
out_dir = project / "runs/whiteboard_week2_dcer"
out_dir.mkdir(parents=True, exist_ok=True)

env = dict(os.environ, PYTHONPATH=str(work / "src"))
status = {"gpu": gpu_name, "torch": torch.__version__, "code_sha256": code_sha256, "started_utc": started, "stage": "setup"}
(out_dir / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
log_path = out_dir / "week2_dcer.log"
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
        "-u", "benchmarks/whiteboard_week2_dcer.py",
        "--out", str(out_dir),
        "--seed", str(SEED),
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
            compile("".join(c["source"]), "<PULSE_Whiteboard_Week2_Route2_DCER_L4>", "exec")
    out_nb = ROOT / "colab/PULSE_Whiteboard_Week2_Route2_DCER_L4.ipynb"
    out_nb.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Built {out_nb} ({out_nb.stat().st_size:,} bytes, sha256={sha[:12]}...)")
    return out_nb


if __name__ == "__main__":
    build_notebook()
