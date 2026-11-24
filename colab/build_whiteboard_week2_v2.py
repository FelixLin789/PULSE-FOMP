"""Build self-contained Colab L4 notebook for Week-2 v2 Fast Truth-Test: FOMP-CMS v2 + APRM v2 vs TF-Modern."""
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

    intro = """# PULSE 白板重构 · 第 2 周 v2 极速双修验真：FOMP-CMS v2 + APRM v2 vs TF-Modern

选择 **NVIDIA L4** 运行时，点击 **全部运行（Run all）** 即可完成两项精准修复后的极速验真并自动释放 GPU（单种子约 **~8–10 分钟 / ~0.7–0.8 CCU**；若 Route 3 种子 41 过关则自动加跑 2 个确认种子，总计约 **~12–14 分钟 / ~1.0–1.1 CCU**）。

- **Part 1 · 路线三 `FOMP-CMS v2`（`3.66M <= TF 3.72M`，100% 一阶自微）**：
  - 修复 v1 跨视图割裂、黑盒记忆 MLP 与软门泄漏三大病根：早层共享 `key_proj`（InfoNCE 零泄漏对齐）+ 实体无关双线性外积编译 + 锐利 ReLU 阈值门控 + 关系条件门控 `rel_gate`（保护同实体 `r2`/`ViewEq` 关系不被误伤）。
  - 在严格未见 `test_ents (48..63)` 上直测 **`M=1` & `M=16` 零重叠 View-B 1-Hop / 1-Pass 2-Hop**（通关线 `≥ 75%`）、**ICML 2026 Sign-Flip 反转暴跌 `≥ 50 pp`**、`M=16` 背景漂移 `≤ 1.0 pp` 与 `g_ext=0` 严格 `0.0000 nats`。
- **Part 2 · 路线一 `APRM v2` vs `TF-Modern`（`P1a-SC` 单大环 16 节点 `Depo1` 8-Hop，种子 41）**：
  - 修复 v1 `delta_norm` 200 倍初始化噪放 bug + 升级全位置严格因果分块槽快照。
  - **严格公平协议**：`TF-Modern`（`3.72M`）、`APRM v2 Full BPTT`（`3.63M`）、`APRM v2 Last-4 Truncated` 三者使用完全相同的**零答案泄漏 `Q=8` 多查询训练**（纯 `Q=1` 评测）、**掌握度门控四阶段课程（`≥90%` 才升级）**与 **WSD `AdamW(lr=1e-3)`**。
  - 完整输出 `hop1..hop10` 准确率、**全层/全循环步线性探针跳数阶梯**、槽位活跃熵与 **`T=8 → T=12` 未见 `hop10` 推理时循环外推**。
- 结果原子写入 `Colab Notebooks/PULSE_AI/runs/whiteboard_week2_v2/`，实时日志 `week2_v2.log`。
"""

    unpack_cell = f"""import base64, hashlib, io, tempfile, zipfile
from pathlib import Path

source = base64.b64decode({base64.b64encode(payload).decode()!r})
code_sha256 = {sha!r}
assert hashlib.sha256(source).hexdigest() == code_sha256
work = Path(tempfile.mkdtemp(prefix="pulse-w2-v2-", dir="/content"))
with zipfile.ZipFile(io.BytesIO(source)) as zf:
    zf.extractall(work)
print("Unpacked PULSE Week-2 v2 bundle:", code_sha256, "->", work, flush=True)
"""

    run_cell = """import json, os, subprocess, sys, time
from pathlib import Path
import torch
from google.colab import drive, runtime

#@title 运行配置（默认单种子极速验真 + Route 3 过关自动 3 种子确认，约 0.8-1.1 CCU）
MAX_HOURS = 0.75 #@param {type:"number"}
PARTS = "route3,route1" #@param {type:"string"}
SEED = 41 #@param {type:"integer"}
AUTO_3SEED_ON_ROUTE3_PASS = True #@param {type:"boolean"}
AUTO_UNASSIGN_GPU = True #@param {type:"boolean"}

if not torch.cuda.is_available():
    raise RuntimeError("请先将 Colab 运行时切换到 NVIDIA L4 GPU。")
gpu_name = torch.cuda.get_device_name(0)
print(f"GPU: {gpu_name} | PyTorch: {torch.__version__}", flush=True)

started = time.time()
deadline = started + MAX_HOURS * 3600
drive.mount("/content/drive")

project = Path("/content/drive/MyDrive/Colab Notebooks/PULSE_AI")
out_dir = project / "runs/whiteboard_week2_v2"
out_dir.mkdir(parents=True, exist_ok=True)

env = dict(os.environ, PYTHONPATH=str(work / "src"))
status = {"gpu": gpu_name, "torch": torch.__version__, "code_sha256": code_sha256, "started_utc": started, "stage": "setup"}
(out_dir / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
log_path = out_dir / "week2_v2.log"
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
        "-u", "benchmarks/whiteboard_week2_v2.py",
        "--out", str(out_dir),
        "--parts", PARTS,
        "--seed", str(SEED),
        "--deadline", str(deadline - 90),
    ]
    if not AUTO_3SEED_ON_ROUTE3_PASS:
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
            compile("".join(c["source"]), "<PULSE_Whiteboard_Week2_v2_L4>", "exec")
    out_nb = ROOT / "colab/PULSE_Whiteboard_Week2_v2_L4.ipynb"
    out_nb.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Built {out_nb} ({out_nb.stat().st_size:,} bytes, sha256={sha[:12]}...)")
    return out_nb


if __name__ == "__main__":
    build_notebook()
