"""Build self-contained Colab L4 notebook for Week-1 PULSE Whiteboard Anvil (P1-P5 + Dual Strong Baselines)."""
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
        ROOT / "tests/test_whiteboard_anvil.py",
    ]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in paths:
            info = zipfile.ZipInfo(p.relative_to(ROOT).as_posix())
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, p.read_bytes())
    payload = buf.getvalue()
    sha = hashlib.sha256(payload).hexdigest()

    intro = """# PULSE 白板重构 · 第 1 周“五大最小分离基准 + 双强基线”铁砧（Week-1 Anvil）

选择 **NVIDIA L4** 运行时，点击 **全部运行（Run all）** 即可完成全部 5 大分离探针双强基线摸底并自动释放 GPU（预计 **~12–16 分钟，约 1.0–1.3 CCU**）。

- **双强对称基线（严格零稻草人 & 零隐藏参数审计）**：
  - **`TF-Modern`（3.75M）**：拆矩阵 Q/K/V/O（FairMuonAdamW 严格分组）+ QK-Norm + Gated SDPA + NoPE + SSMax + Canon-ACD + SwiGLU。
  - **`PULSE-v7B-Ref`（3.67M ≤ TF）**：GDN-Scalar + Gated SWA-32（`W=32 << L_train` 堵死滑窗捷径）+ NoPE-MLA + SSMax + Canon-A。
- **反顿悟（Anti-Grokking）扫参协议**：每个探针均在 `seed 41` 上做 `3` 组学习率扫参（`5e-4 / 1.5e-3 / 3e-3`），随后在最优学习率上跑满 **`3` 个随机种子（`41, 42, 43`）**，同步记录真实 L4 墙钟耗时与全项精确 FLOPs。
- **五大正交最小分离探针 + 自动化审计**：
  - **P1**：`Depo1`（1-Pass 8 跳置换图，禁 CoT + **逐层线性探针跳数阶梯审计**）、`Depo2`（每节点 5–7 词元、词表 4）、`DAG Arithmetic`（增长型寄存器多跳求值）。
  - **P2**：`S5-Retrieval`（非阿贝尔 $S_5$ 追踪 × `K=16→64` 高熵键值检索，4× 长度外推至 `L=768`）。
  - **P3**：`Alias-Swap`（90% 强统计惯例 vs 无惯例控制组 `zz`，测中段反惯例准确率、先验偏移 nats、4× 长度外推与三元状态转移 `AUROC / FAR<=10%`）。
  - **P4**：`No-Hash Rule-2Hop`（零词面共享 ViewA→ViewB 测试期中层 GD、1-Pass 两跳、**ICML 2026 梯度符号翻转自检**与自生成毒化漂移）。
  - **P5**：`Star-Graph`（有向星型图首步短视规划）+ `Lano CFG` 变密度语流与边界直方图审计基线。
- 所有结果逐探针原子写入 `Colab Notebooks/PULSE_AI/runs/whiteboard_week1_anvil/`，完整日志实时写入 `anvil_run.log`。
"""

    unpack_cell = f"""import base64, hashlib, io, tempfile, zipfile
from pathlib import Path

source = base64.b64decode({base64.b64encode(payload).decode()!r})
code_sha256 = {sha!r}
assert hashlib.sha256(source).hexdigest() == code_sha256
work = Path(tempfile.mkdtemp(prefix="pulse-anvil-w1-", dir="/content"))
with zipfile.ZipFile(io.BytesIO(source)) as zf:
    zf.extractall(work)
print("Unpacked PULSE Week-1 Anvil bundle:", code_sha256, "->", work, flush=True)
"""

    run_cell = """import json, os, subprocess, sys, time
from pathlib import Path
import torch
from google.colab import drive, runtime

#@title 运行配置（默认严格控制在 <= 1.5 CCU 内跑完全部 P1-P5）
MAX_HOURS = 1.0 #@param {type:"number"}
PROBES = "P1,P2,P3,P4,P5" #@param {type:"string"}
LR_STEPS = 350 #@param {type:"integer"}
TRAIN_STEPS = 1500 #@param {type:"integer"}
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
out_dir = project / "runs/whiteboard_week1_anvil"
out_dir.mkdir(parents=True, exist_ok=True)

env = dict(os.environ, PYTHONPATH=str(work / "src"))
status = {"gpu": gpu_name, "torch": torch.__version__, "code_sha256": code_sha256, "started_utc": started, "stage": "setup"}
(out_dir / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
log_path = out_dir / "anvil_run.log"
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
        "-u", "benchmarks/whiteboard_probes.py",
        "--out", str(out_dir),
        "--probes", PROBES,
        "--lr-steps", str(LR_STEPS),
        "--train-steps", str(TRAIN_STEPS),
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
            compile("".join(c["source"]), "<PULSE_Whiteboard_Week1_Anvil_L4>", "exec")
    out_nb = ROOT / "colab/PULSE_Whiteboard_Week1_Anvil_L4.ipynb"
    out_nb.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Built {out_nb} ({out_nb.stat().st_size:,} bytes, sha256={sha[:12]}...)")
    return out_nb


if __name__ == "__main__":
    build_notebook()
