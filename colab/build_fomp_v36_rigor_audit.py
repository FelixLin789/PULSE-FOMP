"""Build self-contained Colab A100/H800 notebook for Stage 2.5 Peer-Review Rigor Audit."""
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
        ROOT / "benchmarks/fomp_v3_open_edit.py",
        ROOT / "benchmarks/fomp_v34_stress_ood.py",
        ROOT / "benchmarks/fomp_v35_llama3_8b_sota.py",
        ROOT / "benchmarks/fomp_v36_rigor_audit.py",
        ROOT / "tests/test_fomp_v3.py",
        ROOT / "tests/test_fomp_v36_rigor_audit.py",
    ]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in paths:
            info = zipfile.ZipInfo(p.relative_to(ROOT).as_posix())
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, p.read_bytes())
    payload = buf.getvalue()
    sha = hashlib.sha256(payload).hexdigest()

    intro = """# Stage 2.5 · 顶会审稿人级硬伤终极补考（`IKE-RAG` 同类基线 + `0% MQuAKE` 零样本消融 + `n=200` 配对 McNemar 检验）

⚡ **推荐使用 Colab `A100 40GB / 80GB`**，预计全套 6 臂 **~55 分钟**，**自动复用云盘里已训好的 `fomp_v36_llama3_8b.pt` 与 `4.1GB` AlphaEdit 零空间矩阵 `P`**，**按臂原子断点续跑**，跑完自动释放 GPU！

一次性彻底回应全部三条顶会审稿人硬伤：
1. **补齐同赛道最强检索基线（`IKE / Dense RAG + Prompt`）**：使用 `BAAI/bge-small-en-v1.5`（33M 参数，大于我方 20.9M）同台测试 **始终注入 Top-2 (`ike_rag_always`)** 与 **`cf_train` 严格校准阈值门控 (`ike_rag_gated`)**，完整统计准确率、Locality、查询延迟与额外 Prompt Token 税；
2. **彻底消除 MQuAKE 领域内训练争议**：新增 **`FOMP-v3.6-ZeroMQ`（严格 `0%` MQuAKE 训练数据 + `0%` ZsRE 训练数据）**，与无训练的 `AlphaEdit` / `IKE-RAG` 100% 公平零样本硬刚，并同表展示带 MQuAKE 训练的 `FOMP-v3.6` 消融差值；
3. **高统计功效大样本配对检验**：CounterFact **`n = 200`** (`2×100`)、MQuAKE **`n = 192`** (`3×64`，从 32 条扩大 6 倍！)、Zero-Shot ZsRE **`n = 200`** (`2×100`)，**100% 全量贪婪生成 EM**，输出逐条配对精确 **McNemar $p$ 值** 与 **Wilson 95% 置信区间**！

结果保存至 `Colab Notebooks/PULSE_AI/runs/fomp_v36_rigor_audit/`。
"""

    unpack_cell = f"""import base64, hashlib, io, tempfile, zipfile
from pathlib import Path

source = base64.b64decode({base64.b64encode(payload).decode()!r})
code_sha256 = {sha!r}
assert hashlib.sha256(source).hexdigest() == code_sha256
work = Path(tempfile.mkdtemp(prefix="fomp-v36-rigor-", dir="/content"))
with zipfile.ZipFile(io.BytesIO(source)) as zf:
    zf.extractall(work)
print("Unpacked Stage 2.5 Rigor Audit bundle:", code_sha256, "->", work, flush=True)
"""

    run_cell = """import json, os, subprocess, sys, time
from pathlib import Path
import torch
from google.colab import drive, runtime

#@title Stage 2.5 顶会级硬伤补考配置（自动复用 Stage 2 的 fomp_v36 权重与 AlphaEdit P 矩阵！）
MODEL_ID = "NousResearch/Meta-Llama-3-8B-Instruct" #@param {type:"string"}
IKE_MODEL_ID = "BAAI/bge-small-en-v1.5" #@param {type:"string"}
ARMS = "base,ike_rag_always,ike_rag_gated,fomp_v36,fomp_v36_zero_mq,alphaedit" #@param {type:"string"}
FORCE_ARMS = "" #@param {type:"string"}
FOMP_TRAIN_STEPS = 2000 #@param {type:"integer"}
MAX_HOURS = 1.8 #@param {type:"number"}
AUTO_UNASSIGN_GPU = True #@param {type:"boolean"}

if not torch.cuda.is_available():
    raise RuntimeError("请先将 Colab 运行时切换到 NVIDIA A100 (40GB+) GPU。")
gpu_name = torch.cuda.get_device_name(0)
vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
print(f"GPU: {gpu_name} ({vram_gb:.1f} GiB) | PyTorch: {torch.__version__}", flush=True)

started = time.time()
deadline = started + MAX_HOURS * 3600
drive.mount("/content/drive")

project = Path("/content/drive/MyDrive/Colab Notebooks/PULSE_AI")
stage2_dir = project / "runs/fomp_v35_llama3_8b_sota"
out_dir = project / "runs/fomp_v36_rigor_audit"
out_dir.mkdir(parents=True, exist_ok=True)
easyedit_dir = Path("/content/EasyEdit")

env = dict(os.environ, PYTHONPATH=str(work / "src"), PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", TOKENIZERS_PARALLELISM="false")
status = {"gpu": gpu_name, "vram_gb": round(vram_gb, 1), "torch": torch.__version__, "model_id": MODEL_ID, "arms": ARMS, "code_sha256": code_sha256, "started_utc": started, "stage": "setup"}
(out_dir / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
log_path = out_dir / "rigor_audit.log"
tail_lines = []

def run_cmd(cmd_list, timeout_s=600, tee_log=False):
    if not tee_log:
        subprocess.run(cmd_list, check=True, env=env, cwd=work, timeout=max(30, min(timeout_s, deadline - time.time())))
        return
    with log_path.open("a", encoding="utf-8") as lf:
        lf.write("\\n=== CMD: " + str(cmd_list) + " (" + time.strftime("%Y-%m-%d %H:%M:%S") + ") ===\\n")
        lf.flush()
        proc = subprocess.Popen(cmd_list, env=env, cwd=work, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
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
            raise subprocess.CalledProcessError(rc, cmd_list)

try:
    run_cmd([sys.executable, "-m", "pip", "-q", "install", "transformers", "datasets", "accelerate", "einops", "pyyaml", "hydra-core", "omegaconf", "higher", "nltk", "sentencepiece", "pytest"], 300)
    run_cmd([sys.executable, "-m", "pytest", "tests/test_fomp_v3.py", "tests/test_fomp_v36_rigor_audit.py", "-q"], 120)

    if "alphaedit" in ARMS.lower():
        if not easyedit_dir.exists():
            run_cmd(["git", "clone", "--depth", "1", "https://github.com/zjunlp/EasyEdit.git", str(easyedit_dir)], 180)
        (easyedit_dir / "easyeditor/__init__.py").write_text("# trimmed eager root imports; algorithm code 100% unmodified\\n", encoding="utf-8")
        (easyedit_dir / "easyeditor/models/__init__.py").write_text("# trimmed eager model imports; algorithm code 100% unmodified\\n", encoding="utf-8")

    cmd = [
        sys.executable, "-u", "benchmarks/fomp_v36_rigor_audit.py",
        "--out", str(out_dir),
        "--stage2-dir", str(stage2_dir),
        "--model-id", MODEL_ID,
        "--ike-model-id", IKE_MODEL_ID,
        "--easyedit-dir", str(easyedit_dir),
        "--arms", ARMS,
        "--force-arms", FORCE_ARMS,
        "--fomp-train-steps", str(FOMP_TRAIN_STEPS),
        "--deadline", str(deadline - 120),
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
    if (out_dir / "rigor_audit_report.md").exists():
        print("\\n" + (out_dir / "rigor_audit_report.md").read_text(encoding="utf-8"), flush=True)
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
            compile("".join(c["source"]), "<FOMP_v36_Stage25_Rigor_Audit>", "exec")
    out_nb = ROOT / "colab/FOMP_v36_Stage25_Rigor_Audit.ipynb"
    out_nb.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Built {out_nb} ({out_nb.stat().st_size:,} bytes, sha256={sha[:12]}...)")
    return out_nb


if __name__ == "__main__":
    build_notebook()
