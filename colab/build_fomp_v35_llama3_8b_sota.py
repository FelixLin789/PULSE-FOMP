"""Build self-contained Colab A100/H800 notebook for Stage 2 Llama-3-8B 6-Arm SOTA Showdown."""
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
        ROOT / "tests/test_fomp_v3.py",
    ]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in paths:
            info = zipfile.ZipInfo(p.relative_to(ROOT).as_posix())
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, p.read_bytes())
    payload = buf.getvalue()
    sha = hashlib.sha256(payload).hexdigest()

    intro = """# Stage 2 · `Llama-3-8B-Instruct` 顶会级 6 臂同台大榜（`FOMP-v3.5` vs 官方 `AlphaEdit / WISE / GRACE / MEMIT`）

⚡ **推荐使用 Colab `A100 40GB`（或 `H100 / H800 80GB`）**，预计全套 6 臂 **~35–45 分钟**，**按臂原子断点续跑**（每跑完一个方法立刻落盘到 Google Drive，断线重跑自动跳过已完成项），跑完自动释放 GPU！

- **100% 同底座（免审批公开镜像 `NousResearch/Meta-Llama-3-8B-Instruct`）、同 BOS 分词、同切分零稻草人对照**：
  1. **`Base`**：原始未编辑 `Llama-3-8B-Instruct`
  2. **`FOMP-v3.5`（我方）**：双锁 2 层残差 MLP + 格式无关反捷径训练 + 零梯度单次编译（额外加测 **`M=500 / 1,000`** 千条并发与 **ICML 2026 符号翻转归零检**）
  3. **`GRACE`** *(NeurIPS 2023, `zjunlp/EasyEdit` 官方代码)*
  4. **`WISE`** *(NeurIPS 2024, `zjunlp/EasyEdit` 官方代码)*
  5. **`AlphaEdit`** *(ICLR 2025 Outstanding Paper, `zjunlp/EasyEdit` 官方代码 + GPU 加速零空间投影)*
  6. **`MEMIT`** *(ICLR 2023, `zjunlp/EasyEdit` 官方代码)*
- **三大权威赛道同时出表**：
  - **Track 1 · CounterFact (`M = 100`)**：Efficacy / Paraphrase / Neighborhood Retain / WikiText-2 PPL Drift / 写入耗时
  - **Track 2 · MQuAKE-Remastered 多跳 (`M = 64`)**：Single-Hop + 2-Hop Multi-Hop Reasoning
  - **Track 3 · 零样本 ZsRE (`M = 100`)**：FOMP-v3.5 在 **0% ZsRE 训练数据**下直接硬刚官方编辑基线！

结果 JSON 与自动排版论文级 Markdown 报告保存至 `Colab Notebooks/PULSE_AI/runs/fomp_v35_llama3_8b_sota/`。
"""

    unpack_cell = f"""import base64, hashlib, io, tempfile, zipfile
from pathlib import Path

source = base64.b64decode({base64.b64encode(payload).decode()!r})
code_sha256 = {sha!r}
assert hashlib.sha256(source).hexdigest() == code_sha256
work = Path(tempfile.mkdtemp(prefix="fomp-v35-8b-sota-", dir="/content"))
with zipfile.ZipFile(io.BytesIO(source)) as zf:
    zf.extractall(work)
print("Unpacked Stage 2 Llama-3-8B SOTA Showdown bundle:", code_sha256, "->", work, flush=True)
"""

    run_cell = """import json, os, subprocess, sys, time
from pathlib import Path
import torch
from google.colab import drive, runtime

#@title Stage 2 运行配置（需 A100 40GB / H100 / H800，按臂断点续跑；已完成臂自动秒跳！）
MODEL_ID = "NousResearch/Meta-Llama-3-8B-Instruct" #@param {type:"string"}
ARMS = "base,fomp_v36,grace,wise,alphaedit,memit" #@param {type:"string"}
FORCE_ARMS = "" #@param {type:"string"}
FOMP_TRAIN_STEPS = 2000 #@param {type:"integer"}
M_EDIT = 100 #@param {type:"integer"}
ALPHAEDIT_COV_SAMPLES = 20000 #@param {type:"integer"}
MAX_HOURS = 1.8 #@param {type:"number"}
AUTO_UNASSIGN_GPU = True #@param {type:"boolean"}

if not torch.cuda.is_available():
    raise RuntimeError("请先将 Colab 运行时切换到 NVIDIA A100 (40GB+) GPU。")
gpu_name = torch.cuda.get_device_name(0)
cc_major, cc_minor = torch.cuda.get_device_capability(0)
vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
print(f"GPU: {gpu_name} (sm_{cc_major}{cc_minor}, {vram_gb:.1f} GiB) | PyTorch: {torch.__version__}", flush=True)
if vram_gb < 35.0:
    print(f"WARNING: 当前显存 {vram_gb:.1f} GiB < 35 GiB！运行 AlphaEdit/WISE (Llama-3-8B) 极易 OOM，强烈建议切换至 A100 40GB！", flush=True)

started = time.time()
deadline = started + MAX_HOURS * 3600
drive.mount("/content/drive")

project = Path("/content/drive/MyDrive/Colab Notebooks/PULSE_AI")
out_dir = project / "runs/fomp_v35_llama3_8b_sota"
out_dir.mkdir(parents=True, exist_ok=True)
easyedit_dir = Path("/content/EasyEdit")

env = dict(os.environ, PYTHONPATH=str(work / "src"), PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", TOKENIZERS_PARALLELISM="false")
status = {"gpu": gpu_name, "vram_gb": round(vram_gb, 1), "torch": torch.__version__, "model_id": MODEL_ID, "arms": ARMS, "code_sha256": code_sha256, "started_utc": started, "stage": "setup"}
(out_dir / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
log_path = out_dir / "llama3_8b_sota.log"
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
    # 1. Install lean deps (never overwrite Colab CUDA torch!)
    run_cmd([sys.executable, "-m", "pip", "-q", "install", "transformers", "datasets", "accelerate", "einops", "pyyaml", "hydra-core", "omegaconf", "higher", "nltk", "sentencepiece", "pytest"], 300)
    run_cmd([sys.executable, "-m", "pytest", "tests/test_fomp_v3.py", "-q"], 120)

    # 2. Shallow-clone official zjunlp/EasyEdit & strip unused multimodal/API eager imports in __init__.py only
    if not easyedit_dir.exists():
        run_cmd(["git", "clone", "--depth", "1", "https://github.com/zjunlp/EasyEdit.git", str(easyedit_dir)], 180)
    (easyedit_dir / "easyeditor/__init__.py").write_text("# trimmed eager root imports; algorithm code 100% unmodified\\n", encoding="utf-8")
    (easyedit_dir / "easyeditor/models/__init__.py").write_text("# trimmed eager model imports; algorithm code 100% unmodified\\n", encoding="utf-8")
    # Quick import smoke test on the 4 official EasyEdit modules before loading 8B weights!
    run_cmd([
        sys.executable, "-c",
        f"import sys; sys.path.insert(0, {str(easyedit_dir)!r}); "
        "from easyeditor.models.grace.GRACE import GRACE; "
        "from easyeditor.models.wise.WISE import WISE; "
        "from easyeditor.models.alphaedit.AlphaEdit_main import apply_AlphaEdit_to_model; "
        "from easyeditor.models.memit.memit_main import apply_memit_to_model; "
        "print('Verified all 4 official EasyEdit modules imported cleanly!')"
    ], 120)

    # 3. Run full 6-arm resumable Stage 2 showdown!
    cmd = [
        sys.executable, "-u", "benchmarks/fomp_v35_llama3_8b_sota.py",
        "--out", str(out_dir),
        "--model-id", MODEL_ID,
        "--easyedit-dir", str(easyedit_dir),
        "--arms", ARMS,
        "--force-arms", FORCE_ARMS,
        "--fomp-train-steps", str(FOMP_TRAIN_STEPS),
        "--m-edit", str(M_EDIT),
        "--alphaedit-cov-samples", str(ALPHAEDIT_COV_SAMPLES),
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
    if (out_dir / "llama3_8b_sota_report.md").exists():
        print("\\n" + (out_dir / "llama3_8b_sota_report.md").read_text(encoding="utf-8"), flush=True)
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
            compile("".join(c["source"]), "<FOMP_v35_Llama3_8B_SOTA_Showdown>", "exec")
    out_nb = ROOT / "colab/FOMP_v35_Llama3_8B_SOTA_Showdown.ipynb"
    out_nb.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Built {out_nb} ({out_nb.stat().st_size:,} bytes, sha256={sha[:12]}...)")
    return out_nb


if __name__ == "__main__":
    build_notebook()
