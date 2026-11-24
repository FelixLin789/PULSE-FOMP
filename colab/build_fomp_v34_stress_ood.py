"""Build self-contained Colab GPU notebook for FOMP-v3.4 Stage 1 Stress & OOD Suite (M=1024 + ZsRE + Rel-Disjoint OOD)."""
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

    intro = """# FOMP-v3.5 补丁验证 · 消除 `\\nAnswer:` 模板捷径（`S0 混合M=128` + `S1 M=1024` + `S2 零样本 ZsRE`）

⚡ **支持 `L4`（约 ~10–12 分钟）或 `A100`（约 ~5 分钟）**，点击 **全部运行（Run all）**，跑完自动释放 GPU：

- **v3.5 核心补丁（0% ZsRE 训练数据，严格零样本！）**：
  1. 给所有 QA 格式正样本（MQuAKE + QA 包装 CounterFact）配 **同格式 `Question: ...?\\nAnswer:` 负样本**；
  2. 在 `128×128` 对比矩阵上加入 **全矩阵交叉负样本硬边界惩罚**，从数学上彻底堵死 InfoNCE 常数平移漏洞（强行让 `\\nAnswer:` 模板外壳相似度归零）；
  3. 实体滑窗拓宽至 `W=18`，覆盖更长问句。
- **一次性全验证**：
  - **S0**：CounterFact + MQuAKE 2-Hop 混合 `M=128`（确认 2-Hop EM 与 81% Retain 零回退）
  - **S1**：千条并发 `M = 128 → 256 → 512 → 1,024`
  - **S2**：跨数据集零样本 `CrowdMind/ZsRE`（`M = 1, 16, 100, 500`，重点看 **`Locality` 从 `2.9%` 暴拉到 `90%+`** 与 **`Gen-EM` 飙升**！）

结果与 `fomp_v35.pt` 原子保存至 `Colab Notebooks/PULSE_AI/runs/fomp_v35_stage1_verify/`。
"""

    unpack_cell = f"""import base64, hashlib, io, tempfile, zipfile
from pathlib import Path

source = base64.b64decode({base64.b64encode(payload).decode()!r})
code_sha256 = {sha!r}
assert hashlib.sha256(source).hexdigest() == code_sha256
work = Path(tempfile.mkdtemp(prefix="fomp-v34-ood-", dir="/content"))
with zipfile.ZipFile(io.BytesIO(source)) as zf:
    zf.extractall(work)
print("Unpacked FOMP-v3.4 Stage 1 Stress & OOD bundle:", code_sha256, "->", work, flush=True)
"""

    run_cell = """import json, os, subprocess, sys, time
from pathlib import Path
import torch
from google.colab import drive, runtime

#@title FOMP-v3.5 补丁全验证（L4 约 10–12 分钟，跑完自动释放 GPU）
MODEL_ID = "Qwen/Qwen2.5-1.5B" #@param {type:"string"}
MAIN_TRAIN_STEPS = 1500 #@param {type:"integer"}
SKIP_S3 = True #@param {type:"boolean"}
OOD_TRAIN_STEPS = 1200 #@param {type:"integer"}
MAX_HOURS = 1.0 #@param {type:"number"}
REQUIRE_BF16_GPU = True #@param {type:"boolean"}
AUTO_UNASSIGN_GPU = True #@param {type:"boolean"}

if not torch.cuda.is_available():
    raise RuntimeError("请先将 Colab 运行时切换到 NVIDIA L4 / A100 / H100 GPU。")
gpu_name = torch.cuda.get_device_name(0)
cc_major, cc_minor = torch.cuda.get_device_capability(0)
vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
print(f"GPU: {gpu_name} (sm_{cc_major}{cc_minor}, {vram_gb:.1f} GiB) | PyTorch: {torch.__version__}", flush=True)
if REQUIRE_BF16_GPU and cc_major < 8:
    raise RuntimeError(f"检测到当前 GPU 为 {gpu_name} (sm_{cc_major}{cc_minor})，请切换至 L4 / A100 / H100 后重跑。")

started = time.time()
deadline = started + MAX_HOURS * 3600
drive.mount("/content/drive")

project = Path("/content/drive/MyDrive/Colab Notebooks/PULSE_AI")
out_dir = project / "runs/fomp_v35_stage1_verify"
out_dir.mkdir(parents=True, exist_ok=True)

env = dict(os.environ, PYTHONPATH=str(work / "src"))
status = {"gpu": gpu_name, "cc": f"sm_{cc_major}{cc_minor}", "torch": torch.__version__, "model_id": MODEL_ID, "code_sha256": code_sha256, "started_utc": started, "stage": "setup"}
(out_dir / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
log_path = out_dir / "stage1_stress_ood.log"
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
    run_cmd(["-m", "pip", "-q", "install", "transformers", "datasets", "accelerate", "einops", "pytest"], 300)
    run_cmd(["-m", "pytest", "tests/test_fomp_v3.py", "-q"], 120)
    cmd = [
        "-u", "benchmarks/fomp_v34_stress_ood.py",
        "--out", str(out_dir),
        "--model-id", MODEL_ID,
        "--main-train-steps", str(MAIN_TRAIN_STEPS),
        "--ood-train-steps", str(OOD_TRAIN_STEPS),
        "--deadline", str(deadline - 60),
    ]
    if SKIP_S3:
        cmd.append("--skip-s3")
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
            compile("".join(c["source"]), "<FOMP_v34_Stage1_Stress_OOD>", "exec")
    out_nb = ROOT / "colab/FOMP_v34_Stage1_Stress_OOD.ipynb"
    out_nb.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Built {out_nb} ({out_nb.stat().st_size:,} bytes, sha256={sha[:12]}...)")
    return out_nb


if __name__ == "__main__":
    build_notebook()
