"""Build self-contained Colab GPU notebook for FOMP-v3 Open-Domain 4-Arm Benchmark (Qwen2.5-1.5B)."""
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

    intro = """# FOMP-v3 开放域自然语言多跳编辑 · 同底座四臂硬碰硬对照（`Qwen2.5-1.5B` + `CounterFact` + `MQuAKE-Remastered`）

⚡ **支持任意 `sm_80+` GPU（`A100` / `L4` / `H100` 均可）**，点击 **全部运行（Run all）**，预计 **A100 约 ~8–12 分钟 / L4 约 ~15–22 分钟**，跑完自动释放 GPU：

- **公开真实数据集（严格 0% 实体重叠切分）**：`azhx/counterfact`（真反事实编辑、改写泛化、邻近实体零误伤）+ `henryzhongsc/MQuAKE-Remastered`（真世界知识 2-Hop 多跳编辑）+ `Salesforce/wikitext`（通用语言建模 PPL 遗忘检测）。
- **四臂完全同底座（冻结 `Qwen2.5-1.5B`）、同测试集零稻草人对照**：
  1. **`Base`**：原始未编辑 `Qwen2.5-1.5B`
  2. **`ICL`**：把全部 `M=1, 16, 64, 128` 条事实拼进 Prompt（测准确率 + 上下文延迟税）
  3. **`LoRA (r=8) + Replay`**：在中后层（`L10..L27`）对 `M=16, 64, 128` 条事实做梯度微调 + WikiText-2 回放（测准确率 + 邻近实体误伤与 PPL 遗忘）
  4. **`FOMP-v3.4`（我方 · 双锁 2 层 MLP + 跨跳桥接实体路由）**：**锁 1（关系+语境 2 层残差 MLP） × 锁 2（无监督实体显著性 `W=14` + 句尾隐含实体桥接 2 层残差 MLP，`min_ent_tau=0.32`）** + 陡峭饱和门控（`kappa=0.03`）+ 双倍训练数据与全多句改写采样；默认 **`FOMP_ONLY=True` 快速模式（L4 约 8 分钟）**，直接同口径对接已测完的 Base / ICL / LoRA 基准！
- 结果与权重原子保存至 `Colab Notebooks/PULSE_AI/runs/fomp_v34_open_edit/`。
"""

    unpack_cell = f"""import base64, hashlib, io, tempfile, zipfile
from pathlib import Path

source = base64.b64decode({base64.b64encode(payload).decode()!r})
code_sha256 = {sha!r}
assert hashlib.sha256(source).hexdigest() == code_sha256
work = Path(tempfile.mkdtemp(prefix="fomp-v3-bench-", dir="/content"))
with zipfile.ZipFile(io.BytesIO(source)) as zf:
    zf.extractall(work)
print("Unpacked FOMP-v3 Open-Domain Benchmark bundle:", code_sha256, "->", work, flush=True)
"""

    run_cell = """import json, os, subprocess, sys, time
from pathlib import Path
import torch
from google.colab import drive, runtime

#@title 运行配置（默认 FOMP_ONLY=True 仅测我方 v3.4 + Base 核验，L4 约 8 分钟！）
MODEL_ID = "Qwen/Qwen2.5-1.5B" #@param {type:"string"}
TRAIN_STEPS = 1500 #@param {type:"integer"}
FOMP_ONLY = True #@param {type:"boolean"}
MAX_HOURS = 1.0 #@param {type:"number"}
REQUIRE_BF16_GPU = True #@param {type:"boolean"}
AUTO_UNASSIGN_GPU = True #@param {type:"boolean"}

if not torch.cuda.is_available():
    raise RuntimeError("请先将 Colab 运行时切换到 NVIDIA A100 / L4 / H100 GPU。")
gpu_name = torch.cuda.get_device_name(0)
cc_major, cc_minor = torch.cuda.get_device_capability(0)
vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
print(f"GPU: {gpu_name} (sm_{cc_major}{cc_minor}, {vram_gb:.1f} GiB) | PyTorch: {torch.__version__}", flush=True)
if REQUIRE_BF16_GPU and cc_major < 8:
    raise RuntimeError(f"检测到当前 GPU 为 {gpu_name} (sm_{cc_major}{cc_minor})，请切换至 A100 / L4 / H100 后重跑。")

started = time.time()
deadline = started + MAX_HOURS * 3600
drive.mount("/content/drive")

project = Path("/content/drive/MyDrive/Colab Notebooks/PULSE_AI")
out_dir = project / "runs/fomp_v34_open_edit"
out_dir.mkdir(parents=True, exist_ok=True)

env = dict(os.environ, PYTHONPATH=str(work / "src"))
status = {"gpu": gpu_name, "cc": f"sm_{cc_major}{cc_minor}", "torch": torch.__version__, "model_id": MODEL_ID, "code_sha256": code_sha256, "started_utc": started, "stage": "setup"}
(out_dir / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
log_path = out_dir / "fomp_v3_open_edit.log"
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
        "-u", "benchmarks/fomp_v3_open_edit.py",
        "--out", str(out_dir),
        "--model-id", MODEL_ID,
        "--train-steps", str(TRAIN_STEPS),
        "--deadline", str(deadline - 60),
    ]
    if FOMP_ONLY:
        cmd.append("--fomp-only")
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
            compile("".join(c["source"]), "<FOMP_v3_OpenEdit_Benchmark>", "exec")
    out_nb = ROOT / "colab/FOMP_v3_OpenEdit_Benchmark.ipynb"
    out_nb.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Built {out_nb} ({out_nb.stat().st_size:,} bytes, sha256={sha[:12]}...)")
    return out_nb


if __name__ == "__main__":
    build_notebook()
