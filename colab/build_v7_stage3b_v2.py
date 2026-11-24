"""Build self-contained Colab L4 notebook for PULSE-v7 Stage 3b-v2 (Fair Joint Entity-Binding Adaptation + 8x Extrapolation)."""
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
        ROOT / "benchmarks/v7_stage3b_v2.py",
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

    intro = """# PULSE-v7 · Stage 3b-v2（墙 #4 破局：双架构同等实体绑定联合微调 + 8× 长度外推至 4K + DeCoRe）

选择 **NVIDIA L4** 运行时，点击 **全部运行（Run all）** 即可自动完成全部实验并释放 GPU（预计 **~6–9 分钟，约 0.5–0.8 CCU**）。

- 直接加载 Drive 中 **Stage 2 已训练好的 30M BF16 权重**（`runs/pulse_v7_stage2/checkpoints/seed41_*.pt`），**零重复预训练**。
- 所有结果原子写入 `Colab Notebooks/PULSE_AI/runs/pulse_v7_stage3b_v2/`，完整日志实时写入 `stage3b_v2_run.log`。
- **绝对公平零稻草人设置**：
  - `PULSE-v7B-30M` 与 `TF-30M` 使用**完全相同的 `<50K` 参数三分类验证器（`37,795` 参数，零零填充 `120` 维动态特征 + `h_fin`）**、完全相同的 `750` 步联合微调数据流（20% 短程课程 `{192,256}` -> `{192..512}`）、优化器与三项损失（`L_3way_verifier + 0.5*L_owner + 0.5*L_fineweb_replay`）。
  - **反作弊轨迹设计**：训练/验证/测试人名与物品词**100% 互斥**；FineWeb-TRAIN 与 FineWeb-VAL 干扰文本互斥；Class 2 含 **50% 转移后旧主人过期作废冲突**；Class 0 含 **50% 无关转移干扰句**；训练长度在 `{192..512}` 随机采样并完全保留未见动词模板（模板 3）。
  - **零样本外推评测**：仅训练在 `L <= 512`，直接零样本测试 **`L=512`**、**`L=512 (未见模板)`**、**`L=2048 (4×)`** 与 **`L=4096 (8×)`**，同步评测 **纯 GDN 原生信号消融**、**FineWeb PPL 漂移** 与 **`L=512 & L=4096` DeCoRe 对比解码干预**。
"""

    unpack_cell = f"""import base64, hashlib, io, tempfile, zipfile
from pathlib import Path

source = base64.b64decode({base64.b64encode(payload).decode()!r})
code_sha256 = {sha!r}
assert hashlib.sha256(source).hexdigest() == code_sha256
work = Path(tempfile.mkdtemp(prefix="pulse-v7-s3bv2-", dir="/content"))
with zipfile.ZipFile(io.BytesIO(source)) as zf:
    zf.extractall(work)
print("Unpacked PULSE-v7 Stage 3b-v2 bundle:", code_sha256, "->", work, flush=True)
"""

    run_cell = """import json, os, subprocess, sys, time
from pathlib import Path
import torch
from google.colab import drive, runtime

#@title 运行配置（推荐直接默认运行）
MAX_HOURS = 1.0 #@param {type:"number"}
STEPS = 750 #@param {type:"integer"}
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
ckpt_dir = project / "runs/pulse_v7_stage2/checkpoints"
out_dir = project / "runs/pulse_v7_stage3b_v2"
out_dir.mkdir(parents=True, exist_ok=True)
if not (fineweb_dir / "0000.train.u16").exists():
    raise FileNotFoundError(f"未找到缓存的 FineWeb-Edu 分片: {fineweb_dir}")
if not (ckpt_dir / "seed41_v7b_lt2_looped_30m.pt").exists():
    raise FileNotFoundError(f"未找到 Stage 2 训练好的 30M 权重: {ckpt_dir}")

env = dict(os.environ, PYTHONPATH=str(work / "src"))
status = {"gpu": gpu_name, "torch": torch.__version__, "code_sha256": code_sha256, "started_utc": started, "stage": "setup"}
(out_dir / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
log_path = out_dir / "stage3b_v2_run.log"
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
        "-u", "benchmarks/v7_stage3b_v2.py",
        "--out", str(out_dir),
        "--fineweb-dir", str(fineweb_dir),
        "--ckpt-dir", str(ckpt_dir),
        "--steps", str(STEPS),
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
            compile("".join(c["source"]), "<PULSE_v7_Stage3b_v2_L4>", "exec")
    out_nb = ROOT / "colab/PULSE_v7_Stage3b_v2_L4.ipynb"
    out_nb.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Built {out_nb} ({out_nb.stat().st_size:,} bytes, sha256={sha[:12]}...)")
    return out_nb


if __name__ == "__main__":
    build_notebook()
