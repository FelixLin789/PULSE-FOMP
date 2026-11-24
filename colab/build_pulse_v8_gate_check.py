"""Build self-contained Colab L4 notebook for PULSE-v8 3M Unified Promotion Gate-Check."""
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
        ROOT / "benchmarks/pulse_v8_gate_check.py",
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

    intro = """# PULSE-v8（对称 `4×GDN → 64-Slot → NoPE-MLA` + `FOMP-v2.2`）L4 终审快测

⚠️ **请务必在菜单栏「代码执行程序 → 更改运行时类型」中选中 `NVIDIA L4`（脚本已内置防误用 T4 护栏！）**，然后点击 **全部运行（Run all）**，预计仅需 **~7–9 分钟（~0.6–0.75 CCU）** 并自动释放 GPU：

- **架构升级**：将片上 `64-Slot` 寄存器置于 `L4`（紧贴 `L5 NoPE-MLA` 输出端），形成规整对称的 **`[4× GDN-scalar → 1× 64-Slot → 1× NoPE-MLA]`** 宏观积木（`3.65M <= TF-Modern 3.71M`，`30M` Stage B 直接堆叠 2 组 12 层！），既彻底避免中后层 GDN MLP 覆盖 64-Slot 的反惯例读数（冲刺 `P3 cc_mid ≥ 95%` + `FAR = 0%` + `AUROC ≥ 0.98`），又让 `P4` 的早层编码 `L0..L1` 成为纯高速 `GDN`！
- **三关全检（全部基于真机 NVIDIA L4 `sm_89` 原生 BF16）**：
  1. **Gate 1 · `P3 Alias-Swap 4×`（`L=256 → 1024`）** + 实机 `PULSE-v8 / TF-Modern` 墙钟比标定（~2.0 分钟）；
  2. **Gate 2 · `P2 S5-Retrieval 4×`（`K=16 → K=64`）**（~1.0 分钟）；
  3. **Gate 3 · `P4 No-Hash Rule-2Hop`（2 层 `rel_gate` MLP + 3× 合批，种子 41 过关自动补跑 42/43 确认）**（~4–5 分钟）。
- 结果原子写入 `Colab Notebooks/PULSE_AI/runs/pulse_v8_gate_l4_v2/`，实时日志 `pulse_v8_gate.log`。
"""

    unpack_cell = f"""import base64, hashlib, io, tempfile, zipfile
from pathlib import Path

source = base64.b64decode({base64.b64encode(payload).decode()!r})
code_sha256 = {sha!r}
assert hashlib.sha256(source).hexdigest() == code_sha256
work = Path(tempfile.mkdtemp(prefix="pulse-v8-gate-", dir="/content"))
with zipfile.ZipFile(io.BytesIO(source)) as zf:
    zf.extractall(work)
print("Unpacked PULSE-v8 3M Gate-Check bundle:", code_sha256, "->", work, flush=True)
"""

    run_cell = """import json, os, subprocess, sys, time
from pathlib import Path
import torch
from google.colab import drive, runtime

#@title 运行配置（必须使用 NVIDIA L4，预计 ~7-9 分钟 / ~0.6-0.75 CCU）
MAX_HOURS = 0.50 #@param {type:"number"}
SEED = 41 #@param {type:"integer"}
AUTO_3SEED_ON_GATE3_PASS = True #@param {type:"boolean"}
REQUIRE_BF16_GPU = True #@param {type:"boolean"}
AUTO_UNASSIGN_GPU = True #@param {type:"boolean"}

if not torch.cuda.is_available():
    raise RuntimeError("请先将 Colab 运行时切换到 NVIDIA L4 GPU。")
gpu_name = torch.cuda.get_device_name(0)
print(f"GPU: {gpu_name} | PyTorch: {torch.__version__}", flush=True)
if REQUIRE_BF16_GPU and not torch.cuda.is_bf16_supported():
    raise RuntimeError(f"检测到当前 GPU 为 {gpu_name}（不支持原生 BF16，会比 L4 慢 9 倍！）。请在菜单栏「代码执行程序 → 更改运行时类型」切换为 NVIDIA L4 后重跑。")

started = time.time()
deadline = started + MAX_HOURS * 3600
drive.mount("/content/drive")

project = Path("/content/drive/MyDrive/Colab Notebooks/PULSE_AI")
out_dir = project / "runs/pulse_v8_gate_l4_v2"
out_dir.mkdir(parents=True, exist_ok=True)

env = dict(os.environ, PYTHONPATH=str(work / "src"))
status = {"gpu": gpu_name, "torch": torch.__version__, "code_sha256": code_sha256, "started_utc": started, "stage": "setup"}
(out_dir / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
log_path = out_dir / "pulse_v8_gate.log"
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
        "-u", "benchmarks/pulse_v8_gate_check.py",
        "--out", str(out_dir),
        "--seed", str(SEED),
        "--deadline", str(deadline - 90),
    ]
    if not AUTO_3SEED_ON_GATE3_PASS:
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
            compile("".join(c["source"]), "<PULSE_v8_3M_GateCheck_L4>", "exec")
    out_nb = ROOT / "colab/PULSE_v8_3M_GateCheck_L4.ipynb"
    out_nb.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Built {out_nb} ({out_nb.stat().st_size:,} bytes, sha256={sha[:12]}...)")
    return out_nb


if __name__ == "__main__":
    build_notebook()
