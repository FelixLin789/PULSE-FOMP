"""Execution cell embedded in the PULSE v6 Colab notebook."""

import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import torch
from google.colab import drive, runtime

#@title L4 与本次计算单元预算
# 4.8 沿用旧实验的估算，Colab 面板显示其他数值时可改动。
CCU_PER_HOUR = 4.8 #@param {type:"number"}
TOTAL_CCU = 145.0 #@param {type:"number"}
SESSION_HOURS = 2.0 #@param {type:"number"}

if not torch.cuda.is_available() or "L4" not in torch.cuda.get_device_name():
    raise RuntimeError("请将 Colab 运行时切换到 NVIDIA L4。")
if not 0 < TOTAL_CCU <= 145 or CCU_PER_HOUR <= 0 or not 0 < SESSION_HOURS <= 12:
    raise ValueError("请检查总预算、L4 费率及单次运行时长。")

started = time.time()
drive.mount("/content/drive")
project = Path("/content/drive/MyDrive/Colab Notebooks/PULSE_AI")
legacy = project / "runs/exp_pulse_v53_micro2k"
out = project / "runs/review_v6_l4_v1"
if not (legacy / "checkpoints").is_dir():
    raise FileNotFoundError(legacy / "checkpoints")
out.mkdir(parents=True, exist_ok=True)
meter, lease, receipt = (out / name for name in ("budget.json", "active_session.json", "status.json"))

def read(path, fallback):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else fallback

def write(path, payload):
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)

entry = read(lease, {})
if entry and time.time() < entry["expires"] + 60:
    raise RuntimeError("另一个会话正在运行这组实验。")
if entry:
    recovered = read(meter, {"sessions": {}})
    recovered["sessions"].setdefault(entry["id"], entry["reserved_ccu"])
    write(meter, recovered)
    lease.unlink()

ledger = read(meter, {"sessions": {}})
available = TOTAL_CCU - sum(ledger["sessions"].values())
session_duration = min(SESSION_HOURS * 3600, available / CCU_PER_HOUR * 3600)
if session_duration < 300:
    runtime.unassign()
    raise SystemExit("本组实验剩余的估算预算不足五分钟。检查点已保留。")
deadline = started + session_duration
session_id = uuid.uuid4().hex
entry = {"id": session_id, "expires": deadline, "rate": CCU_PER_HOUR,
         "reserved_ccu": session_duration / 3600 * CCU_PER_HOUR}
with lease.open("x", encoding="utf-8") as f:
    json.dump(entry, f)
ledger["sessions"][session_id] = entry["reserved_ccu"]
write(meter, ledger)

status = {"stage": "setup", "source_sha256": code_sha256, "gpu": torch.cuda.get_device_name(),
          "torch": torch.__version__, "started": started, "session_id": session_id}
write(receipt, status)
env = dict(os.environ, PYTHONPATH=str(work / "src") + os.pathsep + str(work / "benchmarks"))

def command(args, limit):
    subprocess.run([sys.executable, *args], check=True, env=env, cwd=work,
                   timeout=max(1, min(limit, deadline - time.time())))

try:
    command(["-m", "pip", "-q", "install", "tiktoken", "huggingface_hub", "pyarrow", "einops"], 360)
    try:
        command(["-m", "pip", "-q", "install", "--no-deps", "fla-core"], 240)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        status["fla_install_error"] = str(exc)

    status["stage"] = "pre105_audit"
    write(receipt, status)
    command(["-u", "benchmarks/review_v6_pre105.py", "--out", str(out),
             "--legacy-run", str(legacy), "--rate", str(CCU_PER_HOUR), "--deadline", str(deadline - 45)],
            session_duration)
    evidence = out / "pre105"
    mqar = read(evidence / "mqar.json", {})
    iso = read(evidence / "iso_time.json", {})
    small = read(out / "small_ablation.json", {})
    rls = read(evidence / "rls.json", {})
    profiler = read(evidence / "profiler.json", {})
    status.update(mqar_short_control_passed=mqar.get("short_context_passed"),
                  fineweb_small_complete=small.get("complete", False), iso_time_complete=iso.get("complete", False),
                  rls_target_passed=rls.get("passed"),
                  profiler_eager_complete=all("cuda" in profiler.get(f"{name}.eager", {})
                                              for name in ("v6_no_attnres", "transformer")),
                  mqar_distance_complete=(evidence / "distance.json").exists(), scale_paused=True)
    status["stage"] = ("pre105_complete" if small.get("complete") and iso.get("complete")
                       and rls and status["profiler_eager_complete"] and status["mqar_distance_complete"]
                       and mqar.get("short_context_passed") else "pre105_saved")
except BaseException as exc:
    status.update(stage="stopped", error=str(exc))
    raise
finally:
    status["ended"] = time.time()
    spent = (status["ended"] - started) / 3600 * CCU_PER_HOUR
    ledger = read(meter, {"sessions": {}})
    ledger["sessions"][session_id] = spent
    write(meter, ledger)
    if read(lease, {}).get("id") == session_id:
        lease.unlink()
    status["estimated_ccu_this_session"] = spent
    status["estimated_ccu_total"] = sum(ledger["sessions"].values())
    write(receipt, status)
    print(out, status, flush=True)
    runtime.unassign()
