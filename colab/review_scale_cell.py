"""This file is embedded as the last cell of the dedicated FineWeb-Edu notebook."""
import importlib.util
import json
import uuid

#@title L4 预算与断点续训
# 首次进行短校准。费率默认沿用旧实验的估算，面板显示其他值时可直接替换。
PANEL_CCU_PER_HOUR = 4.8 #@param {type:"number"}
CALIBRATION_ONLY = True #@param {type:"boolean"}
SCALE_BUDGET_CCU = 145.0 #@param {type:"number"}
MAX_SESSION_HOURS = 8.0 #@param {type:"number"}
if not 0 < SCALE_BUDGET_CCU <= 145 or not 0 < MAX_SESSION_HOURS <= 10 or PANEL_CCU_PER_HOUR <= 0:
    raise ValueError("请在本次 160 CCU 总范围内设置参数。规模训练最多使用 145 CCU。")

out=project/"runs"/"review_v53_fineweb_1b_v1"
out.mkdir(exist_ok=True,parents=True)
meter=out/"budget.json"
lease=out/"active_session.json"
pilot=CALIBRATION_ONLY
rate=PANEL_CCU_PER_HOUR
started=time.time()
def atomic(path, data):
    tmp=path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
    tmp.write_text(json.dumps(data,indent=2))
    os.replace(tmp,path)
if pilot and (out/"calibration.json").exists():
    print((out/"calibration.json").read_text(),flush=True)
    runtime.unassign()
    raise SystemExit("校准已完成。取消 CALIBRATION_ONLY 可继续已保存的训练。")

# Reservations are retained at face value if Colab forcibly disconnects before settlement.
# Only a single dedicated runtime may claim each budget. A second launch does not train.
if lease.exists():
    old=json.loads(lease.read_text())
    if time.time()<old["expires"]+60:
        raise RuntimeError("已有训练会话持有这次预算，请保留正在运行的会话。")
    old_ledger=json.loads(meter.read_text()) if meter.exists() else {"sessions":{}}
    old_ledger["sessions"].setdefault(old["id"],old["reserved_ccu"])
    atomic(meter,old_ledger)
    lease.unlink()
ledger=json.loads(meter.read_text()) if meter.exists() else {"sessions":{}}
available=SCALE_BUDGET_CCU-sum(ledger["sessions"].values())
hours=min(.60 if pilot else MAX_SESSION_HOURS,available/rate)
if hours < .10:
    runtime.unassign()
    raise SystemExit("剩余预算不足以启动并保存一轮配对训练。")
end=started+hours*3600
sid=uuid.uuid4().hex
entry={"id":sid,"expires":end,"rate":rate,"reserved_ccu":hours*rate,"pilot":pilot}
with lease.open("x") as f: json.dump(entry,f)
ledger["sessions"][sid]=entry["reserved_ccu"]
atomic(meter,ledger)
try:
    subprocess.run([sys.executable,"-m","pip","-q","install","tiktoken","huggingface_hub",
                    "pyarrow","einops"],check=True,timeout=min(300,max(1,end-time.time())))
    if importlib.util.find_spec("fla") is None:
        subprocess.run([sys.executable,"-m","pip","-q","install","--no-deps","fla-core"],
                       timeout=min(240,max(1,end-time.time())))
    args=[sys.executable,"-u",str(work/"review_scale.py"),"--out",str(out),"--deadline",str(end-30),
          "--started",str(started),"--rate",str(rate),"--available",str(available)]
    if pilot: args.append("--pilot")
    subprocess.run(args,check=True,env=env,timeout=max(1,end-time.time()))
finally:
    spent=max(0,time.time()-started)*rate/3600
    ledger=json.loads(meter.read_text())
    ledger["sessions"][sid]=spent
    atomic(meter,ledger)
    if lease.exists() and json.loads(lease.read_text())["id"]==sid:
        lease.unlink()
    print("本阶段估算用量",round(spent,3),"CCU；累计",round(sum(ledger["sessions"].values()),3),
          "CCU。输出",out,flush=True)
    runtime.unassign()
