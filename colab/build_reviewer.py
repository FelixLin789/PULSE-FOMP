"""Build self-contained Colab audit notebooks from the reviewed scripts."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def code(text):
    return {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],
            "source":text.strip().splitlines(keepends=True)}


SETUP = '''
from pathlib import Path
import hashlib, os, subprocess, sys, tempfile, time, zipfile
import torch
from google.colab import drive, runtime

if not torch.cuda.is_available() or "L4" not in torch.cuda.get_device_name():
    raise RuntimeError("请选择 NVIDIA L4 再运行。")
drive.mount("/content/drive")
project = Path("/content/drive/MyDrive/Colab Notebooks/PULSE_AI")
source = project / "pulse_ai_src.zip"
with source.open("rb") as f:
    sha = hashlib.file_digest(f,"sha256").hexdigest()
if sha != "0c4258fa39670f2c0df7ce48ddcae3039b33068358f4d88fddb41f7ea7e3c6c9":
    raise RuntimeError("Drive 源码已变化。请核对版本后重新运行。")
work = Path(tempfile.mkdtemp(prefix="pulse-review-",dir="/content"))
with zipfile.ZipFile(source) as z:
    for member in z.infolist():
        target=(work/member.filename).resolve()
        if not target.is_relative_to(work):
            raise ValueError(member.filename)
    z.extractall(work)
env = dict(os.environ, PYTHONPATH=str(work/"src")+os.pathsep+str(work))
print("Drive 源码 SHA-256",sha,"GPU",torch.cuda.get_device_name(),"PyTorch",torch.__version__,flush=True)
'''


def main():
    cli=argparse.ArgumentParser()
    cli.add_argument("stage",choices=["checkpoints","scale"])
    stage=cli.parse_args().stage
    audit_code=(ROOT/"benchmarks/review_v53.py").read_text(encoding="utf-8")
    title=("PULSE-v5.3 检查点消融与 4K / 8K 检索" if stage=="checkpoints"
           else "PULSE-v5.3 与全注意力 Transformer 的 FineWeb-Edu 配对训练")
    desc=("从训练到 2,000 步的两份原始权重评测。三个干预共用同一份 PULSE 权重。"
          "输出保留逐题命中、逐窗口损失和配对区间。专用运行时完成或超时后释放 L4。"
          if stage=="checkpoints" else "使用相同真实语料顺序训练两种模型。运行参数、吞吐量测算及成对检查点写入项目目录。")
    cells=[{"cell_type":"markdown","metadata":{},"source":[f"# {title}\n\n",desc]},
           code(SETUP),code("(work/'review_v53.py').write_text("+repr(audit_code)+",encoding='utf-8')")]
    if stage=="checkpoints":
        cells.append(code("""
import json, importlib.util
out = project/"runs"/"review_v53_checkpoints_v1"
out.mkdir(exist_ok=True,parents=True)
receipt={"stage":"starting","gpu":torch.cuda.get_device_name(),"source_sha256":sha,"started_unix":time.time()}
def record():
    p=out/"status.json"
    tmp=p.with_suffix(".tmp")
    tmp.write_text(json.dumps(receipt,indent=2))
    os.replace(tmp,p)
record()
try:
    subprocess.run([sys.executable,"-m","pip","-q","install","tiktoken","einops"],check=True,timeout=240)
    if importlib.util.find_spec("fla") is None:
        subprocess.run([sys.executable,"-m","pip","-q","install","--no-deps","fla-core"],timeout=240)
    old = project/"runs"/"exp_pulse_v53_micro2k"
    receipt["stage"]="running"
    record()
    subprocess.run([sys.executable,"-u",str(work/"review_v53.py"),"--run",str(old),"--out",str(out)],
                   env=env,check=True,timeout=4500)
    receipt["stage"]="complete"
except BaseException as exc:
    receipt.update(stage="stopped",error=str(exc))
    raise
finally:
    receipt["ended_unix"]=time.time()
    record()
    print(out,receipt,flush=True)
    runtime.unassign()
"""))
    else:
        scale=(ROOT/"benchmarks/review_scale.py").read_text(encoding="utf-8")
        cells.append(code("(work/'review_scale.py').write_text("+repr(scale)+",encoding='utf-8')"))
        cells.append(code((ROOT/"colab/review_scale_cell.py").read_text(encoding="utf-8")))
    data={"nbformat":4,"nbformat_minor":4,"metadata":{"accelerator":"GPU","colab":{"gpuType":"L4"},
          "kernelspec":{"name":"python3","display_name":"Python 3"}},"cells":cells}
    path=ROOT/"colab"/("PULSE_v53_"+stage+".ipynb")
    path.write_text(json.dumps(data,ensure_ascii=False,indent=1),encoding="utf-8")
    for c in cells:
        if c["cell_type"]=="code":
            compile("".join(c["source"]),str(path),"exec")
    print(path)


if __name__=="__main__":
    main()
