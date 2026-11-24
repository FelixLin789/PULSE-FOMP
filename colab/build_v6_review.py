"""Package the current PULSE code into one self-contained Colab notebook."""

import base64
import hashlib
import io
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def cell(kind, source):
    item = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        item.update(execution_count=None, outputs=[])
    return item


def notebook():
    paths = sorted((ROOT / "src/pulse").glob("*.py"))
    paths += [ROOT / "benchmarks" / name for name in (
        "review_v53.py", "review_scale.py", "review_v6.py", "review_v6_scale.py", "review_v6_pre105.py"
    )]
    paths += [ROOT / "tests/test_pulse_v6.py", ROOT / "README.md",
              ROOT / "experiment_outputs/reviewer_v6/PULSE_v6_REVIEW.md",
              ROOT / "colab/build_v6_review.py", ROOT / "colab/review_v6_cell.py"]
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            info = zipfile.ZipInfo(path.relative_to(ROOT).as_posix())
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
    payload = data.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    setup = f"""
import base64
import hashlib
import io
import tempfile
import zipfile
from pathlib import Path

source = base64.b64decode({base64.b64encode(payload).decode()!r})
code_sha256 = {digest!r}
assert hashlib.sha256(source).hexdigest() == code_sha256
work = Path(tempfile.mkdtemp(prefix="pulse-v6-", dir="/content"))
with zipfile.ZipFile(io.BytesIO(source)) as archive:
    archive.extractall(work)
print("PULSE v6", code_sha256, flush=True)
"""
    intro = """# PULSE v6 · 105M 前的 L4 审计

选择 NVIDIA L4 后运行全部单元。授权挂载 Drive 后，结果和检查点会写入
`Colab Notebooks/PULSE_AI/runs/review_v6_l4_v1`。

1. 复测 RLS，补齐 FineWeb-Edu，实测相同墙钟预算，并运行 PyTorch Profiler。
2. 扫描滑窗宽度与检索距离。双方短程通过门槛后测 256 / 1K / 4K / 8K。
3. 结果写入 pre105。完成后暂停，105M 训练需另行确认。

每次会话结束会释放 L4。再次运行本笔记本可读取已写入的检查点并继续。
"""
    runtime = (ROOT / "colab/review_v6_cell.py").read_text(encoding="utf-8")
    cells = [cell("markdown", intro), cell("code", setup), cell("code", runtime)]
    for item in cells:
        if item["cell_type"] == "code":
            compile("".join(item["source"]), "<PULSE_v6_Colab_L4>", "exec")
    return {"nbformat": 4, "nbformat_minor": 4, "cells": cells,
            "metadata": {"accelerator": "GPU", "colab": {"gpuType": "L4"},
                         "kernelspec": {"name": "python3", "display_name": "Python 3"}}}


if __name__ == "__main__":
    path = ROOT / "colab/PULSE_v6_Pre105_L4.ipynb"
    path.write_text(json.dumps(notebook(), ensure_ascii=False, indent=1), encoding="utf-8")
    print(path)
