"""Pack a tiny, self-contained tarball (`dist/pulse_h100_phase2.tar.gz`) for 1-command H100 deployment."""
from __future__ import annotations

import hashlib
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build_bundle() -> Path:
    files = sorted((ROOT / "src/pulse").glob("*.py")) + [
        ROOT / "benchmarks/v7_stage0_stage1.py",
        ROOT / "benchmarks/fomp_v3_open_edit.py",
        ROOT / "benchmarks/pulse_v81_scaling_law.py",
        ROOT / "tests/test_fomp_v3.py",
        ROOT / "tests/test_pulse_v8.py",
        ROOT / "tests/test_pulse_v81_scaling_law.py",
        ROOT / "scripts/run_h100_scaling_law.sh",
        ROOT / "scripts/H100_QUICKSTART.md",
    ]
    out_dir = ROOT / "dist"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_tar = out_dir / "pulse_h100_phase2.tar.gz"
    with tarfile.open(out_tar, "w:gz") as tf:
        for p in files:
            arcname = p.relative_to(ROOT).as_posix()
            data = p.read_bytes()
            # Enforce LF line endings on Linux shell scripts!
            if p.suffix == ".sh":
                data = data.replace(b"\r\n", b"\n")
            info = tarfile.TarInfo(name=arcname)
            info.size = len(data)
            info.mode = 0o755 if p.suffix == ".sh" else 0o644
            import io
            tf.addfile(info, io.BytesIO(data))
    sha = hashlib.sha256(out_tar.read_bytes()).hexdigest()
    print(f"Packed {out_tar} ({out_tar.stat().st_size:,} bytes, {len(files)} files, sha256={sha[:12]}...)")
    return out_tar


if __name__ == "__main__":
    build_bundle()
