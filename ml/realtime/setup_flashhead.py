"""Install the talking-video renderer (SoulX-FlashHead-Lite) into ml/realtime/flashhead: its code and models.
Its Python libraries are part of the app's one environment (requirements.txt).

Run by installer.py. Safe to run again: finished steps are skipped. Needs internet; about 8 GB.
"""
from __future__ import annotations

import io
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
FH = HERE / "flashhead"
ZIP = "https://github.com/Soul-AILab/SoulX-FlashHead/archive/refs/heads/main.zip"

# Everything the renderer opens. The install counts as finished only when ALL of these are there, so a download that stopped half way is
# resumed (and the renderer is not started before they exist: it would only crash). The Lite model needs the LTX VAE; the Wan one belongs to the Pro model.
REQUIRED = [
    "flash_head",
    "models/SoulX-FlashHead-1_3B/Model_Lite/diffusion_pytorch_model.safetensors",
    "models/SoulX-FlashHead-1_3B/VAE_LTX/diffusion_pytorch_model.safetensors",
    "models/wav2vec2-base-960h/model.safetensors",
    "models/wav2vec2-base-960h/pytorch_model.bin",
]


def complete() -> bool:
    return all((FH / r).exists() for r in REQUIRED)


def run(*args: str, check: bool = True) -> int:
    print("  >", " ".join(args[:6]), "...", flush=True)
    r = subprocess.run(list(args))
    if check and r.returncode:
        raise SystemExit(f"Step failed: {' '.join(args[:5])}")
    return r.returncode


def main(base_python: str, cb=None) -> int:
    """cb(state, pct) is told how it is going: 'download' (with a percentage for the models) or 'setup'."""
    say = cb or (lambda *a: None)
    if not (FH / "flash_head").exists():
        say("download", 0)
        print("Downloading the renderer code...")
        with urllib.request.urlopen(ZIP) as r:
            data = r.read()
        tmp = HERE / "_fh_unpack"
        shutil.rmtree(tmp, ignore_errors=True)
        zipfile.ZipFile(io.BytesIO(data)).extractall(tmp)
        src = next(tmp.iterdir())
        FH.mkdir(exist_ok=True)
        for item in src.iterdir():
            dest = FH / item.name
            if not dest.exists():
                shutil.move(str(item), str(dest))
        shutil.rmtree(tmp, ignore_errors=True)
    say("setup", 0)
    models = FH / "models"
    if not complete():
        from huggingface_hub import snapshot_download
        from tqdm.auto import tqdm

        bars: dict[int, tuple[float, float]] = {}

        class _Bar(tqdm):
            def __init__(self, *a, **k):
                k.pop("name", None)
                super().__init__(*a, **k)

            def update(self, n=1):
                r = super().update(n)
                if self.total and self.total > 1_000_000:
                    bars[id(self)] = (self.n, self.total)
                    done, total = (sum(v[i] for v in bars.values()) for i in (0, 1))
                    say("download", min(99, int(100 * done / max(total, 1))))
                return r

        say("download", 0)
        snapshot_download("Soul-AILab/SoulX-FlashHead-1_3B", local_dir=str(models / "SoulX-FlashHead-1_3B"),
                          allow_patterns=["Model_Lite/*", "VAE_LTX/*", "*.json", "*.md", "*.txt"], tqdm_class=_Bar)
        snapshot_download("facebook/wav2vec2-base-960h", local_dir=str(models / "wav2vec2-base-960h"),
                          allow_patterns=["*.json", "*.txt", "*.safetensors", "*.bin"], tqdm_class=_Bar)
    print("Video renderer ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else sys.executable))
