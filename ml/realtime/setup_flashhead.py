"""Install the talking-video renderer (SoulX-FlashHead-Lite) into ml/realtime/flashhead: its code and models.
Its Python libraries are part of the app's one environment (requirements.txt).

Run by installer.py. Safe to run again: finished steps are skipped. Needs internet; about 8 GB.
"""
from __future__ import annotations

import io
import shutil
import subprocess
import sys
import threading
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


# What the model downloads add up to when finished (measured: 8.54 GB), to show a percentage.
EXPECTED_BYTES = 8_540_000_000


def _bytes_on_disk(models: Path) -> int:
    """Everything downloaded so far, including files still being written."""
    total = 0
    for p in models.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            pass  # a file that was just renamed into place
    return total


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

        # Progress is measured from the bytes on disk (the downloader's own progress bars are not reliable: they stayed at 0% for the whole
        # download). Files being downloaded count as they grow, and a resumed download starts from where it stopped.
        stop = threading.Event()

        def watch() -> None:
            while not stop.wait(1.5):
                say("download", min(99, int(100 * _bytes_on_disk(models) / EXPECTED_BYTES)))

        say("download", 0)
        threading.Thread(target=watch, daemon=True).start()
        try:
            snapshot_download("Soul-AILab/SoulX-FlashHead-1_3B", local_dir=str(models / "SoulX-FlashHead-1_3B"),
                              allow_patterns=["Model_Lite/*", "VAE_LTX/*", "*.json", "*.md", "*.txt"])
            snapshot_download("facebook/wav2vec2-base-960h", local_dir=str(models / "wav2vec2-base-960h"),
                              allow_patterns=["*.json", "*.txt", "*.safetensors", "*.bin"])
        finally:
            stop.set()
    print("Video renderer ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else sys.executable))
