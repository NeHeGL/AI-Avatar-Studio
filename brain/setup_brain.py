"""Download the chat brain: the llama.cpp server (CUDA) and a Gemma 3 12B model, into this folder.

Run once:  "Setup Brain.bat"   (about 8 GB). Safe to run again: finished files are skipped.
Use another model: put any chat-tuned .gguf file in brain/models/ (the first one found is used).
"""
from __future__ import annotations

import os
import sys
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
BIN, MODELS = HERE / "bin", HERE / "models"
LLAMA = "b12589"
GH = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA}/"
ZIPS = [f"llama-{LLAMA}-bin-win-cuda-13.5-x64.zip", "cudart-llama-bin-win-cuda-13.5-x64.zip"]
sys.path.insert(0, str(HERE.parent))
from app import hardware  # noqa: E402  (picks the chat model that fits this graphics card)

MODEL_URL = hardware.tier()["brain_url"]


def fetch(url: str, dest: Path, cb=None) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"Downloading {dest.name} ...")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total, got, last = int(r.headers.get("Content-Length") or 0), 0, -1
        while chunk := r.read(1 << 20):
            f.write(chunk)
            got += len(chunk)
            if total and got * 100 // total != last:
                last = got * 100 // total
                print(f"\r  {last}%", end="", flush=True)
                if cb:
                    cb("download", int(last))
    print()
    tmp.replace(dest)


def main(cb=None) -> int:
    """cb(state, pct) is told how it is going: 'download' with a percentage, then 'setup'."""
    BIN.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    if not (BIN / "llama-server.exe").exists():
        for name in ZIPS:
            z = HERE / name
            fetch(GH + name, z, cb)
            if cb:
                cb("setup", 0)
            with zipfile.ZipFile(z) as zf:
                zf.extractall(BIN)
            z.unlink()
    else:
        print("llama-server already installed.")
    target = MODELS / hardware.tier()["brain_file"]  # saved under this card's own model name
    need = not target.exists()
    if need:
        fetch(MODEL_URL, target, cb)
    else:
        print("A model is already in brain/models.")
    print(f"\nChat brain ready ({hardware.tier()['brain_size'].upper()} model for a {hardware.tier()['vram_gb']} GB card). Start Avatar Studio as usual.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
