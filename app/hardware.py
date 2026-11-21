"""Pick the models that fit this PC's graphics card.

Measured GPU memory (Windows, 4k context): chat brain Gemma 3 12B Q4 = 8.4 GB, Gemma 3 4B Q4 = about 3.3 GB; video animation 6.8 GB;
voice 3.2 GB; ears Whisper large-v3 turbo 2.1 GB, Whisper small about 0.6 GB (on the CPU it takes about 3 s per 8 s of speech). Everything has to share the card with the desktop.

AVATAR_VRAM_GB=16 pretends the card has that much (for testing a smaller tier).
"""
from __future__ import annotations

import os
import subprocess
from functools import lru_cache

# (minimum total VRAM in GB, tier name, chat model size, Whisper model, Whisper display name, Whisper device)
# The GPU is used whenever the memory allows. A card too small for everything gets a smaller chat model and Whisper on the CPU.
TIERS = [
    (28, "large", "12b", "openai/whisper-large-v3-turbo", "Whisper large-v3 turbo", "cuda"),
    (20, "medium", "12b", "openai/whisper-small", "Whisper small", "cuda"),
    (0, "small", "4b", "Systran/faster-distil-whisper-small.en", "Whisper distil-small.en (CPU)", "cpu"),  # English-only CTranslate2 build: the fastest accurate choice on a CPU
]


@lru_cache(maxsize=1)
def vram_gb() -> float:
    """Total memory of the first NVIDIA GPU in GB (0 when there is none)."""
    forced = os.environ.get("AVATAR_VRAM_GB", "").strip()
    if forced:
        try:
            return float(forced)
        except ValueError:
            pass
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
        return max(float(line) for line in out.split() if line.strip().replace(".", "").isdigit()) / 1024
    except Exception:
        return 0.0


@lru_cache(maxsize=1)
def tier() -> dict:
    gb = vram_gb()
    # a card reports a little under its nominal size (a "32 GB" card shows about 31.5), so allow some slack
    for minimum, name, size, whisper, whisper_name, whisper_device in TIERS:
        if gb >= minimum - 1.5:
            return {
                "name": name, "vram_gb": round(gb, 1), "brain_size": size, "whisper": whisper, "whisper_name": whisper_name, "whisper_device": whisper_device,
                "brain_file": f"gemma-3-{size}-it-Q4_K_M.gguf",
                "brain_url": f"https://huggingface.co/ggml-org/gemma-3-{size}-it-GGUF/resolve/main/gemma-3-{size}-it-Q4_K_M.gguf",
            }
    raise RuntimeError("no tier")


if __name__ == "__main__":
    print(tier())
