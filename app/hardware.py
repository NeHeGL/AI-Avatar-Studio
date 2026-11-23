"""Pick the models that fit this PC's graphics card.

Measured GPU memory (Windows, 4k context): chat brain Gemma 3 12B Q4 = 8.4 GB, Gemma 3 4B Q4 = about 3.3 GB; video animation 6.8 GB;
voice (Qwen3-TTS 1.7B) about 4.6 GB at its peak; ears Whisper large-v3 turbo 2.1 GB, Whisper small about 0.6 GB (on the CPU it takes about 3 s per 8 s of speech). Everything has to share the card with the desktop.

To test a smaller card for real, on a bigger one: reserve the difference, then pretend. For a 16 GB card on a 32 GB one:
  python -m app.hardware --reserve 16223        (holds that many MiB of graphics memory; leave it running)
  set AVATAR_VRAM_GB=16 and AVATAR_VRAM_RESERVED_MIB=16223, then start the app as usual.
The app then picks the 16 GB models, has only 16 GB left to live in (together with Windows), and the footer shows the card as 16 GB with the
reserve taken off the usage.
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


def reserved_gb() -> float:
    """Graphics memory held back on purpose to imitate a smaller card (AVATAR_VRAM_RESERVED_MIB, see above); 0 normally."""
    try:
        return max(0.0, float(os.environ.get("AVATAR_VRAM_RESERVED_MIB", "0") or 0)) / 1024
    except ValueError:
        return 0.0


@lru_cache(maxsize=1)
def tier() -> dict:
    gb = vram_gb()
    # a card reports a little under its nominal size (a "32 GB" card shows about 31.5), so allow some slack
    for minimum, name, size, whisper, whisper_name, whisper_device in TIERS:
        if gb >= minimum - 1.5:
            return {
                "name": name, "vram_gb": round(gb, 1), "brain_size": size, "whisper": whisper, "whisper_name": whisper_name, "whisper_device": whisper_device,
                # Gemma 3 with its refusal behaviour removed ("abliterated", mlabonne v2): the same size and speed as the stock model, but it
                # does not lecture or refuse in casual private chat. The stock ggml-org build refuses a lot even with a relaxed prompt.
                "brain_file": f"gemma-3-{size}-it-abliterated-v2-Q4_K_M.gguf",
                "brain_url": f"https://huggingface.co/mlabonne/gemma-3-{size}-it-abliterated-v2-GGUF/resolve/main/gemma-3-{size}-it-abliterated-v2.q4_k_m.gguf",
            }
    raise RuntimeError("no tier")


if __name__ == "__main__":
    import sys

    if "--reserve" in sys.argv:  # hold graphics memory (MiB) until this program is stopped; see the top of this file
        import time

        import torch

        mib = int(sys.argv[sys.argv.index("--reserve") + 1])
        held = []
        left = max(0, mib - 400)  # the program's own CUDA context takes about 400 MiB, so the total held is what was asked for
        while left > 0:
            n = min(512, left)
            held.append(torch.empty(n * 1024 * 1024, dtype=torch.uint8, device="cuda"))
            left -= n
        print(f"holding about {mib} MiB of graphics memory", flush=True)
        while True:
            time.sleep(5)
    print(tier())
