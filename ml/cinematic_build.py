"""Build the pre-rendered full-quality clip library for one person. Everything runs locally.

    ml\\.venv\\Scripts\\python.exe ml\\cinematic_build.py <user_slug> [--voice-ref wav] [--only id,id] [--steps 4] [--size 512]

For each line in ml/cinematic_lines.json:
  1. speak it in the person's cloned voice (Chatterbox)          -> <cinematic>/src/<id>.wav
  2. render a talking video from the source photo (InfiniteTalk)  -> <cinematic>/<id>.mp4
  3. write <cinematic>/manifest.json so the app can find the clips

Safety: the GPU is only 32 GB. The language model and the chat app's voice models also live there, so this
script refuses to start a render when GPU memory is already high and aborts if it climbs too far.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ML = ROOT / "ml"
PY = ML / ".venv" / "Scripts" / "python.exe"


def gpu_mb() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    return int(out.strip().splitlines()[0])


def unload_llm():
    """Free the chat language model from the GPU (Ollama keeps it loaded for 30 minutes)."""
    import urllib.request

    try:
        req = urllib.request.Request("http://127.0.0.1:11434/api/ps")
        for m in json.loads(urllib.request.urlopen(req, timeout=5).read()).get("models", []):
            body = json.dumps({"model": m["name"], "keep_alive": 0}).encode()
            urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:11434/api/generate", data=body, method="POST"), timeout=30).read()
            print("unloaded language model", m["name"], flush=True)
    except Exception as e:  # Ollama not running is fine
        print("(no language model to unload)", e, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("user")
    ap.add_argument("--voice-ref", default=str(ML / "out" / "_ref_jeff_molofee.wav"))
    ap.add_argument("--only", default="")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--prompt", default="a man speaking calmly to the camera, head almost still, natural blinking, photorealistic")
    a = ap.parse_args()

    cdir = ROOT / "data" / "media" / a.user / "cinematic"
    src = cdir / "src"
    src.mkdir(parents=True, exist_ok=True)
    photo = cdir / "source.jpg"
    if not photo.exists():
        sys.exit(f"missing source photo {photo}")
    lines = json.load(open(ML / "cinematic_lines.json", encoding="utf-8"))["lines"]
    if a.only:
        want = set(a.only.split(","))
        lines = [l for l in lines if l["id"] in want]

    # ---- 1. voices (one model load for all lines)
    todo = [l for l in lines if not (src / f"{l['id']}.wav").exists()]
    if todo:
        import numpy as np
        import soundfile as sf
        import torch

        data, sr = sf.read(a.voice_ref, always_2d=True)
        ref_wav = src / "_voice_ref.wav"
        sf.write(ref_wav, data.mean(axis=1).astype(np.float32), sr)
        from chatterbox.tts import ChatterboxTTS

        tts = ChatterboxTTS.from_pretrained(device="cuda")
        for l in todo:
            t = time.time()
            wav = tts.generate(l["text"], audio_prompt_path=str(ref_wav), exaggeration=0.5, cfg_weight=0.5)
            sf.write(src / f"{l['id']}.wav", wav.squeeze().cpu().numpy(), tts.sr)
            print(f"voice {l['id']}: {wav.shape[-1] / tts.sr:.1f}s of speech in {time.time() - t:.1f}s", flush=True)
        del tts
        torch.cuda.empty_cache()

    # ---- 2. videos
    unload_llm()
    manifest = []
    for l in lines:
        out = cdir / f"{l['id']}.mp4"
        wav = src / f"{l['id']}.wav"
        if not out.exists():
            used = gpu_mb()
            if used > 14000:
                sys.exit(f"GPU already at {used} MiB; stop other GPU jobs / the chat app first, then re-run (finished clips are kept).")
            t = time.time()
            proc = subprocess.Popen([str(PY), str(ML / "comfy_run.py"), "--image", str(photo), "--audio", str(wav), "--out", str(out),
                                     "--size", str(a.size), "--steps", str(a.steps), "--prompt", a.prompt], cwd=str(ML))
            while proc.poll() is None:
                time.sleep(5)
                if gpu_mb() > 30500:
                    proc.kill()
                    sys.exit("GPU memory nearly full - aborted to protect the driver. Free GPU memory and re-run.")
            if proc.returncode != 0 or not out.exists():
                print("render failed for", l["id"], flush=True)
                continue
            print(f"video {l['id']}: rendered in {time.time() - t:.0f}s", flush=True)
        import soundfile as sf

        dur = sf.info(str(wav)).duration
        manifest.append({"id": l["id"], "kind": l["kind"], "text": l["text"], "video": f"{l['id']}.mp4", "seconds": round(dur, 2)})
        json.dump({"user": a.user, "clips": manifest}, open(cdir / "manifest.json", "w", encoding="utf-8"), indent=2)
    print("done:", len(manifest), "clips ->", cdir / "manifest.json", flush=True)


if __name__ == "__main__":
    main()
