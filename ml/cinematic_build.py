"""Build the pre-rendered full-quality clip library for one person. Everything runs locally.

    ml\\.venv\\Scripts\\python.exe ml\\cinematic_build.py <user_slug> [--voice-ref wav] [--only id,id]

For each line in ml/cinematic_lines.json:
  1. speak it in the person's cloned voice (Chatterbox)          -> <cinematic>/src/<id>.wav
  2. render a talking video from the source photo (Ditto)         -> <cinematic>/<id>.mp4
  3. write <cinematic>/manifest.json so the app can find the clips

The studio (Ditto render service) must be running for step 2; step 1 loads the voice model in this process.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ML = ROOT / "ml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("user")
    ap.add_argument("--voice-ref", default=str(ML / "out" / "_ref_jeff_molofee.wav"))
    ap.add_argument("--only", default="")
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

    # ---- 2. videos (the local Ditto render service must be running: start the studio first)
    import urllib.parse
    import urllib.request

    manifest = []
    for l in lines:
        out = cdir / f"{l['id']}.mp4"
        wav = src / f"{l['id']}.wav"
        if not out.exists():
            t = time.time()
            url = "http://127.0.0.1:8771/render?" + urllib.parse.urlencode({"source": str(photo), "out": str(out)})
            try:
                with urllib.request.urlopen(urllib.request.Request(url, data=wav.read_bytes(), method="POST", headers={"Content-Type": "audio/wav"}), timeout=600) as r:
                    ok = json.loads(r.read()).get("ok")
            except Exception as e:
                sys.exit(f"Ditto render service not reachable ({e}). Start the studio first, then re-run (finished clips are kept).")
            if not ok or not out.exists():
                print("render failed for", l["id"], flush=True)
                continue
            print(f"video {l['id']}: rendered in {time.time() - t:.1f}s", flush=True)
        import soundfile as sf

        dur = sf.info(str(wav)).duration
        manifest.append({"id": l["id"], "kind": l["kind"], "text": l["text"], "video": f"{l['id']}.mp4", "seconds": round(dur, 2)})
        json.dump({"user": a.user, "clips": manifest}, open(cdir / "manifest.json", "w", encoding="utf-8"), indent=2)
    print("done:", len(manifest), "clips ->", cdir / "manifest.json", flush=True)


if __name__ == "__main__":
    main()
