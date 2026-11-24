"""Local render service: speech WAV in -> talking-head MP4 out, made by SoulX-FlashHead-Lite (4 steps). Nothing leaves this machine.

    ml\\realtime\\flashhead\\.venv\\Scripts\\python.exe ml\\realtime\\flashhead_render_server.py [--port 8771]

    POST /render?source=<photo path>&out=<mp4 path>   body = WAV bytes   ->  {"ok": true, "path": "...", "seconds": 4.2}
    GET  /health

Same interface the app already used for Ditto. A reply is FlashHead's own 512x512 render of the head-and-shoulders crop of the
person's photo (background included, nothing cut out or pasted), used as it is.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
FH = HERE / "flashhead"
FFMPEG_BIN = HERE / "ffmpeg" / "ffmpeg-master-latest-win64-gpl" / "bin"
os.environ["PATH"] = str(FFMPEG_BIN) + os.pathsep + os.environ.get("PATH", "")
os.environ.setdefault("PYTHONUTF8", "1")
sys.path.insert(0, str(HERE / "flashhead_stubs"))  # single-GPU stand-in for xfuser (which does not import on Windows)
sys.path.insert(0, str(FH))
os.chdir(FH)  # FlashHead reads its config relative to its own folder

import cv2
import librosa
import numpy as np
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from PIL import Image

from flash_head.inference import get_audio_embedding, get_base_data, get_infer_params, get_pipeline, run_pipeline
from flash_head.utils.cpu_face_handler import CPUFaceHandler

CKPT = str(FH / "models" / "SoulX-FlashHead-1_3B")
W2V = str(FH / "models" / "wav2vec2-base-960h")
SEED = 42
LEAN = os.environ.get("AVATAR_LEAN") == "1"  # set by start_all on the small (16 GB) tier only
TAIL_S = 0.6        # silence added after the speech, so the head settles before the clip ends
EASE_FRAMES = 16    # the last 0.64 s blend onto the rest frame (a shorter blend reads as a snap from the talking pose to the waiting animation)
EASE_IN_FRAMES = 14  # the first 0.56 s blend from the rest frame into the talking
TMP = ROOT / "data" / "tmp_render"

print("loading FlashHead-Lite...", flush=True)
_t = time.time()
PIPE = get_pipeline(world_size=1, ckpt_dir=CKPT, model_type="lite", wav2vec_dir=W2V)
print(f"model loaded in {time.time() - _t:.1f}s", flush=True)
LOCK = threading.Lock()
FACE = CPUFaceHandler()
app = FastAPI()
_PREP: dict = {}


# ---------------------------------------------------------------- preparing a person
def prepare(source: Path) -> dict:
    """One-time preparation of a person's photo: FlashHead's head-and-shoulders crop (a square 2x the face width, the face 55% of the
    way down). Saved next to the photo (talk_crop.png, talk_meta.json) and only redone when the photo changes. The page uses the
    crop box in talk_meta.json to zoom its idle pictures to the same view, so idle and talking line up."""
    d = source.parent
    stamp = [source.stat().st_mtime, "crop-v2-rest"]
    hit = _PREP.get(str(source))
    if hit and hit["stamp"] == stamp:
        return hit
    meta_f = d / "talk_meta.json"
    meta = json.loads(meta_f.read_text(encoding="utf-8")) if meta_f.exists() else {}
    if meta.get("stamp") != stamp or not (d / "talk_crop.png").exists() or not (d / "talk_rest.jpg").exists():
        photo = cv2.imread(str(source))
        H, W = photo.shape[:2]
        rgb = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)
        try:  # a picture made by the Photo step is already the head-and-shoulders view: use it as it is, no further zoom
            prepared = bool(json.loads((d / "source_info.json").read_text(encoding="utf-8")).get("prepared"))
        except Exception:
            prepared = False
        boxes, _ = ([[0.0, 0.0, 1.0, 1.0]], None) if prepared else FACE(rgb)
        try:  # a drawing or cartoon the detector cannot read: the person can say where the face is (face/options.json -> "face_box")
            manual = json.loads((d / "options.json").read_text(encoding="utf-8")).get("face_box")
        except Exception:
            manual = None
        if manual and len(boxes) == 0:
            boxes = [manual]  # [x1, y1, x2, y2] as fractions of the picture
        if len(boxes) == 0:
            raise RuntimeError("no face found in the photo")
        x1, y1, x2, y2 = boxes[0][0] * W, boxes[0][1] * H, boxes[0][2] * W, boxes[0][3] * H
        cx, cy, side = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * 2.0
        box = [int(max(0, cx - side * 0.5)), int(max(0, cy - side * 0.55)), int(min(W, cx + side * 0.5)), int(min(H, cy + side * 0.45))]
        if prepared:
            box = [0, 0, W, H]
        Image.fromarray(cv2.resize(rgb[box[1]:box[3], box[0]:box[2]], (512, 512), interpolation=cv2.INTER_AREA)).save(d / "talk_crop.png")
        # the REST frame: how FlashHead itself draws the still photo (frame 0 of a render of silence). Every clip starts from it and
        # lands back on it, and the page shows it as the still picture, so one clip follows the next without a visible join
        silence = TMP / "fh_silence.wav"
        TMP.mkdir(parents=True, exist_ok=True)
        import soundfile as sf

        sf.write(silence, np.zeros(16000, dtype=np.float32), 16000)
        Image.fromarray(_frames_for(str(silence), d / "talk_crop.png")[0]).save(d / "talk_rest.jpg", quality=95)
        meta = {"stamp": stamp, "box": box, "size": [W, H]}
        meta_f.write_text(json.dumps(meta), encoding="utf-8")
        print(f"prepared {d.parent.parent.name}: crop box {box} of {W}x{H}", flush=True)
    hit = dict(stamp=stamp, crop_png=d / "talk_crop.png")
    _PREP[str(source)] = hit
    return hit


# ---------------------------------------------------------------- rendering
def _frames_for(wav_path: str, crop_png: Path, tail_s: float = 0.0, seed: int = SEED) -> list[np.ndarray]:
    """FlashHead's streaming loop for one utterance -> list of 512x512 RGB uint8 frames."""
    get_base_data(PIPE, cond_image_path_or_dir=str(crop_png), base_seed=seed, use_face_crop=False)
    p = get_infer_params()
    sr, fps, frame_num = p["sample_rate"], p["tgt_fps"], p["frame_num"]
    motion = p["motion_frames_num"]
    slice_len = frame_num - motion
    speech, _ = librosa.load(wav_path, sr=sr, mono=True)
    if tail_s:
        speech = np.concatenate([speech, np.zeros(int(tail_s * sr), dtype=speech.dtype)])  # a moment of silence: the face settles
    per_slice = slice_len * sr // fps
    rem = len(speech) % per_slice
    if rem:
        speech = np.concatenate([speech, np.zeros(per_slice - rem, dtype=speech.dtype)])
    cached = sr * p["cached_audio_duration"]
    end_idx = p["cached_audio_duration"] * fps
    start_idx = end_idx - frame_num
    dq = deque([0.0] * cached, maxlen=cached)
    out = []
    for piece in speech.reshape(-1, per_slice):
        dq.extend(piece.tolist())
        emb = get_audio_embedding(PIPE, np.array(dq), start_idx, end_idx)
        vid = run_pipeline(PIPE, emb)[motion:]
        out.append(vid.cpu().numpy().astype(np.uint8))
    return list(np.concatenate(out, axis=0))


def render_reply(wav_bytes: bytes, source: Path, out_mp4: Path) -> float:
    TMP.mkdir(parents=True, exist_ok=True)
    stem = f"fh_{os.getpid()}_{int(time.time() * 1000)}"
    wav = TMP / f"{stem}.wav"
    wav.write_bytes(wav_bytes)
    silent = TMP / f"{stem}_v.mp4"
    t0 = time.time()
    try:
        prep = prepare(source)
        frames = _frames_for(str(wav), prep["crop_png"], tail_s=TAIL_S)
        rest = np.array(Image.open(prep["crop_png"].parent / "talk_rest.jpg").convert("RGB")).astype(np.float32)
        # start exactly on the rest frame and land exactly back on it, so a reply can follow any other clip without a visible join
        frames = _ease(frames, rest, n_in=EASE_IN_FRAMES, n_out=min(EASE_FRAMES, len(frames)))
        proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "512x512", "-r", "25", "-i", "-",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "17", "-bf", "0", str(silent)],
            stdin=subprocess.PIPE,
        )
        for fr in frames:
            proc.stdin.write(np.ascontiguousarray(fr).tobytes())
        proc.stdin.close()
        proc.wait()
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(silent), "-i", str(wav), "-c:v", "copy", "-af", "apad", "-c:a", "aac", "-shortest", str(out_mp4)], check=True)
    finally:
        for p in (wav, silent):
            p.unlink(missing_ok=True)
    return time.time() - t0


# ---------------------------------------------------------------- the waiting animations (idle bank)
BANK_COUNT = int(os.environ.get("FLASHHEAD_BANK_COUNT", "14"))
BANK_SECONDS = float(os.environ.get("FLASHHEAD_BANK_SECONDS", "6"))
_bank = {"running": False, "done": 0, "total": 0}


def _align_to(frame: np.ndarray, rest: np.ndarray, small: int = 128):
    """The affine move (2x3, full-size pixels) that puts `frame` on top of `rest` (head position, tilt and size), or None when it cannot be found."""
    try:
        k = small / frame.shape[0]
        g = lambda im: cv2.GaussianBlur(cv2.cvtColor(cv2.resize(im.astype(np.uint8), (small, small), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY), (5, 5), 0)  # noqa: E731
        M = np.eye(2, 3, dtype=np.float32)
        cv2.findTransformECC(g(rest), g(frame), M, cv2.MOTION_AFFINE, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5), None, 5)
        M[:, 2] /= k
        return M
    except Exception:
        return None


def _ease(frames, rest, n_in=8, n_out=10):
    """Start exactly on the rest frame and land exactly back on it, so any clip can follow any other without a visible join.

    A plain cross-fade between two different head positions shows two faces at once (double eyes, a smeared nose). So each frame in a blend is
    first moved toward the rest position (by the part of the blend that is still 'rest'), and only then faded: the head glides back to where
    the waiting animation will take over, instead of ghosting there."""
    sm = lambda t: t * t * t * (t * (6 * t - 15) + 10)  # noqa: E731  (smootherstep: no sudden start or stop)
    out = [f.astype(np.float32) for f in frames]
    I = np.eye(2, 3, dtype=np.float32)

    def blend(k, w):  # w = how much of the real frame is shown (1 = all of it, 0 = the rest frame)
        f = frames[k]
        M = _align_to(f, rest) if w < 0.98 else None
        if M is not None:
            a = 1.0 - w
            f = cv2.warpAffine(f.astype(np.float32), I * (1 - a) + M * a, (f.shape[1], f.shape[0]), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
        else:
            f = f.astype(np.float32)
        return (1 - w) * rest + w * f

    for j in range(min(n_in, len(out))):
        out[j] = blend(j, sm((j + 1) / n_in))
    for j in range(min(n_out, len(out))):
        k = len(out) - n_out + j
        out[k] = blend(k, 1 - sm((j + 1) / n_out))
    return [np.clip(f, 0, 255).astype(np.uint8) for f in out]


def build_bank(source: Path, count: int = BANK_COUNT, seconds: float = BANK_SECONDS) -> None:
    """Fresh set of waiting animations: FlashHead given near-silence, a different seed each time (blinks, small head moves, breathing)."""
    if _bank["running"]:
        return
    import soundfile as sf

    _bank.update(running=True, done=0, total=count)
    try:
        with LOCK:
            prep = prepare(source)
        d = source.parent / "idle_bank"
        d.mkdir(exist_ok=True)
        for old in list(d.glob("*.mp4")) + list(d.glob("*.webm")) + [d / "manifest.json"]:
            old.unlink(missing_ok=True)  # a fresh set: this folder holds only what this function makes
        rest_img = Image.open(source.parent / "talk_rest.jpg").convert("RGB")
        rest_img.save(d / "rest.jpg", quality=95)
        rest = np.array(rest_img).astype(np.float32)
        TMP.mkdir(parents=True, exist_ok=True)
        clips = []
        built = time.time()  # written into the manifest: the page adds it to every clip address, so a browser never replays an older set
        for i in range(count):
            wav = TMP / f"fh_idle_{i}.wav"
            rng = np.random.default_rng(1000 + i)
            sf.write(wav, (rng.standard_normal(int(seconds * 16000)) * 0.0008).astype(np.float32), 16000)
            with LOCK:
                frames = _frames_for(str(wav), prep["crop_png"], seed=1000 + i)
            wav.unlink(missing_ok=True)
            frames = _ease(frames, rest)
            name = f"calm_{i:03d}"
            proc = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "512x512", "-r", "25", "-i", "-",
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "17", "-bf", "0", str(d / f"{name}.mp4")],
                stdin=subprocess.PIPE,
            )
            for fr in frames:
                proc.stdin.write(np.ascontiguousarray(fr).tobytes())
            proc.stdin.close()
            proc.wait()
            clips.append({"id": name, "kind": "breathe", "file": f"{name}.mp4", "dur": round(len(frames) / 25, 2), "webm": None, "think": False})
            (d / "manifest.json").write_text(
                json.dumps({"rest": "rest.jpg", "space": "crop", "width": 512, "height": 512, "fps": 25, "built": built, "clips": clips}, indent=1), encoding="utf-8")
            _bank["done"] = i + 1
            print(f"idle bank: {name} ({i + 1}/{count})", flush=True)
    finally:
        _bank["running"] = False


def _bank_ready(source: Path) -> bool:
    try:
        j = json.loads((source.parent / "idle_bank" / "manifest.json").read_text(encoding="utf-8"))
        return j.get("space") == "crop" and len(j.get("clips", [])) >= 6
    except Exception:
        return False


@app.post("/idle_bank")
def idle_bank(source: str, count: int = BANK_COUNT, force: int = 0):
    """Build the waiting animations in the background (a no-op if a good set exists, unless force=1)."""
    src = Path(source)
    if _bank["running"] or (_bank_ready(src) and not force):
        return {"ok": True, "started": False, **_bank}
    threading.Thread(target=build_bank, args=(src, count), daemon=True).start()
    return {"ok": True, "started": True, **_bank}


@app.get("/idle_bank_status")
def idle_bank_status():
    return _bank


@app.get("/health")
def health():
    return {"ok": True, "engine": "flashhead-lite-4", "out_size": 512}


@app.post("/prepare")
def prepare_person(source: str):
    """Do the one-time photo preparation now (the app calls this when a person's photo is chosen) instead of on their first reply."""
    try:
        with LOCK:
            prepare(Path(source))
        try:  # some pictures (stylized or cartoon faces) do badly when the engine animates them at rest: stay on the still picture
            wants_bank = json.loads((Path(source).parent / "options.json").read_text(encoding="utf-8")).get("idle_bank", True)
        except Exception:
            wants_bank = True
        if wants_bank and not _bank_ready(Path(source)) and not _bank["running"]:  # the first time a person is seen: make their waiting animations too
            threading.Thread(target=build_bank, args=(Path(source),), daemon=True).start()
        return {"ok": True}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)[:300]}, status_code=500)


@app.post("/render")
async def render(request: Request, source: str, out: str):
    data = await request.body()  # the speech, as WAV
    import asyncio

    def work():
        with LOCK:
            return render_reply(data, Path(source), Path(out))

    try:
        secs = await asyncio.to_thread(work)
        if LEAN:
            torch.cuda.empty_cache()
        return {"ok": True, "path": out, "seconds": round(secs, 2)}
    except Exception as e:  # the app falls back to the animated face on any failure
        return JSONResponse({"ok": False, "error": str(e)[:300]}, status_code=500)


if __name__ == "__main__":
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8771)
    a = ap.parse_args()
    # one short warm-up so the one-time compile happens now, not during the first reply
    try:
        warm = TMP / "fh_warmup.wav"
        TMP.mkdir(parents=True, exist_ok=True)
        import soundfile as sf

        sf.write(warm, np.zeros(16000, dtype=np.float32), 16000)
        _frames_for(str(warm), next(ROOT.glob("data/profiles/*/cache/face/talk_crop.png"), None) or Path(FH / "examples" / "girl.png"))
        if LEAN:
            torch.cuda.empty_cache()  # small cards only: hand back scratch memory the warm-up left behind
        print("warm-up done", flush=True)
    except Exception as e:  # not fatal: the first reply just takes longer
        print("warm-up skipped:", e, flush=True)
    print(f"FlashHead render service on http://127.0.0.1:{a.port}/", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")
