"""Local render service: speech WAV in -> talking-head MP4 out, made by SoulX-FlashHead-Lite (4 steps). Nothing leaves this machine.

    ml\\realtime\\flashhead\\.venv\\Scripts\\python.exe ml\\realtime\\flashhead_render_server.py [--port 8771]

    POST /render?source=<photo path>&out=<mp4 path>   body = WAV bytes   ->  {"ok": true, "path": "...", "seconds": 4.2}
    GET  /health

Same interface the app already used for Ditto. How a reply is made:
  1. the person is cut out of the source photo (cinematic/matte.png, cleaned, plus optional per-person fixes in matte_fix.json)
     and put on a flat gray background, because a busy background confuses the model (shelf items turn up in the hair);
  2. FlashHead renders the head-and-shoulders crop (512x512) speaking the audio;
  3. the gray is keyed away again and the person is pasted back into the full photo, so the video has the same framing as the
     photo and the page can treat it like any other clip.
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
from flash_head.src.pipeline.flash_head_pipeline import FlashHeadPipeline  # noqa: F401  (import check)

CKPT = str(FH / "models" / "SoulX-FlashHead-1_3B")
W2V = str(FH / "models" / "wav2vec2-base-960h")
OUT_SIZE = int(os.environ.get("FLASHHEAD_OUT_SIZE", "768"))  # the finished video is this many pixels square
GRAY_BGR = (128, 118, 110)  # the flat background the model sees
SEED = 42
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
def _clean_matte(photo: np.ndarray, matte_png: Path, fix_json: Path) -> np.ndarray:
    """0..1 person mask the size of the photo: the stored cut-out without small attachments (a poster behind the hair),
    plus the polygons listed in matte_fix.json (things the cut-out wrongly kept: a chair arm, a sticker)."""
    m = cv2.imread(str(matte_png), cv2.IMREAD_UNCHANGED)
    m = m[..., 3] if m.ndim == 3 and m.shape[2] == 4 else (m if m.ndim == 2 else cv2.cvtColor(m, cv2.COLOR_BGR2GRAY))
    m = cv2.resize(m, (photo.shape[1], photo.shape[0])).astype(np.float32) / 255.0
    b = (m > 0.5).astype(np.uint8)
    b = cv2.morphologyEx(b, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(b)
    if n > 1:
        b = (lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    if fix_json.exists():
        for poly in json.loads(fix_json.read_text(encoding="utf-8")).get("erase", []):
            cv2.fillPoly(b, [np.array(poly, np.int32)], 0)
    return b.astype(np.float32)


def prepare(source: Path) -> dict:
    """One-time preparation of a person's photo (cut-out, flat gray, FlashHead crop, empty-room plate). The results are saved next
    to the photo (talk_crop.png, talk_plate.png, talk_meta.json) and only rebuilt when the photo, its cut-out or its fixes change."""
    d = source.parent
    stamp = [source.stat().st_mtime, (d / "matte.png").stat().st_mtime, ((d / "matte_fix.json").stat().st_mtime if (d / "matte_fix.json").exists() else 0), OUT_SIZE, "plate-v2"]
    hit = _PREP.get(str(source))
    if hit and hit["stamp"] == stamp:
        return hit
    meta_f = d / "talk_meta.json"
    meta = json.loads(meta_f.read_text(encoding="utf-8")) if meta_f.exists() else {}
    photo = cv2.imread(str(source))
    H, W = photo.shape[:2]
    person = _clean_matte(photo, d / "matte.png", d / "matte_fix.json")
    if meta.get("stamp") != stamp or not (d / "talk_crop.png").exists() or not (d / "talk_plate.png").exists():
        soft = cv2.GaussianBlur(cv2.dilate(person, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))), (0, 0), 2.5)[..., None]
        flat = np.zeros_like(photo)
        flat[:] = GRAY_BGR
        rgb = cv2.cvtColor((photo * soft + flat * (1 - soft)).astype(np.uint8), cv2.COLOR_BGR2RGB)
        boxes, _ = FACE(rgb)
        if len(boxes) == 0:
            raise RuntimeError("no face found in the photo")
        x1, y1, x2, y2 = boxes[0][0] * W, boxes[0][1] * H, boxes[0][2] * W, boxes[0][3] * H
        cx, cy, side = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * 2.0  # FlashHead's recipe: a square 2x the face, face 55% down
        box = [int(max(0, cx - side * 0.5)), int(max(0, cy - side * 0.55)), int(min(W, cx + side * 0.5)), int(min(H, cy + side * 0.45))]
        bx1, by1, bx2, by2 = box
        Image.fromarray(cv2.resize(rgb[by1:by2, bx1:bx2], (512, 512), interpolation=cv2.INTER_AREA)).save(d / "talk_crop.png")
        Image.fromarray(rgb).save(d / "talk_flat.png")
        # the room with the person painted out inside the crop: no ghost of the still photo shows around the moving head
        plate = photo.copy()
        region = photo[by1:by2, bx1:bx2]
        pm = cv2.dilate((person[by1:by2, bx1:bx2] > 0.5).astype(np.uint8) * 255, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
        mg = int(0.12 * (by2 - by1))  # near the left, right and bottom edge the person carries on out of the crop (shoulders, shirt):
        pm[:, :mg] = 0                # keep the real pixels there so the video blends into them without a seam
        pm[:, -mg:] = 0
        pm[-mg:, :] = 0
        s3 = 3
        small = cv2.inpaint(cv2.resize(region, (region.shape[1] // s3, region.shape[0] // s3)), cv2.resize(pm, (region.shape[1] // s3, region.shape[0] // s3)), 5, cv2.INPAINT_TELEA)
        plate[by1:by2, bx1:bx2] = np.where(pm[..., None] > 0, cv2.resize(small, (region.shape[1], region.shape[0])), region)
        cv2.imwrite(str(d / "talk_plate.png"), plate)
        meta = {"stamp": stamp, "box": box}
        meta_f.write_text(json.dumps(meta), encoding="utf-8")
        print(f"prepared {source.parent.parent.parent.name}: crop box {box}", flush=True)
    plate = cv2.imread(str(d / "talk_plate.png"))
    bx1, by1, bx2, by2 = meta["box"]
    k = OUT_SIZE / W
    box_o = (int(round(bx1 * k)), int(round(by1 * k)), int(round(bx2 * k)), int(round(by2 * k)))
    bw, bh = box_o[2] - box_o[0], box_o[3] - box_o[1]
    pm_crop = person[by1:by2, bx1:bx2]
    inner = cv2.erode((pm_crop > 0.5).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))).astype(np.float32)
    outer = cv2.dilate((pm_crop > 0.5).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61))).astype(np.float32)
    f = max(6, int(min(bw, bh) * 0.04))  # feather the paste window so the join with the still photo is invisible
    win = np.ones((bh, bw), np.float32)
    ramp = np.linspace(0, 1, f, dtype=np.float32)
    win[:f, :] *= ramp[:, None]; win[-f:, :] *= ramp[::-1][:, None]; win[:, :f] *= ramp[None, :]; win[:, -f:] *= ramp[::-1][None, :]
    hit = dict(
        stamp=stamp, crop_png=d / "talk_crop.png", box_o=box_o, bw=bw, bh=bh,
        base=cv2.resize(plate, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_AREA),
        inner=cv2.resize(inner, (bw, bh)), outer=cv2.resize(outer, (bw, bh)), win=win,
    )
    _PREP[str(source)] = hit
    return hit


# ---------------------------------------------------------------- rendering
def _frames_for(wav_path: str, crop_png: Path) -> list[np.ndarray]:
    """FlashHead's streaming loop for one utterance -> list of 512x512 RGB uint8 frames."""
    get_base_data(PIPE, cond_image_path_or_dir=str(crop_png), base_seed=SEED, use_face_crop=False)
    p = get_infer_params()
    sr, fps, frame_num = p["sample_rate"], p["tgt_fps"], p["frame_num"]
    motion = p["motion_frames_num"]
    slice_len = frame_num - motion
    speech, _ = librosa.load(wav_path, sr=sr, mono=True)
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


def _composite(frame_rgb: np.ndarray, prep: dict) -> np.ndarray:
    """One FlashHead frame (person on gray) -> the full picture: the room, with the person pasted in."""
    bw, bh = prep["bw"], prep["bh"]
    f = cv2.resize(frame_rgb, (bw, bh), interpolation=cv2.INTER_AREA).astype(np.float32)
    corners = np.concatenate([frame_rgb[:40, :40].reshape(-1, 3), frame_rgb[:40, -40:].reshape(-1, 3)])
    bg = np.median(corners, axis=0)  # the gray as the model happened to render it in this frame
    d = np.linalg.norm(f - bg, axis=2)
    key = np.clip((d - 14.0) / 26.0, 0, 1)
    alpha = np.maximum(key * prep["outer"], prep["inner"]) * prep["win"]  # key the gray away; the cut-out protects hair and shirt
    x1, y1, x2, y2 = prep["box_o"]
    out = prep["base"].copy()
    region = out[y1:y2, x1:x2].astype(np.float32)
    person_bgr = f[..., ::-1]
    out[y1:y2, x1:x2] = (person_bgr * alpha[..., None] + region * (1 - alpha[..., None])).astype(np.uint8)
    return out


def render_reply(wav_bytes: bytes, source: Path, out_mp4: Path) -> float:
    TMP.mkdir(parents=True, exist_ok=True)
    stem = f"fh_{os.getpid()}_{int(time.time() * 1000)}"
    wav = TMP / f"{stem}.wav"
    wav.write_bytes(wav_bytes)
    silent = TMP / f"{stem}_v.mp4"
    t0 = time.time()
    try:
        prep = prepare(source)
        frames = _frames_for(str(wav), prep["crop_png"])
        proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{OUT_SIZE}x{OUT_SIZE}", "-r", "25", "-i", "-",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "18", "-bf", "0", str(silent)],
            stdin=subprocess.PIPE,
        )
        for fr in frames:
            proc.stdin.write(np.ascontiguousarray(_composite(fr, prep)).tobytes())
        proc.stdin.close()
        proc.wait()
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(silent), "-i", str(wav), "-c:v", "copy", "-c:a", "aac", "-shortest", str(out_mp4)], check=True)
    finally:
        for p in (wav, silent):
            p.unlink(missing_ok=True)
    return time.time() - t0


@app.get("/health")
def health():
    return {"ok": True, "engine": "flashhead-lite-4", "out_size": OUT_SIZE}


@app.post("/prepare")
def prepare_person(source: str):
    """Do the one-time photo preparation now (the app calls this when a person's photo is chosen) instead of on their first reply."""
    try:
        with LOCK:
            prepare(Path(source))
        return {"ok": True}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)[:300]}, status_code=500)


@app.post("/render")
async def render(request: Request, source: str, out: str, emotion: str = "", intensity: float = 0.55, calm_eyes: int = 0):
    data = await request.body()  # emotion / intensity / calm_eyes are accepted (the app sends them) and not used by FlashHead
    import asyncio

    def work():
        with LOCK:
            return render_reply(data, Path(source), Path(out))

    try:
        secs = await asyncio.to_thread(work)
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
        _frames_for(str(warm), next(ROOT.glob("data/media/*/cinematic/talk_crop.png"), None) or Path(FH / "examples" / "girl.png"))
        print("warm-up done", flush=True)
    except Exception as e:  # not fatal: the first reply just takes longer
        print("warm-up skipped:", e, flush=True)
    print(f"FlashHead render service on http://127.0.0.1:{a.port}/", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")
