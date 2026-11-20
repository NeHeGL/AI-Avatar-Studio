"""Live idle layer: the person's photo, alive in real time (blinks, breathing, slow head drift, an occasional glance).
Runs locally (LivePortrait); nothing leaves this machine. The page shows this stream whenever no clip or reply video is playing.

    ml\\realtime\\lpenv\\Scripts\\python.exe ml\\realtime\\idle_server.py [--port 8770] [--user jeff_molofee] [--fps 20]

    GET  /stream             MJPEG video of the idle avatar
    POST /use?user=<slug>    switch to data/media/<slug>/cinematic/source.jpg (the same wide photo the clips use)
    GET  /health
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
LP = HERE / "LivePortrait"
sys.path.insert(0, str(LP))
os.chdir(LP)  # LivePortrait resolves its weights relative to its own folder

import cv2
import numpy as np
import torch
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse

from src.config.crop_config import CropConfig
from src.config.inference_config import InferenceConfig
from src.live_portrait_pipeline import LivePortraitPipeline
from src.utils.crop import paste_back, prepare_paste_back
from src.utils.retargeting_utils import calc_eye_close_ratio, calc_lip_close_ratio

OUT_PX = int(os.environ.get("IDLE_PX", "720"))  # longest side of the streamed picture


class Avatar:
    """One person's photo prepared for animation. The (heavy) LivePortrait models are shared between avatars."""

    def __init__(self, photo: Path, pipe: LivePortraitPipeline):
        self.pipe = pipe
        # per-person options (cinematic/options.json): {"glasses": true} calms everything around the eyes, because the face
        # model drags spectacles along with the eyelids and the eyeballs
        try:
            opts = json.loads((photo.parent / "options.json").read_text(encoding="utf-8"))
        except Exception:
            opts = {}
        self.glasses = bool(opts.get("glasses"))
        self.rest_blink = 0.0 if self.glasses else REST_BLINK
        self.blink_scale = 0.3 if self.glasses else 1.0
        self.lock = threading.RLock()
        self.w = pipe.live_portrait_wrapper
        crop = CropConfig()
        bgr = cv2.imread(str(photo))
        if bgr is None:
            raise FileNotFoundError(photo)
        h, wd = bgr.shape[:2]
        k = OUT_PX / max(h, wd)
        if k < 1:
            bgr = cv2.resize(bgr, (round(wd * k), round(h * k)), interpolation=cv2.INTER_AREA)
        self.src_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        info = pipe.cropper.crop_source_image(self.src_rgb, crop)
        if info is None:
            raise RuntimeError("No face found in the photo")
        self.crop_info = info
        self.src_lmk = info["lmk_crop"]
        self.mask = prepare_paste_back(pipe.live_portrait_wrapper.inference_cfg.mask_crop, info["M_c2o"], dsize=(self.src_rgb.shape[1], self.src_rgb.shape[0]))
        I_s = self.w.prepare_source(info["img_crop_256x256"])
        self.s_info = self.w.get_kp_info(I_s)
        self.f_s = self.w.extract_feature_3d(I_s)
        self.x_s = self.w.transform_keypoint(self.s_info)
        self.eye0 = float(calc_eye_close_ratio(self.src_lmk[None]).mean())
        dev = self.w.device
        self.c_s_e = torch.from_numpy(calc_eye_close_ratio(self.src_lmk[None])).float().to(dev)
        self.eye_open_delta = self.w.retarget_eye(self.x_s, torch.cat([self.c_s_e, self.c_s_e.mean(dim=1, keepdim=True)], dim=1))
        self.lip0 = float(calc_lip_close_ratio(self.src_lmk[None]).mean())
        self.c_s_l = torch.from_numpy(calc_lip_close_ratio(self.src_lmk[None])).float().to(dev)
        self.lip_rest_delta = self.w.retarget_lip(self.x_s, torch.cat([self.c_s_l, self.c_s_l], dim=1))

    def library(self, clip_dir: Path, step: float = 0.25) -> dict:
        """Facial expressions taken from this person's own clips: how each sampled frame's face differs from their neutral photo.
        Keys look like 'reaction_laugh@1.25'. Only the expression is kept (no head pose), so it can be mixed into any movement."""
        lib = {}
        for mp4 in sorted(clip_dir.glob("*.mp4")):
            cap = cv2.VideoCapture(str(mp4))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            t = 0.0
            while int(t * fps) < n:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
                ok, bgr = cap.read()
                if not ok:
                    break
                try:
                    m = self.motion_of(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                    if m is not None:
                        lib[f"{mp4.stem}@{t:.2f}"] = m["exp"]
                except Exception:
                    pass
                t += step
            cap.release()
        return lib

    @torch.no_grad()
    def motion_of(self, rgb: np.ndarray):
        """How this frame differs from the neutral photo (pose + expression), or None if no face is found."""
        info = self.pipe.cropper.crop_source_image(rgb, CropConfig())
        if info is None:
            return None
        k = self.w.get_kp_info(self.w.prepare_source(info["img_crop_256x256"]))
        s = self.s_info
        return {"exp": k["exp"] - s["exp"], "yaw": k["yaw"] - s["yaw"], "pitch": k["pitch"] - s["pitch"],
                "roll": k["roll"] - s["roll"], "t": k["t"] - s["t"]}

    @torch.no_grad()
    def render(self, *a, **k) -> np.ndarray:
        with self.lock:
            return self._render(*a, **k)

    @torch.no_grad()
    def _render(self, yaw=0.0, pitch=0.0, roll=0.0, dx=0.0, dy=0.0, blink=0.0, mouth=0.0, seed=None, w=0.0, exp=None, exp_w=0.0, emotion=None, emo_w=0.0) -> np.ndarray:
        """blink 0..1 (1 = closed), mouth 0..1 (1 = wide open); angles in degrees. Returns an RGB picture the size of the source photo."""
        info = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in self.s_info.items()}
        info["yaw"] = info["yaw"] + yaw
        info["pitch"] = info["pitch"] + pitch
        info["roll"] = info["roll"] + roll
        info["t"][:, 0] += dx
        info["t"][:, 1] += dy
        if seed is not None and w > 0.01:  # start from the pose/expression of the frame that was just on screen
            info["exp"] = info["exp"] + w * seed["exp"]
            info["yaw"] = info["yaw"] + w * seed["yaw"]
            info["pitch"] = info["pitch"] + w * seed["pitch"]
            info["roll"] = info["roll"] + w * seed["roll"]
            info["t"] = info["t"] + w * seed["t"]
        if emotion and emo_w > 0.01:
            e_delta, e_blink = emotion_table(self.s_info["exp"].device).get(emotion, (None, 0.0))
            if e_delta is not None:
                if self.glasses:  # no eye or gaze offsets (they cross the eyes and shove the glasses about)
                    mask = torch.ones_like(e_delta)
                    mask[:, [1, 2, 11, 13, 15, 16], :] = 0.0
                    e_delta, e_blink = e_delta * mask, 0.0
                info["exp"] = info["exp"] + emo_w * e_delta
                blink = max(blink, e_blink * emo_w)
        if exp is not None and exp_w > 0.01:  # a facial expression (smile, raised brows, ...) laid over the neutral face
            info["exp"] = info["exp"] + exp_w * exp
        x_d = self.w.transform_keypoint(info)
        if self.glasses:
            blink = max(blink, 0.0)  # never wider than the photo; blinks are small so the glasses hardly move
        blink = self.rest_blink + (blink * self.blink_scale if blink > 0 else blink) * (1.0 - self.rest_blink)
        if abs(blink) > 0.01:
            eye_t = torch.tensor([[self.eye0 * (1.0 - blink)]], device=self.w.device)
            x_d = x_d + self.w.retarget_eye(self.x_s, torch.cat([self.c_s_e, eye_t], dim=1)) - self.eye_open_delta
        if mouth > 0.01:
            lip_t = torch.tensor([[self.lip0 + 0.55 * mouth]], device=self.w.device)
            x_d = x_d + self.w.retarget_lip(self.x_s, torch.cat([self.c_s_l, lip_t], dim=1)) - self.lip_rest_delta
        x_d = self.w.stitching(self.x_s, x_d)
        out = self.w.warp_decode(self.f_s, self.x_s, x_d)
        crop = self.w.parse_output(out["out"])[0]
        return paste_back(crop, self.crop_info["M_c2o"], self.src_rgb, self.mask)


EXPR_FILE = HERE / "expressions.json"
_expr_cache = {"mtime": 0.0, "table": {}}


def emotion_table(device) -> dict:
    """name -> (21x3 expression offset, extra eye-narrowing). Recipes live in expressions.json (the same for every face,
    because they are offsets of the face model's own control points) and are re-read whenever the file changes."""
    try:
        mt = EXPR_FILE.stat().st_mtime
    except OSError:
        return {}
    if mt != _expr_cache["mtime"]:
        table = {}
        for name, rec in json.loads(EXPR_FILE.read_text(encoding="utf-8")).items():
            e = torch.zeros(1, 21, 3, device=device)
            for kp, ax, amt in rec.get("exp", []):
                e[0, int(kp), "xyz".index(ax)] += float(amt)
            table[name] = (e, float(rec.get("blink", 0.0)))
        _expr_cache.update(mtime=mt, table=table)
    return _expr_cache["table"]


REST_BLINK = -0.22  # < 0 = eyes slightly wider than in the photo; every animation is relative to this rest face

MATTE_MODEL = HERE / "models" / "u2net_human_seg.onnx"


_seg = {"sess": None}


def matte_of(rgb: np.ndarray) -> np.ndarray:
    """How much of each pixel is the person, 0..1 (hair included), for ANY frame. Used for the still photo and, frame by frame,
    for every animation, so the outline follows the head when it tilts or moves."""
    if _seg["sess"] is None:
        import onnxruntime as ort

        try:
            _seg["sess"] = ort.InferenceSession(str(MATTE_MODEL), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        except Exception:
            _seg["sess"] = ort.InferenceSession(str(MATTE_MODEL), providers=["CPUExecutionProvider"])
    h, w = rgb.shape[:2]
    x = cv2.resize(rgb, (320, 320), interpolation=cv2.INTER_AREA).astype(np.float32)
    x = x / max(float(x.max()), 1e-6)
    x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    sess = _seg["sess"]
    y = sess.run(None, {sess.get_inputs()[0].name: x.transpose(2, 0, 1)[None].astype(np.float32)})[0][0, 0]
    y = (y - y.min()) / max(float(y.max() - y.min()), 1e-6)
    m = cv2.resize(y, (w, h), interpolation=cv2.INTER_CUBIC)
    m = np.clip((m - 0.5) / 0.3, 0.0, 1.0)  # firm edge, soft transition
    k = max(3, w // 160) | 1
    m = cv2.erode(m, np.ones((k, k), np.uint8))  # pull the edge in a little so no old background shows as a halo
    # keep only the person: drop isolated specks and fragments of background (a bit of chair, a shelf corner)
    n, lab, stats, _ = cv2.connectedComponentsWithStats((m > 0.5).astype(np.uint8), connectivity=8)
    if n > 2:
        keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        m = m * cv2.dilate((lab == keep).astype(np.uint8), np.ones((k + 2, k + 2), np.uint8))
    return cv2.GaussianBlur(m, (0, 0), max(1.2, w / 350))


def make_matte(photo: Path, out: Path) -> bool:
    """Cut the person out of the prepared photo: writes a white PNG whose alpha is the person. The page uses it to swap the
    background behind still pictures and talking videos; animations carry their own per-frame cut-out."""
    if out.exists() and out.stat().st_mtime >= photo.stat().st_mtime:
        return True
    if not MATTE_MODEL.exists():
        return False
    img = cv2.imread(str(photo))
    h, w = img.shape[:2]
    m = matte_of(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    rgba = np.dstack([np.full((h, w, 3), 255, np.uint8), (m * 255).astype(np.uint8)])
    cv2.imwrite(str(out), rgba)
    print(f"matte written: {out.name}", flush=True)
    return True


def smooth(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def bump(x: float) -> float:
    """Smooth 0 -> 1 -> 0 over x in [0, 1]."""
    return math.sin(math.pi * min(max(x, 0.0), 1.0)) ** 2


class Gesture:
    """A freshly invented bit of body language: a few random poses strung together with smooth moves.
    It starts and ends at the rest pose, so it always hands back to the idle loop (and to the talking video) on the same frame."""

    KINDS = ["look_around", "think", "tilt", "sigh", "double_take", "ponder", "curious", "perk"]

    def __init__(self, rnd: random.Random, now: float, kind: str | None = None):
        self.t0 = now
        kind = kind or rnd.choice(self.KINDS)
        sgn = rnd.choice([-1, 1])
        A = lambda lo, hi: rnd.uniform(lo, hi)  # noqa: E731
        # keyframes: (seconds, yaw, pitch, roll, squint 0..1, mouth 0..1)
        if kind == "look_around":
            k = [(0, 0, 0, 0, 0, 0), (A(.7, 1.0), sgn * A(5, 9), A(-1, 1), sgn * A(.5, 1.5), 0, 0), (A(1.6, 2.4), -sgn * A(3, 6), A(-1, 1), -sgn * A(.3, 1), 0, 0), (A(3.0, 3.8), 0, 0, 0, 0, 0)]
        elif kind == "think":
            k = [(0, 0, 0, 0, 0, 0), (A(.8, 1.2), sgn * A(4, 7), A(2.5, 5), sgn * A(1, 3), A(.1, .3), 0), (A(2.4, 3.4), sgn * A(3, 6), A(2, 4), sgn * A(1, 2.5), A(.1, .25), 0), (A(3.8, 4.6), 0, 0, 0, 0, 0)]
        elif kind == "tilt":
            k = [(0, 0, 0, 0, 0, 0), (A(.8, 1.3), sgn * A(1, 3), A(-.5, 1.5), sgn * A(3.5, 6), 0, 0), (A(2.0, 3.0), sgn * A(1, 3), A(0, 1.5), sgn * A(3, 5.5), 0, 0), (A(3.2, 4.0), 0, 0, 0, 0, 0)]
        elif kind == "sigh":
            k = [(0, 0, 0, 0, 0, 0), (A(.9, 1.2), 0, A(-2.5, -1.5), 0, A(.2, .4), A(.12, .22)), (A(1.7, 2.1), 0, A(2.5, 3.8), A(-.5, .5), A(.5, .8), A(.2, .35)), (A(2.8, 3.4), 0, 0, 0, 0, 0)]
        elif kind == "double_take":
            k = [(0, 0, 0, 0, 0, 0), (.35, sgn * A(6, 9), A(-.5, 1), sgn * A(.5, 1.5), 0, 0), (A(.9, 1.2), sgn * A(6, 9), 0, sgn, 0, 0), (A(1.5, 1.9), -sgn * A(1, 3), 0, 0, A(.3, .6), 0), (A(2.6, 3.2), 0, 0, 0, 0, 0)]
        elif kind == "ponder":
            k = [(0, 0, 0, 0, 0, 0), (A(.7, 1.1), 0, A(-3, -1.5), 0, A(.1, .3), 0), (A(1.5, 2.2), sgn * A(2, 4), A(-2.5, -1), sgn * A(.5, 1.5), A(.2, .4), A(.05, .12)), (A(2.8, 3.6), 0, 0, 0, 0, 0)]
        elif kind == "perk":  # something caught its attention: eyes open wide, head lifts, then relaxes
            k = [(0, 0, 0, 0, 0, 0), (A(.35, .6), sgn * A(.5, 2), A(1.5, 3), 0, -A(.5, .9), 0), (A(1.4, 2.2), sgn * A(.5, 2), A(1, 2.5), sgn * A(.3, 1), -A(.4, .8), 0), (A(2.4, 3.2), 0, 0, 0, 0, 0)]
        else:  # curious: lean in toward the screen
            k = [(0, 0, 0, 0, 0, 0), (A(.6, 1.0), sgn * A(1, 2.5), A(1.5, 3), sgn * A(1.5, 3), 0, 0), (A(2.0, 3.0), sgn * A(1, 2.5), A(1, 2.5), sgn * A(1.5, 3), 0, A(.05, .15)), (A(3.2, 4.0), 0, 0, 0, 0, 0)]
        self.kind, self.k = kind, k
        self.dur = k[-1][0]

    def at(self, now: float):
        t = now - self.t0
        if t >= self.dur:
            return None
        k = self.k
        for i in range(len(k) - 1):
            if k[i][0] <= t <= k[i + 1][0]:
                f = smooth((t - k[i][0]) / max(1e-6, k[i + 1][0] - k[i][0]))
                v = [a + (b - a) * f for a, b in zip(k[i][1:], k[i + 1][1:])]
                v[3] = min(v[3], 0.18)  # never the droopy look
                return tuple(v)
        return None


def hermite(keys, t):
    """Smooth value at time t through (time, value) keys: a cubic curve with continuous speed (no stop-and-go at the keys).
    The first and last keys have zero speed, so a clip eases out of and back into the rest pose."""
    n = len(keys)
    if t <= keys[0][0]:
        return keys[0][1]
    if t >= keys[-1][0]:
        return keys[-1][1]
    i = max(j for j in range(n - 1) if keys[j][0] <= t)
    t0, v0 = keys[i]
    t1, v1 = keys[i + 1]

    def slope(k):
        if k <= 0 or k >= n - 1:
            return 0.0
        return (keys[k + 1][1] - keys[k - 1][1]) / max(1e-6, keys[k + 1][0] - keys[k - 1][0])

    h = t1 - t0
    u = (t - t0) / h
    m0, m1 = slope(i) * h, slope(i + 1) * h
    return (2 * u**3 - 3 * u**2 + 1) * v0 + (u**3 - 2 * u**2 + u) * m0 + (-2 * u**3 + 3 * u**2) * v1 + (u**3 - u**2) * m1


class Clip:
    """One pre-made waiting animation, invented on the spot from random poses joined by smooth curves, so no two are alike.
    Every clip starts and ends exactly on the rest frame (all offsets are zero there), so any clip can follow any other clip,
    or a pause on the rest frame, with no visible join. Moods are mild and mostly on the eyes and brows."""

    FPS = 25
    # (expression, strength at its peak): kept gentle so the face never looks forced
    MOOD = {"think": ("thoughtful", 0.45), "ponder": ("thoughtful", 0.35), "curious": ("interested", 0.6), "perk": ("interested", 0.5),
            "tilt": ("puzzled", 0.3), "sigh": ("concerned", 0.18), "double_take": ("surprised", 0.3), "smile": ("happy", 0.55),
            "reading": ("interested", 0.2), "look_up": ("thoughtful", 0.3), "glance_back": ("interested", 0.35),
            # goofy faces: short, subtle, rare
            "wink": ("wink", 1.0), "brow_up": ("brow_raise", 1.0), "pucker": ("pucker", 0.9), "stare": ("stare", 0.8)}

    def __init__(self, rnd: random.Random, kind: str):
        self.kind = kind
        A = rnd.uniform
        sg = rnd.choice([-1, 1])
        # keyframes: (time, yaw, pitch, roll, squint (<0 = eyes wider), mouth, mood strength 0..1)
        if kind == "breathe":
            D = A(2.2, 3.6)
            k = [(0,), (D * A(.35, .55), A(-2.5, 2.5), A(-1.2, 1.2), A(-.8, .8)), (D,)]
        elif kind == "look_around":
            D = A(4.2, 6.0)
            k = [(0,), (D * .2, sg * A(6, 10), A(-1, 1.5), sg * A(.5, 1.5)), (D * .45, sg * A(5, 9), A(-1, 1), sg * A(.3, 1)),
                 (D * .72, -sg * A(4, 8), A(-1, 1.5), -sg * A(.4, 1)), (D,)]
        elif kind == "look_up":
            D = A(3.6, 5.0)
            k = [(0,), (D * .25, sg * A(2, 5), A(4, 6.5), sg * A(.5, 1.5), 0, 0, 1), (D * .65, sg * A(3, 6), A(3, 6), sg * A(.5, 1.5), 0, 0, 1), (D,)]
        elif kind == "look_down":
            D = A(3.0, 4.5)
            k = [(0,), (D * .3, sg * A(1, 3), -A(3.5, 5.5), sg * A(.3, 1)), (D * .7, sg * A(2, 4), -A(3, 5), sg * A(.3, 1)), (D,)]
        elif kind == "reading":  # eyes sweep along lines of text
            D = A(5.5, 7.5)
            x = A(2.5, 3.5)
            k = [(0,), (D * .12, -x, -A(3, 4), 0, .08, 0, 1), (D * .30, x, -A(3.2, 4), 0, .08, 0, 1), (D * .42, -x, -A(2.8, 3.8), 0, .08, 0, 1),
                 (D * .60, x * 1.1, -A(3, 4), 0, .08, 0, 1), (D * .80, 0, -A(.5, 1.5)), (D,)]
        elif kind == "tilt":
            D = A(3.2, 4.4)
            k = [(0,), (D * .28, sg * A(1, 3), A(-.5, 1.5), sg * A(3.5, 6), 0, 0, 1), (D * .7, sg * A(1, 3), A(0, 1.5), sg * A(3, 5.5), 0, 0, 1), (D,)]
        elif kind == "sigh":
            D = A(3.0, 3.8)
            k = [(0,), (D * .32, 0, -A(2.2, 3.2), 0, A(.1, .18), A(.08, .16), 1), (D * .6, 0, A(2.2, 3.4), A(-.5, .5), A(.1, .2), A(.12, .22), 1), (D,)]
        elif kind == "double_take":
            D = A(3.0, 3.8)
            k = [(0,), (.45, sg * A(6, 9), A(-.5, 1), sg * A(.5, 1.5), 0, 0, 0), (D * .4, sg * A(6, 9), 0, sg, 0, 0, 1), (D * .62, -sg * A(1, 3), 0, 0, 0, 0, .5), (D,)]
        elif kind == "ponder":
            D = A(3.0, 4.0)
            k = [(0,), (D * .25, 0, -A(2, 3.2), 0, .08, 0, 1), (D * .6, sg * A(2, 4), -A(1.5, 2.5), sg * A(.5, 1.5), .1, A(.03, .08), 1), (D,)]
        elif kind == "think":
            D = A(3.8, 5.0)
            k = [(0,), (D * .22, sg * A(4, 7), A(2.5, 5), sg * A(1, 3), .06, 0, 1), (D * .65, sg * A(3, 6), A(2, 4), sg * A(1, 2.5), .06, 0, 1), (D,)]
        elif kind == "curious":
            D = A(3.2, 4.2)
            k = [(0,), (D * .25, sg * A(1, 2.5), A(1.5, 3), sg * A(1.5, 3), 0, 0, 1), (D * .68, sg * A(1, 2.5), A(1, 2.5), sg * A(1.5, 3), 0, A(.02, .08), 1), (D,)]
        elif kind == "perk":
            D = A(2.8, 3.6)
            k = [(0,), (D * .2, sg * A(.5, 2), A(1.5, 3), 0, -A(.4, .8), 0, 1), (D * .62, sg * A(.5, 2), A(1, 2.5), sg * A(.3, 1), -A(.3, .7), 0, 1), (D,)]
        elif kind == "smile":  # something pleasant crosses its mind
            D = A(3.2, 4.4)
            k = [(0,), (D * .3, sg * A(1, 3), A(.5, 2), sg * A(1.5, 3.5), 0, 0, 1), (D * .7, sg * A(.5, 2), A(0, 1.5), sg * A(1, 3), 0, 0, 1), (D,)]
        elif kind == "glance_back":  # looks off to one side, comes back to the screen, looks once more
            D = A(4.0, 5.5)
            k = [(0,), (D * .2, sg * A(5, 8), A(-1, 1)), (D * .42, sg * A(5, 8), A(-1, 1)), (D * .6, 0, A(.3, 1.2), 0, 0, 0, 1), (D * .78, sg * A(2, 4), 0), (D,)]
        elif kind in ("wink", "brow_up"):  # a quick gag: in, hold a beat, out
            D = A(1.9, 2.8)
            tilt = sg * A(1.5, 3.5)
            k = [(0,), (D * .3, sg * A(.5, 2), A(0, 1.5), tilt, 0, 0, 1), (D * .62, sg * A(.5, 2), A(0, 1.2), tilt, 0, 0, 1), (D,)]
        else:  # yawn
            kind = self.kind = "yawn"
            D = A(2.8, 3.2)
            k = None
        self.dur = D
        self.keys = None
        if k is not None:
            chan = [[] for _ in range(6)]  # yaw, pitch, roll, squint, mouth, mood
            for fr in k:
                t = fr[0]
                vals = list(fr[1:]) + [0] * (6 - len(fr[1:]))
                for c in range(6):
                    chan[c].append((t, float(vals[c])))
            self.keys = chan
        self.blinks = []
        if kind != "yawn":
            for _ in range(rnd.choice([0, 1, 1, 2])):
                self.blinks.append(rnd.uniform(0.5, max(0.6, D - 0.9)))
        self.ph = [rnd.uniform(0, 6.28) for _ in range(3)]
        self.frames = int(round(D * self.FPS))
        self.dur = self.frames / self.FPS

    def at(self, t: float) -> dict:
        D = self.dur
        e = bump(t / D)  # 0 at both ends, 1 in the middle: scales the little constant "alive" drift
        p = dict(yaw=0.0, pitch=0.0, roll=0.0, dx=0.0, dy=0.0035 * math.sin(2 * math.pi * t / D), blink=0.0, mouth=0.0, emotion=None, emo_w=0.0)
        if self.kind == "yawn":
            x = t / D
            hold = 1.0 if 0.22 < x < 0.6 else (smooth(x / 0.22) if x <= 0.22 else smooth((0.82 - x) / 0.22))
            hold = max(0.0, hold)
            p.update(mouth=0.28 * hold, blink=0.6 * hold, pitch=3.0 * hold, roll=0.8 * hold)
        else:
            tt = min(t, D)
            yaw, pitch, roll, squint, mouth, mood = (hermite(c, tt) for c in self.keys)
            p.update(yaw=yaw, pitch=pitch, roll=roll, blink=squint, mouth=max(0.0, mouth))
            if self.kind in self.MOOD:
                name, w = self.MOOD[self.kind]
                p["emotion"], p["emo_w"] = name, max(0.0, min(1.0, mood)) * w
        p["yaw"] += 0.45 * e * math.sin(2 * math.pi * 0.37 * t + self.ph[0])  # never perfectly still while it moves
        p["pitch"] += 0.3 * e * math.sin(2 * math.pi * 0.53 * t + self.ph[1])
        p["roll"] += 0.2 * e * math.sin(2 * math.pi * 0.29 * t + self.ph[2])
        for b in self.blinks:
            x = (t - b) / 0.19
            if 0 <= x <= 1:
                p["blink"] = max(p["blink"], math.sin(math.pi * x))
        return p


BANK_KINDS = {"breathe": 6, "look_around": 3, "look_up": 3, "look_down": 2, "reading": 2, "tilt": 2, "sigh": 2, "double_take": 1.5,
              "ponder": 3, "think": 3, "curious": 3, "perk": 2, "smile": 3, "glance_back": 2.5, "yawn": 0.8,
              "wink": 0.5, "brow_up": 0.5}
GOOFY_KINDS = ("wink", "brow_up")
GOOFY_LIFE_SEC = 25 * 60  # goofy faces are made at random and expire; they are never a permanent part of the bank
THINK_KINDS = ("think", "ponder", "curious", "tilt", "look_up", "reading", "glance_back")
MAX_BANK = 160  # per person: new ones keep being invented until there are this many
FFMPEG = HERE / "ffmpeg" / "ffmpeg-master-latest-win64-gpl" / "bin" / "ffmpeg.exe"


def _ffmpeg_cmd(w2: int, h2: int, out: Path, alpha: bool) -> list:
    base = [str(FFMPEG), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgra" if alpha else "bgr24", "-s", f"{w2}x{h2}", "-r", str(Clip.FPS), "-i", "-"]
    if alpha:
        return base + ["-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-b:v", "0", "-crf", "24", "-deadline", "realtime", "-cpu-used", "6",
                       "-auto-alt-ref", "0", str(out)]
    return base + ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "17", "-preset", "veryfast", "-movflags", "+faststart", str(out)]


def _with_alpha(frame_bgr: np.ndarray, alpha: np.ndarray) -> bytes:
    return np.ascontiguousarray(np.dstack([frame_bgr, (alpha * 255).astype(np.uint8)])).tobytes()


def build_bank(av: "Avatar", folder: Path, still_current, quiet=lambda: True, note=lambda m: None) -> None:
    """Keep inventing new waiting animations whenever nothing else is going on, and save them (one per file) so later
    sessions have them ready. Each animation is saved twice: the normal video, and a transparent twin whose cut-out of the
    person is worked out for every single frame, so a replaced background follows the head when it tilts or moves.
    A background job; the manifest lists whatever is finished."""
    import json
    import subprocess

    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / "manifest.json"
    rest = av.render()
    cv2.imwrite(str(folder / "rest.jpg"), cv2.cvtColor(rest, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    h, w = rest.shape[:2]
    w2, h2 = w - w % 2, h - h % 2
    import hashlib

    try:
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        done, stamp = old["clips"], old.get("source_sha")
    except Exception:
        done, stamp = [], None
    done = [c for c in done if (folder / c["file"]).exists()]
    # Every animation must come from the SAME photo as the rest frame (and the intro). The set records which photo it was made
    # from; clips made from any other photo are thrown away instead of being mixed in (that is what made the picture jump).
    try:
        sha = hashlib.sha1((folder.parent / "source.jpg").read_bytes()).hexdigest()[:12]
    except Exception:
        sha = ""
    rest_g = cv2.GaussianBlur(cv2.cvtColor(rest, cv2.COLOR_RGB2GRAY).astype(np.float32), (0, 0), 1.5)

    def matches_rest(c) -> bool:
        cap = cv2.VideoCapture(str(folder / c["file"]))
        ok, fr = cap.read()
        cap.release()
        if not ok:
            return False
        g = cv2.GaussianBlur(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.5)
        return g.shape == rest_g.shape and float(np.abs(g - rest_g).mean()) < 8.0

    if stamp != sha:
        keep = [c for c in done if stamp is None and matches_rest(c)]  # sets from before the stamp existed: keep only matching clips
        for c in done:
            if c not in keep:
                for nm in (c["file"], c.get("webm")):
                    if nm:
                        (folder / nm).unlink(missing_ok=True)
        print(f"bank: photo changed or unrecorded; kept {len(keep)} of {len(done)} clips", flush=True)
        done = keep
    for c in done:  # clips made before the think list grew
        c["think"] = c["kind"] in THINK_KINDS
        if c.get("webm") and not (folder / c["webm"]).exists():
            c.pop("webm")

    def purge():  # goofy faces expire
        now = time.time()
        for c in [c for c in done if c.get("expires") and c["expires"] < now]:
            (folder / c["file"]).unlink(missing_ok=True)
            if c.get("webm"):
                (folder / c["webm"]).unlink(missing_ok=True)
            done.remove(c)

    def save():
        purge()
        manifest_path.write_text(json.dumps({"rest": "rest.jpg", "source_sha": sha, "width": w2, "height": h2, "fps": Clip.FPS, "clips": done}, indent=1), encoding="utf-8")

    save()
    have_model = MATTE_MODEL.exists()

    def smoothed(prev, frame_rgb):
        a = matte_of(frame_rgb)[:h2, :w2]
        return a if prev is None else 0.65 * a + 0.35 * prev  # calm the edge flicker from frame to frame

    # first: give the clips made earlier their transparent twin
    for c in list(done):
        if not have_model or c.get("webm") or c.get("ephemeral"):
            continue
        if not still_current():
            return
        while not quiet():
            note("paused: a conversation is going on")
            time.sleep(1.0)
        note(f"adding cut-out to {c['id']}")
        try:
            cap = cv2.VideoCapture(str(folder / c["file"]))
            out = folder / (Path(c["file"]).stem + ".webm")
            pr = subprocess.Popen(_ffmpeg_cmd(w2, h2, out, True), stdin=subprocess.PIPE)
            prev, n = None, 0
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                bgr = bgr[:h2, :w2]
                prev = smoothed(prev, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                pr.stdin.write(_with_alpha(bgr, prev))
                n += 1
            cap.release()
            pr.stdin.close()
            pr.wait()
            if n:
                c["webm"] = out.name
                save()
            else:
                out.unlink(missing_ok=True)
        except Exception as exc:
            print("cut-out failed for", c["id"], exc, flush=True)

    rnd = random.Random()
    recent: list[str] = []
    kinds, weights = list(BANK_KINDS), list(BANK_KINDS.values())
    serial = max([int(c["id"].rsplit("_", 1)[-1]) for c in done if c["id"].rsplit("_", 1)[-1].isdigit()] + [0])
    while True:
        if not still_current():
            return
        while not quiet():  # a chat is going on: leave the graphics card to it
            note("paused: a conversation is going on")
            time.sleep(1.0)
            if not still_current():
                return
        permanent = sum(1 for c in done if not c.get("ephemeral"))
        goofy_now = sum(1 for c in done if c.get("ephemeral"))
        pool = [
            (k, wt) for k, wt in zip(kinds, weights)
            if k not in recent[-3:]  # never the same thing twice in a row
            and (permanent < MAX_BANK or k in GOOFY_KINDS)
            and (k not in GOOFY_KINDS or goofy_now < 3)  # only a few goofy faces exist at any time
        ]
        if not pool:
            time.sleep(30)
            continue
        kind = rnd.choices([k for k, _ in pool], [wt for _, wt in pool])[0]
        recent.append(kind)
        serial += 1
        cid = f"{kind}_{serial:03d}"
        clip = Clip(rnd, kind)
        note(f"rendering {cid}")
        out = folder / f"{cid}.mp4"
        outw = folder / f"{cid}.webm"
        pr = subprocess.Popen(_ffmpeg_cmd(w2, h2, out, False), stdin=subprocess.PIPE)
        prw = subprocess.Popen(_ffmpeg_cmd(w2, h2, outw, True), stdin=subprocess.PIPE) if have_model else None
        aborted = False
        prev = None
        for f in range(clip.frames + 1):  # the last frame is the rest frame again
            if f % 10 == 0 and not quiet():  # a chat started mid-clip: drop it, try again later
                aborted = True
                break
            frame = av.render(**clip.at(f / Clip.FPS))
            bgr = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)[:h2, :w2])
            pr.stdin.write(bgr.tobytes())
            if prw is not None:
                prev = smoothed(prev, frame)
                prw.stdin.write(_with_alpha(bgr, prev))
            time.sleep(0.01)  # leave the live face some room
        for q in (pr, prw):
            if q is not None:
                q.stdin.close()
                q.wait()
        if aborted:
            out.unlink(missing_ok=True)
            outw.unlink(missing_ok=True)
            continue
        entry = {"id": cid, "file": out.name, "kind": clip.kind, "dur": round((clip.frames + 1) / Clip.FPS, 3), "think": clip.kind in THINK_KINDS}
        if prw is not None:
            entry["webm"] = outw.name
        if clip.kind in GOOFY_KINDS:
            entry["ephemeral"] = True
            entry["expires"] = time.time() + GOOFY_LIFE_SEC
        done.append(entry)
        save()
        print(f"bank: {cid} ({len(done)} clips)", flush=True)
        note(f"idle ({len(done)} clips)")


class Idle:
    """What a person does while waiting and listening: breathe, blink, drift a little, now and then glance away."""

    def __init__(self):
        self.t0 = time.time()
        rnd = random.Random()
        self.s = [rnd.uniform(0, 100) for _ in range(8)]
        self.next_blink = self.t0 + rnd.uniform(1.0, 3.0)
        self.blink_start = None
        self.next_glance = self.t0 + rnd.uniform(6.0, 11.0)
        self.glance = None  # (start, duration, yaw, pitch)
        self.next_nod = self.t0 + rnd.uniform(9.0, 16.0)
        self.nod_start = None
        self.next_yawn = self.t0 + rnd.uniform(45.0, 90.0)
        self.yawn_start = None
        self.rnd = rnd
        self.gesture = None
        self.next_gesture = self.t0 + rnd.uniform(5.0, 9.0)
        self.settle_until = 0.0
        self.settle = False  # True while a reply is being prepared: stay quiet and neutral so the video fades in cleanly
        self.amp = 1.0
        self.last = self.t0

    def trigger_yawn(self):
        self.yawn_start = time.time()

    def state(self, now: float) -> dict:
        t = now - self.t0
        s = self.s
        if self.settle and now > self.settle_until:
            self.settle = False
        dt = max(0.0, now - self.last)
        self.last = now
        self.amp += ((0.0 if self.settle else 1.0) - self.amp) * min(1.0, dt / 0.5)  # ease into / out of the quiet mode
        a = self.amp
        yaw = a * (1.6 * math.sin(0.31 * t + s[0]) + 0.8 * math.sin(0.77 * t + s[1]))
        pitch = a * (0.8 * math.sin(0.37 * t + s[2]) + 0.4 * math.sin(1.07 * t + s[3]))
        roll = a * 0.6 * math.sin(0.23 * t + s[4])
        dy = 0.0035 * math.sin(2 * math.pi * t / 4.3 + s[5])  # breathing never stops
        dx = a * 0.002 * math.sin(0.19 * t + s[6])

        # blinks: every 2-5.5 s, sometimes doubled
        blink = 0.0
        if self.blink_start is None and not self.settle and now >= self.next_blink:
            self.blink_start = now
        if self.blink_start is not None:
            p = (now - self.blink_start) / 0.19
            if p >= 1:
                self.blink_start = None
                self.next_blink = now + (0.28 if self.rnd.random() < 0.18 else self.rnd.uniform(2.0, 5.5))
            else:
                blink = math.sin(math.pi * p)

        # an occasional glance to one side and back
        if self.glance is None and not self.settle and now >= self.next_glance:
            self.glance = (now, self.rnd.uniform(0.9, 1.5), self.rnd.choice([-1, 1]) * self.rnd.uniform(3.0, 5.0), self.rnd.uniform(-1.2, 0.8))
        if self.glance:
            g0, gd, gy, gp = self.glance
            x = (now - g0) / gd
            if x >= 1:
                self.glance = None
                self.next_glance = now + self.rnd.uniform(7.0, 14.0)
            else:
                yaw += gy * bump(x)
                pitch += gp * bump(x)

        # an occasional small nod
        if self.nod_start is None and not self.settle and now >= self.next_nod:
            self.nod_start = now
        if self.nod_start is not None:
            x = (now - self.nod_start) / 0.9
            if x >= 1:
                self.nod_start = None
                self.next_nod = now + self.rnd.uniform(10.0, 18.0)
            else:
                pitch += 1.6 * bump(x)
        # an occasional yawn: head tips back, mouth opens wide, eyes screw shut, then everything relaxes
        mouth = 0.0
        if self.yawn_start is None and not self.settle and now >= self.next_yawn:
            self.yawn_start = now
        if self.yawn_start is not None:
            x = (now - self.yawn_start) / 2.8
            if x >= 1:
                self.yawn_start = None
                self.next_yawn = now + self.rnd.uniform(60.0, 140.0)
            else:
                open_ = bump(min(1.0, x / 0.82)) if x < 0.82 else 0.0  # rises, holds, closes
                hold = 1.0 if 0.22 < x < 0.6 else (smooth(x / 0.22) if x <= 0.22 else smooth((0.82 - x) / 0.22))
                mouth = 0.28 * max(0.0, hold)
                blink = max(blink, 0.65 * max(0.0, hold))
                pitch += 3.0 * max(0.0, hold)
                roll += 0.8 * max(0.0, hold)
        # a newly invented gesture every so often, so the waiting never looks like a loop
        if self.gesture is None and not self.settle and self.yawn_start is None and now >= self.next_gesture:
            self.gesture = Gesture(self.rnd, now)
        if self.gesture is not None:
            g = self.gesture.at(now)
            if g is None or self.settle:
                self.gesture = None
                self.next_gesture = now + self.rnd.uniform(6.0, 14.0)
            else:
                yaw += g[0]
                pitch += g[1]
                roll += g[2]
                blink = max(blink, g[3])
                mouth = max(mouth, g[4])
        return dict(yaw=yaw, pitch=pitch, roll=roll, dx=dx, dy=dy, blink=blink, mouth=mouth)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--user", default="")
    ap.add_argument("--fps", type=float, default=float(os.environ.get("IDLE_FPS", "20")))
    a = ap.parse_args()

    inf = InferenceConfig()
    inf.flag_use_half_precision = True
    pipe = LivePortraitPipeline(inference_cfg=inf, crop_cfg=CropConfig())
    state = {"avatar": None, "user": "", "n": 0, "jpg": b"", "times": [], "seed": None, "seed_t": 0.0, "lib": {}, "live_on": True, "busy_until": 0.0, "bank_msg": ""}
    cond = threading.Condition()
    idle = Idle()

    def load(user: str) -> float:
        t = time.time()
        photo = ROOT / "data" / "media" / user / "cinematic" / "source.jpg"
        state["avatar"] = Avatar(photo, pipe)
        av = state["avatar"]
        threading.Thread(
            target=lambda: build_bank(av, photo.parent / "idle_bank", lambda: state["avatar"] is av, lambda: time.time() > state["busy_until"], lambda m: state.__setitem__("bank_msg", m)),
            daemon=True,
        ).start()
        try:
            make_matte(photo, photo.with_name("matte.png"))
        except Exception as e:  # the avatar works fine without a cut-out
            print("matte failed:", e, flush=True)
        state["user"] = user
        return time.time() - t

    if a.user:
        print(f"loaded {a.user} in {load(a.user):.1f}s", flush=True)

    def loop():
        dt = 1.0 / a.fps
        nxt = time.time()
        while True:
            av = state["avatar"]
            if av is None or not state["live_on"]:  # the pre-made animations are on screen: leave the GPU alone
                time.sleep(0.1)
                nxt = time.time()
                continue
            now = time.time()
            try:
                w = math.exp(-(now - state["seed_t"]) / 1.4) if state["seed"] is not None else 0.0  # fades out over a few seconds
                rgb = av.render(**idle.state(now), seed=state["seed"], w=w)
                ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
            except Exception as e:  # a bad frame must not kill the loop
                print("render error:", e, flush=True)
                time.sleep(0.2)
                continue
            with cond:
                state["jpg"] = buf.tobytes()
                state["n"] += 1
                state["times"] = (state["times"] + [time.time()])[-40:]
                cond.notify_all()
            nxt += dt
            time.sleep(max(0.0, nxt - time.time()))
            if time.time() - nxt > 1.0:  # fell behind: do not try to catch up
                nxt = time.time()

    threading.Thread(target=loop, daemon=True).start()
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:8765", "http://127.0.0.1:8765"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health():
        t = state["times"]
        fps = (len(t) - 1) / (t[-1] - t[0]) if len(t) > 2 and t[-1] > t[0] else 0.0
        return {"ok": True, "user": state["user"], "fps": round(fps, 1), "frames": state["n"], "live": state["live_on"], "bank": state["bank_msg"], "busy_for": round(max(0.0, state["busy_until"] - time.time()), 1), "ready": state["avatar"] is not None}

    @app.post("/seed")
    async def seed(request: Request):
        """Body = a JPEG of the frame that is on screen right now. The live face continues from that pose and expression."""
        data = await request.body()
        av = state["avatar"]
        if av is None:
            return {"ok": False}
        bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            return {"ok": False}
        m = av.motion_of(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if m is None:
            return {"ok": False, "why": "no face"}
        state["seed"], state["seed_t"] = m, time.time()
        idle.settle = False
        return {"ok": True}

    @app.post("/settle")
    def settle(on: int = 1, keep: int = 0):
        """on=1: go quiet and neutral (a reply is being prepared); on=0: resume normal idling.
        keep=1: calm down but keep easing out of the pose just inherited from a video (a smooth hand-back, no snap)."""
        idle.settle = bool(on)
        idle.settle_until = time.time() + 25.0
        if on and not keep:
            state["seed"] = None
        return {"ok": True, "settle": idle.settle}

    @app.post("/trigger")
    def trigger(what: str = "yawn"):
        if what == "yawn":
            idle.trigger_yawn()
        return {"ok": True, "what": what}

    @app.post("/use")
    def use(user: str):
        if user == state["user"] and state["avatar"] is not None:
            return {"ok": True, "user": user, "cached": True}
        try:
            return {"ok": True, "user": user, "seconds": round(load(user), 1)}
        except Exception as e:
            return JSONResponse({"ok": False, "error": str(e)[:200]}, status_code=404)

    @app.get("/expr/recipes")
    def expr_recipes(px: int = 260, cols: int = 4, w: float = 1.0):
        """Every named emotion applied to the neutral face, side by side (to tune the recipes by eye)."""
        av = state["avatar"]
        pts = np.asarray(av.crop_info["lmk_crop"], np.float32)
        x0, y0 = pts.min(0) - 25
        x1, y1 = pts.max(0) + 25
        x0, y0 = int(max(0, x0)), int(max(0, y0))
        x1, y1 = int(min(av.src_rgb.shape[1], x1)), int(min(av.src_rgb.shape[0], y1))
        names = ["neutral"] + list(emotion_table(av.w.device).keys())
        tiles = []
        for n in names:
            img = av.render(emotion=n, emo_w=w)[y0:y1, x0:x1]
            img = cv2.cvtColor(cv2.resize(img, (px, px)), cv2.COLOR_RGB2BGR)
            cv2.putText(img, n, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            tiles.append(img)
        while len(tiles) % cols:
            tiles.append(np.zeros_like(tiles[0]))
        ok, buf = cv2.imencode(".jpg", np.vstack([np.hstack(tiles[i : i + cols]) for i in range(0, len(tiles), cols)]), [cv2.IMWRITE_JPEG_QUALITY, 88])
        return Response(buf.tobytes(), media_type="image/jpeg")

    @app.get("/expr/probe")
    def expr_probe(amt: float = 0.05, px: int = 150, axes: str = "xy", lo: int = 0, hi: int = 21):
        """Contact sheet: each of the 21 face control points nudged along each axis, both directions (face area only).
        Used to learn what every control does so expressions can be composed from them."""
        av = state["avatar"]
        lm = av.crop_info["lmk_crop"]
        pts = np.asarray(lm, np.float32)  # already in the photo's own coordinates
        x0, y0 = pts.min(0) - 20
        x1, y1 = pts.max(0) + 20
        x0, y0 = int(max(0, x0)), int(max(0, y0))
        x1, y1 = int(min(av.src_rgb.shape[1], x1)), int(min(av.src_rgb.shape[0], y1))
        rows = []
        for kp in range(lo, hi):
            tiles = []
            for ax in axes:
                for sign in (-1, 1):
                    e = torch.zeros_like(av.s_info["exp"])
                    e[0, kp, "xyz".index(ax)] = sign * amt
                    img = av.render(exp=e, exp_w=1.0)[y0:y1, x0:x1]
                    img = cv2.cvtColor(cv2.resize(img, (px, px)), cv2.COLOR_RGB2BGR)
                    cv2.putText(img, f"k{kp}{ax}{'+' if sign > 0 else '-'}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
                    tiles.append(img)
            rows.append(np.hstack(tiles))
        ok, buf = cv2.imencode(".jpg", np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
        return Response(buf.tobytes(), media_type="image/jpeg")

    @app.post("/busy")
    def busy(sec: float = 45.0):
        """The page says a conversation is going on: pause inventing new animations for a while."""
        state["busy_until"] = time.time() + min(max(sec, 1.0), 12.0)  # a short pause per report: real chats report every few seconds
        return {"ok": True}

    @app.post("/live")
    def live(on: int = 1):
        """on=0: stop rendering the live face (the page is playing pre-made animations); on=1: resume."""
        state["live_on"] = bool(on)
        return {"ok": True, "live": state["live_on"]}

    @app.get("/expr/sheet")
    def expr_sheet(cols: int = 8, px: int = 200):
        """Contact sheet of every expression in the library applied to the neutral face (for choosing which ones to use)."""
        av, lib = state["avatar"], state["lib"]
        tiles = []
        for key, e in lib.items():
            with torch.no_grad():
                img = av.render(exp=e, exp_w=1.0)
            img = cv2.resize(img, (px, int(px * img.shape[0] / img.shape[1])))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.putText(img, key.replace("reaction_", "r_").replace("greeting_", "g_").replace("thinking_", "t_"), (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
            tiles.append(img)
        while len(tiles) % cols:
            tiles.append(np.zeros_like(tiles[0]))
        rows = [np.hstack(tiles[i : i + cols]) for i in range(0, len(tiles), cols)]
        ok, buf = cv2.imencode(".jpg", np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
        return Response(buf.tobytes(), media_type="image/jpeg")

    @app.get("/frame.jpg")
    def frame(user: str = ""):
        """The newest frame as a plain image (browsers handle this more reliably than a live stream).
        If the caller names a person who is not the one currently loaded, no picture is sent (never show someone else's face)."""
        if user and user != state["user"]:
            return Response(status_code=409)
        return Response(state["jpg"], media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/stream")
    def stream():
        def gen():
            seen = 0
            while True:
                with cond:
                    cond.wait_for(lambda: state["n"] != seen, timeout=2)
                    seen = state["n"]
                    jpg = state["jpg"]
                if jpg:
                    yield b"--f\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n"

        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=f")

    import uvicorn

    print(f"Idle avatar on http://127.0.0.1:{a.port}/stream", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
