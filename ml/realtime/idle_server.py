"""Live idle layer: the person's photo, alive in real time (blinks, breathing, slow head drift, an occasional glance).
Runs locally (LivePortrait); nothing leaves this machine. The page shows this stream whenever no clip or reply video is playing.

    ml\\realtime\\lpenv\\Scripts\\python.exe ml\\realtime\\idle_server.py [--port 8770] [--user jeff_molofee] [--fps 20]

    GET  /stream             MJPEG video of the idle avatar
    POST /use?user=<slug>    switch to data/media/<slug>/cinematic/source.jpg (the same wide photo the clips use)
    GET  /health
"""
from __future__ import annotations

import argparse
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
    def _render(self, yaw=0.0, pitch=0.0, roll=0.0, dx=0.0, dy=0.0, blink=0.0, mouth=0.0, seed=None, w=0.0, exp=None, exp_w=0.0) -> np.ndarray:
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
        if exp is not None and exp_w > 0.01:  # a facial expression (smile, raised brows, ...) laid over the neutral face
            info["exp"] = info["exp"] + exp_w * exp
        x_d = self.w.transform_keypoint(info)
        blink = REST_BLINK + max(blink, 0.0) * (1.0 - REST_BLINK)
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


REST_BLINK = -0.22  # < 0 = eyes slightly wider than in the photo; every animation is relative to this rest face

MATTE_MODEL = HERE / "models" / "u2net_human_seg.onnx"


def make_matte(photo: Path, out: Path) -> bool:
    """Cut the person out of the prepared photo: writes a white PNG whose alpha is the person (hair included).
    Runs a small segmentation model locally (onnxruntime); the page uses it to swap the background."""
    if out.exists() and out.stat().st_mtime >= photo.stat().st_mtime:
        return True
    if not MATTE_MODEL.exists():
        return False
    import onnxruntime as ort

    img = cv2.imread(str(photo))
    h, w = img.shape[:2]
    sess = ort.InferenceSession(str(MATTE_MODEL), providers=["CPUExecutionProvider"])
    x = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (320, 320), interpolation=cv2.INTER_AREA).astype(np.float32)
    x = x / max(float(x.max()), 1e-6)
    x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    y = sess.run(None, {sess.get_inputs()[0].name: x.transpose(2, 0, 1)[None].astype(np.float32)})[0][0, 0]
    y = (y - y.min()) / max(float(y.max() - y.min()), 1e-6)
    m = cv2.resize(y, (w, h), interpolation=cv2.INTER_CUBIC)
    m = np.clip((m - 0.5) / 0.3, 0.0, 1.0)  # firm edge, soft transition
    k = max(3, w // 160) | 1
    m = cv2.erode(m, np.ones((k, k), np.uint8))  # pull the edge in a little so no old background shows as a halo
    m = cv2.GaussianBlur(m, (0, 0), max(1.2, w / 350))
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


class Clip:
    """One pre-made waiting animation. Every clip starts and ends exactly on the rest frame (all offsets are zero there),
    so any clip can follow any other clip, or a pause on the rest frame, with no visible join."""

    FPS = 25

    def __init__(self, rnd: random.Random, kind: str):
        self.kind = kind
        self.g = None
        self.blinks = []
        if kind == "breathe":
            self.dur = rnd.uniform(1.8, 2.8)
            self.amp = [rnd.uniform(-1.0, 1.0) for _ in range(3)]
        elif kind == "yawn":
            self.dur = 2.8
        else:
            self.g = Gesture(rnd, 0.0, kind)
            self.dur = self.g.dur
        nb = 1 if kind in ("breathe", "yawn") else rnd.choice([0, 1, 1])
        if kind != "yawn":
            for _ in range(nb):
                self.blinks.append(rnd.uniform(0.35, max(0.4, self.dur - 0.7)))
        self.frames = int(round(self.dur * self.FPS))
        self.dur = self.frames / self.FPS

    def at(self, t: float) -> dict:
        D = self.dur
        e = bump(t / D)
        p = dict(yaw=0.0, pitch=0.0, roll=0.0, dx=0.0, dy=0.0035 * math.sin(2 * math.pi * t / D), blink=0.0, mouth=0.0)
        if self.kind == "breathe":
            p["yaw"], p["pitch"], p["roll"] = self.amp[0] * e, 0.5 * self.amp[1] * e, 0.3 * self.amp[2] * e
        elif self.kind == "yawn":
            x = t / D
            hold = 1.0 if 0.22 < x < 0.6 else (smooth(x / 0.22) if x <= 0.22 else smooth((0.82 - x) / 0.22))
            hold = max(0.0, hold)
            p.update(mouth=0.85 * hold, blink=0.65 * hold, pitch=3.0 * hold, roll=0.8 * hold)
        else:
            g = self.g.at(self.g.t0 + t)
            if g is not None:
                p.update(yaw=g[0], pitch=g[1], roll=g[2], blink=max(0.0, g[3]), mouth=g[4])
                if g[3] < 0:
                    p["blink"] = g[3]  # eyes wider than at rest
        for b in self.blinks:
            x = (t - b) / 0.19
            if 0 <= x <= 1:
                p["blink"] = max(p["blink"], math.sin(math.pi * x))
        return p


BANK_PLAN = [("breathe", 14), ("look_around", 3), ("think", 4), ("tilt", 3), ("sigh", 2), ("double_take", 2), ("ponder", 3),
             ("curious", 3), ("perk", 3), ("yawn", 2)]
THINK_KINDS = ("think", "ponder", "curious", "tilt")
FFMPEG = HERE / "ffmpeg" / "ffmpeg-master-latest-win64-gpl" / "bin" / "ffmpeg.exe"


def build_bank(av: "Avatar", folder: Path, still_current) -> None:
    """Render the waiting animations once per person and keep them on disk, so later sessions need no rendering at all.
    A background job: it adds a few clips at a time, and the manifest lists whatever is finished."""
    import json
    import subprocess

    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / "manifest.json"
    rest = av.render()
    cv2.imwrite(str(folder / "rest.jpg"), cv2.cvtColor(rest, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    h, w = rest.shape[:2]
    w2, h2 = w - w % 2, h - h % 2
    try:
        done = json.loads(manifest_path.read_text(encoding="utf-8"))["clips"]
    except Exception:
        done = []
    have = {c["id"] for c in done if (folder / c["file"]).exists()}
    done = [c for c in done if c["id"] in have]

    def save():
        manifest_path.write_text(json.dumps({"rest": "rest.jpg", "width": w2, "height": h2, "fps": Clip.FPS, "clips": done}, indent=1), encoding="utf-8")

    save()
    plan = []
    for i in range(max(n for _, n in BANK_PLAN)):  # interleave kinds so a useful mix exists early
        for kind, n in BANK_PLAN:
            if i < n:
                plan.append((kind, i))
    rnd = random.Random()
    for kind, i in plan:
        cid = f"{kind}_{i:02d}"
        if cid in have:
            continue
        if not still_current():
            return
        clip = Clip(rnd, kind)
        out = folder / f"{cid}.mp4"
        cmd = [str(FFMPEG), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w2}x{h2}", "-r", str(Clip.FPS), "-i", "-",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "17", "-preset", "veryfast", "-movflags", "+faststart", str(out)]
        pr = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for f in range(clip.frames + 1):  # +1: the closing frame is the rest frame again
            frame = av.render(**clip.at(min(f / Clip.FPS, clip.dur) if f < clip.frames else 0.0 if False else clip.dur))
            pr.stdin.write(np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)[:h2, :w2]).tobytes())
            time.sleep(0.01)  # leave the live face some room
        pr.stdin.close()
        pr.wait()
        done.append({"id": cid, "file": out.name, "kind": kind, "dur": round((clip.frames + 1) / Clip.FPS, 3), "think": kind in THINK_KINDS})
        save()
        print(f"bank: {cid} ({len(done)} clips)", flush=True)


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
                mouth = 0.85 * max(0.0, hold)
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
    state = {"avatar": None, "user": "", "n": 0, "jpg": b"", "times": [], "seed": None, "seed_t": 0.0, "lib": {}}
    cond = threading.Condition()
    idle = Idle()

    def load(user: str) -> float:
        t = time.time()
        photo = ROOT / "data" / "media" / user / "cinematic" / "source.jpg"
        state["avatar"] = Avatar(photo, pipe)
        av = state["avatar"]
        threading.Thread(
            target=lambda: build_bank(av, photo.parent / "idle_bank", lambda: state["avatar"] is av),
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
            if av is None:
                time.sleep(0.1)
                continue
            now = time.time()
            try:
                w = math.exp(-(now - state["seed_t"]) / 2.2) if state["seed"] is not None else 0.0  # fades out over a few seconds
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
        return {"ok": True, "user": state["user"], "fps": round(fps, 1), "frames": state["n"]}

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
    def settle(on: int = 1):
        """on=1: go quiet and neutral (a reply is being prepared); on=0: resume normal idling."""
        idle.settle = bool(on)
        idle.settle_until = time.time() + 25.0
        if on:
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
    def frame():
        """The newest frame as a plain image (browsers handle this more reliably than a live stream)."""
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
