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
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse

from src.config.crop_config import CropConfig
from src.config.inference_config import InferenceConfig
from src.live_portrait_pipeline import LivePortraitPipeline
from src.utils.crop import paste_back, prepare_paste_back
from src.utils.retargeting_utils import calc_eye_close_ratio

OUT_PX = int(os.environ.get("IDLE_PX", "720"))  # longest side of the streamed picture


class Avatar:
    """One person's photo prepared for animation. The (heavy) LivePortrait models are shared between avatars."""

    def __init__(self, photo: Path, pipe: LivePortraitPipeline):
        self.pipe = pipe
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

    @torch.no_grad()
    def render(self, yaw=0.0, pitch=0.0, roll=0.0, dx=0.0, dy=0.0, blink=0.0) -> np.ndarray:
        """blink 0..1 (1 = closed); angles in degrees. Returns an RGB picture the size of the source photo."""
        info = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in self.s_info.items()}
        info["yaw"] = info["yaw"] + yaw
        info["pitch"] = info["pitch"] + pitch
        info["roll"] = info["roll"] + roll
        info["t"][:, 0] += dx
        info["t"][:, 1] += dy
        x_d = self.w.transform_keypoint(info)
        if blink > 0.01:
            eye_t = torch.tensor([[self.eye0 * (1.0 - blink)]], device=self.w.device)
            x_d = x_d + self.w.retarget_eye(self.x_s, torch.cat([self.c_s_e, eye_t], dim=1)) - self.eye_open_delta
        x_d = self.w.stitching(self.x_s, x_d)
        out = self.w.warp_decode(self.f_s, self.x_s, x_d)
        crop = self.w.parse_output(out["out"])[0]
        return paste_back(crop, self.crop_info["M_c2o"], self.src_rgb, self.mask)


def bump(x: float) -> float:
    """Smooth 0 -> 1 -> 0 over x in [0, 1]."""
    return math.sin(math.pi * min(max(x, 0.0), 1.0)) ** 2


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
        self.rnd = rnd

    def state(self, now: float) -> dict:
        t = now - self.t0
        s = self.s
        yaw = 1.6 * math.sin(0.31 * t + s[0]) + 0.8 * math.sin(0.77 * t + s[1])
        pitch = 0.8 * math.sin(0.37 * t + s[2]) + 0.4 * math.sin(1.07 * t + s[3])
        roll = 0.6 * math.sin(0.23 * t + s[4])
        dy = 0.0035 * math.sin(2 * math.pi * t / 4.3 + s[5])  # breathing
        dx = 0.002 * math.sin(0.19 * t + s[6])

        # blinks: every 2-5.5 s, sometimes doubled
        blink = 0.0
        if self.blink_start is None and now >= self.next_blink:
            self.blink_start = now
        if self.blink_start is not None:
            p = (now - self.blink_start) / 0.19
            if p >= 1:
                self.blink_start = None
                self.next_blink = now + (0.28 if self.rnd.random() < 0.18 else self.rnd.uniform(2.0, 5.5))
            else:
                blink = math.sin(math.pi * p)

        # an occasional glance to one side and back
        if self.glance is None and now >= self.next_glance:
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
        if self.nod_start is None and now >= self.next_nod:
            self.nod_start = now
        if self.nod_start is not None:
            x = (now - self.nod_start) / 0.9
            if x >= 1:
                self.nod_start = None
                self.next_nod = now + self.rnd.uniform(10.0, 18.0)
            else:
                pitch += 1.6 * bump(x)
        return dict(yaw=yaw, pitch=pitch, roll=roll, dx=dx, dy=dy, blink=blink)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--user", default="")
    ap.add_argument("--fps", type=float, default=float(os.environ.get("IDLE_FPS", "20")))
    a = ap.parse_args()

    inf = InferenceConfig()
    inf.flag_use_half_precision = True
    pipe = LivePortraitPipeline(inference_cfg=inf, crop_cfg=CropConfig())
    state = {"avatar": None, "user": "", "n": 0, "jpg": b"", "times": []}
    cond = threading.Condition()
    idle = Idle()

    def load(user: str) -> float:
        t = time.time()
        photo = ROOT / "data" / "media" / user / "cinematic" / "source.jpg"
        state["avatar"] = Avatar(photo, pipe)
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
                rgb = av.render(**idle.state(now))
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

    @app.get("/health")
    def health():
        t = state["times"]
        fps = (len(t) - 1) / (t[-1] - t[0]) if len(t) > 2 and t[-1] > t[0] else 0.0
        return {"ok": True, "user": state["user"], "fps": round(fps, 1), "frames": state["n"]}

    @app.post("/use")
    def use(user: str):
        if user == state["user"] and state["avatar"] is not None:
            return {"ok": True, "user": user, "cached": True}
        try:
            return {"ok": True, "user": user, "seconds": round(load(user), 1)}
        except Exception as e:
            return JSONResponse({"ok": False, "error": str(e)[:200]}, status_code=404)

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
