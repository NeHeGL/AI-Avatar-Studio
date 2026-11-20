"""Local real-time avatar engine (LivePortrait). Nothing leaves this machine.

Run with the realtime venv:
    lpenv\\Scripts\\python.exe live_engine.py --photo <wide photo.jpg> [--port 8770]

Serves:
    GET  /          small demo page (video + "Speak test" button)
    GET  /stream    MJPEG video of the avatar (idle blinking / head sway, mouth follows speech)
    POST /speak     body = audio file (wav/ogg/flac); starts the mouth following that audio right now
    GET  /sample.wav  a short voice sample for the demo button
    GET  /health
    --selftest      writes a contact sheet of mouth / head poses and exits (for tuning)
"""
from __future__ import annotations

import argparse
import io
import math
import os
import random
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LP = HERE / "LivePortrait"
ORIG_CWD = Path.cwd()  # relative paths on the command line are relative to where the user ran this
sys.path.insert(0, str(LP))
os.chdir(LP)  # LivePortrait resolves its weights relative to its own folder

import cv2
import numpy as np
import soundfile as sf
import torch
from fastapi import Request  # module level: FastAPI must resolve this annotation by name

from src.config.argument_config import ArgumentConfig
from src.config.crop_config import CropConfig
from src.config.inference_config import InferenceConfig
from src.live_portrait_pipeline import LivePortraitPipeline
from src.utils.camera import get_rotation_matrix  # noqa: F401  (kept for tuning experiments)
from src.utils.crop import paste_back, prepare_paste_back
from src.utils.retargeting_utils import calc_eye_close_ratio, calc_lip_close_ratio

OUT_PX = 720  # size of the streamed picture


class Avatar:
    def __init__(self, photo: Path):
        inf = InferenceConfig()
        inf.flag_use_half_precision = True
        crop = CropConfig()
        self.pipe = LivePortraitPipeline(inference_cfg=inf, crop_cfg=crop)
        self.w = self.pipe.live_portrait_wrapper
        self.inf = inf

        bgr = cv2.imread(str(photo))
        if bgr is None:
            raise FileNotFoundError(photo)
        h, wd = bgr.shape[:2]
        k = OUT_PX / max(h, wd)
        if k < 1:
            bgr = cv2.resize(bgr, (round(wd * k), round(h * k)), interpolation=cv2.INTER_AREA)
        self.src_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        info = self.pipe.cropper.crop_source_image(self.src_rgb, crop)
        if info is None:
            raise RuntimeError("No face found in the photo")
        self.crop_info = info
        self.src_lmk = info["lmk_crop"]
        self.mask = prepare_paste_back(inf.mask_crop, info["M_c2o"], dsize=(self.src_rgb.shape[1], self.src_rgb.shape[0]))

        I_s = self.w.prepare_source(info["img_crop_256x256"])
        self.s_info = self.w.get_kp_info(I_s)
        self.f_s = self.w.extract_feature_3d(I_s)
        self.x_s = self.w.transform_keypoint(self.s_info)
        self.eye0 = float(calc_eye_close_ratio(self.src_lmk[None]).mean())  # eyes-open ratio of this person
        self.lip0 = float(calc_lip_close_ratio(self.src_lmk[None]).mean())  # mouth ratio at rest

    @torch.no_grad()
    def render(self, yaw=0.0, pitch=0.0, roll=0.0, dx=0.0, dy=0.0, blink=0.0, mouth=0.0) -> np.ndarray:
        """blink 0..1 (1 = closed), mouth 0..1 (1 = wide open); angles in degrees. Returns an RGB picture."""
        info = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in self.s_info.items()}
        info["yaw"] = info["yaw"] + yaw
        info["pitch"] = info["pitch"] + pitch
        info["roll"] = info["roll"] + roll
        info["t"][:, 0] += dx
        info["t"][:, 1] += dy
        x_d = self.w.transform_keypoint(info)

        dev = self.w.device
        eye_t = torch.tensor([[self.eye0 * (1.0 - blink)]], device=dev)
        c_s_e = torch.from_numpy(calc_eye_close_ratio(self.src_lmk[None])).float().to(dev)
        x_d = x_d + self.w.retarget_eye(self.x_s, torch.cat([c_s_e, eye_t], dim=1)) - self.w.retarget_eye(
            self.x_s, torch.cat([c_s_e, c_s_e.mean(dim=1, keepdim=True)], dim=1)
        )  # only the *change* from the open-eye pose

        lip_t = torch.tensor([[self.lip0 + 0.55 * mouth]], device=dev)
        c_s_l = torch.from_numpy(calc_lip_close_ratio(self.src_lmk[None])).float().to(dev)
        x_d = x_d + self.w.retarget_lip(self.x_s, torch.cat([c_s_l, lip_t], dim=1)) - self.w.retarget_lip(
            self.x_s, torch.cat([c_s_l, c_s_l], dim=1)
        )
        x_d = self.w.stitching(self.x_s, x_d)
        out = self.w.warp_decode(self.f_s, self.x_s, x_d)
        crop = self.w.parse_output(out["out"])[0]
        return paste_back(crop, self.crop_info["M_c2o"], self.src_rgb, self.mask)


class Director:
    """Decides what the avatar does each frame: idle sway, blinks, mouth from the speech audio."""

    def __init__(self):
        self.t0 = time.time()
        self.next_blink = self.t0 + random.uniform(1.5, 4.0)
        self.blink_start = None
        self.speech = None  # (start_time, envelope array at ENV_HZ)
        self.lock = threading.Lock()
        self.seed = [random.uniform(0, 100) for _ in range(6)]

    ENV_HZ = 100

    def speak(self, wav: np.ndarray, sr: int):
        mono = wav.mean(axis=1) if wav.ndim > 1 else wav
        hop = sr // self.ENV_HZ
        n = len(mono) // hop
        rms = np.sqrt((mono[: n * hop].reshape(n, hop) ** 2).mean(axis=1) + 1e-12)
        ref = np.percentile(rms, 92) + 1e-6
        env = np.clip(rms / ref, 0, 1.2)
        env[env < 0.12] = 0  # quiet = closed mouth
        # smooth a little so the mouth doesn't flicker
        k = np.ones(3) / 3
        env = np.convolve(env, k, mode="same")
        with self.lock:
            self.speech = (time.time(), env)

    def state(self, now: float) -> dict:
        t = now - self.t0
        s = self.seed
        yaw = 3.5 * math.sin(t * 0.37 + s[0]) + 1.8 * math.sin(t * 0.91 + s[1])
        pitch = 1.8 * math.sin(t * 0.43 + s[2]) + 0.8 * math.sin(t * 1.3 + s[3])
        roll = 1.0 * math.sin(t * 0.29 + s[4])
        dy = 0.004 * math.sin(t * 0.25 + s[5])  # slow breathing
        blink = 0.0
        if self.blink_start is None and now >= self.next_blink:
            self.blink_start = now
        if self.blink_start is not None:
            p = (now - self.blink_start) / 0.18
            if p >= 1:
                self.blink_start = None
                self.next_blink = now + random.uniform(2.0, 5.5)
                if random.random() < 0.2:
                    self.next_blink = now + 0.25  # occasional double blink
            else:
                blink = math.sin(p * math.pi)
        mouth = 0.0
        with self.lock:
            if self.speech:
                st, env = self.speech
                i = int((now - st) * self.ENV_HZ)
                if 0 <= i < len(env):
                    mouth = float(env[i])
                    yaw += 1.5 * math.sin(t * 2.1)  # a little extra motion while talking
                    pitch += 1.2 * mouth
                elif i >= len(env):
                    self.speech = None
        return dict(yaw=yaw, pitch=pitch, roll=roll, dy=dy, blink=blink, mouth=min(1.0, mouth))


RECORD_HTML = """<!doctype html><meta charset=utf-8><title>Record a talking sample</title>
<body style="background:#0b1020;color:#ddd;font-family:sans-serif;max-width:820px;margin:20px auto;padding:0 16px">
<h2>Record a talking sample</h2>
<p>This teaches the avatar how <b>your</b> mouth moves for each sound, and how you move while you talk.
<b>Lean in so your face fills about half the picture height</b>, with good light on your face. Read the text below clearly, with <b>slightly bigger mouth movements than normal</b> (like talking to someone across a noisy room), but still in your own natural voice and rhythm. Move your head, smile, raise your eyebrows if that's natural for you.
About 2 minutes is plenty; more is better. Everything stays on this computer.</p>
<video id=pv autoplay muted playsinline style="width:100%;max-width:480px;border-radius:10px;background:#000;transform:scaleX(-1)"></video><br>
<button id=rec style="font-size:18px;padding:10px 22px">Start recording</button>
<span id=st style="margin-left:12px;opacity:.8"></span>
<div id=txt style="background:#141b33;padding:14px 18px;border-radius:10px;margin-top:14px;line-height:1.6;font-size:18px">
<p><b>Read aloud:</b></p>
<p>When the sunlight strikes raindrops in the air, they act as a prism and form a rainbow. The rainbow is a division of white light into many beautiful colors. These take the shape of a long round arch, with its path high above, and its two ends apparently beyond the horizon. There is, according to legend, a boiling pot of gold at one end. People look, but no one ever finds it. When a man looks for something beyond his reach, his friends say he is looking for the pot of gold at the end of the rainbow.</p>
<p>The birch canoe slid on the smooth planks. Glue the sheet to the dark blue background. It's easy to tell the depth of a well. These days a chicken leg is a rare dish. Rice is often served in round bowls. The juice of lemons makes fine punch. The box was thrown beside the parked truck. The hogs were fed chopped corn and garbage. Four hours of steady work faced us. A large size in stockings is hard to sell.</p>
<p><b>Now just talk for 30 seconds</b> about anything: your day, what you had for breakfast, something that annoyed you recently, a happy memory. Be yourself.</p>
</div>
<script>
let rec,chunks=[],stream,t0;
const st=document.getElementById('st'),pv=document.getElementById('pv'),btn=document.getElementById('rec');
navigator.mediaDevices.getUserMedia({video:{width:{ideal:1920},height:{ideal:1080},frameRate:{ideal:30}},audio:{echoCancellation:false,noiseSuppression:false,autoGainControl:false}})
 .then(s=>{stream=s;pv.srcObject=s;const t=s.getVideoTracks()[0].getSettings();st.textContent='Camera: '+t.width+'x'+t.height+' @ '+Math.round(t.frameRate||0)+' fps'+(t.width<1280?' (low - a better camera setting or USB port helps)':'')}).catch(e=>st.textContent='Camera/mic blocked: '+e.message);
btn.onclick=()=>{
 if(rec&&rec.state==='recording'){rec.stop();return}
 chunks=[];
 const mt=MediaRecorder.isTypeSupported('video/webm;codecs=vp9,opus')?'video/webm;codecs=vp9,opus':'video/webm';
 rec=new MediaRecorder(stream,{mimeType:mt,videoBitsPerSecond:8e6});
 rec.ondataavailable=e=>e.data.size&&chunks.push(e.data);
 rec.onstop=async()=>{
  clearInterval(tm);btn.disabled=true;st.textContent='Saving...';
  const blob=new Blob(chunks,{type:'video/webm'});
  const r=await (await fetch('/upload_take',{method:'POST',body:blob,headers:{'content-type':'video/webm'}})).json();
  st.textContent='Saved ('+r.mb+' MB). You can record another take, or close this page.';btn.disabled=false;btn.textContent='Record another take';
 };
 rec.start(1000);t0=Date.now();btn.textContent='Stop recording';
 var tm=setInterval(()=>{st.textContent='Recording '+Math.round((Date.now()-t0)/1000)+' s'},500);
};
</script>"""


def selftest(av: Avatar, out: Path):
    tiles = []
    for m in (0.0, 0.3, 0.6, 1.0):
        tiles.append(av.render(mouth=m))
    row1 = np.hstack([cv2.resize(t, (360, 360)) for t in tiles])
    tiles2 = [av.render(blink=1.0), av.render(yaw=12), av.render(yaw=-12), av.render(pitch=8)]
    row2 = np.hstack([cv2.resize(t, (360, 360)) for t in tiles2])
    cv2.imwrite(str(out), cv2.cvtColor(np.vstack([row1, row2]), cv2.COLOR_RGB2BGR))
    # speed
    av.render()
    torch.cuda.synchronize()
    t = time.time()
    n = 60
    for i in range(n):
        av.render(yaw=i % 5, mouth=(i % 10) / 10)
    torch.cuda.synchronize()
    dt = (time.time() - t) / n
    print(f"render: {dt * 1000:.1f} ms/frame = {1 / dt:.0f} fps (incl. paste-back at {av.src_rgb.shape[1]}x{av.src_rgb.shape[0]})")


def serve(av: Avatar, port: int, sample: Path | None, takes_dir: Path | None = None):
    import uvicorn
    from fastapi import FastAPI, Request, Response
    from fastapi.responses import HTMLResponse, StreamingResponse

    director = Director()
    latest = {"jpg": b"", "n": 0}
    cond = threading.Condition()
    stats = {"fps": 0.0}

    def loop():
        target = 1 / 30
        last = time.time()
        frames = 0
        tfps = time.time()
        while True:
            now = time.time()
            rgb = av.render(**director.state(now))
            ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
            with cond:
                latest["jpg"] = buf.tobytes()
                latest["n"] += 1
                cond.notify_all()
            frames += 1
            if time.time() - tfps >= 2:
                stats["fps"] = frames / (time.time() - tfps)
                frames, tfps = 0, time.time()
            sleep = target - (time.time() - last)
            last = time.time()
            if sleep > 0:
                time.sleep(sleep)

    threading.Thread(target=loop, daemon=True).start()

    app = FastAPI()

    @app.get("/health")
    def health():
        return {"ok": True, "fps": round(stats["fps"], 1)}

    @app.get("/stream")
    def stream():
        def gen():
            seen = 0
            while True:
                with cond:
                    cond.wait_for(lambda: latest["n"] != seen, timeout=2)
                    seen = latest["n"]
                    jpg = latest["jpg"]
                if jpg:
                    yield b"--f\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n"

        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=f")

    @app.post("/speak")
    async def speak(request: Request):
        data = await request.body()
        wav, sr = sf.read(io.BytesIO(data), always_2d=True)
        director.speak(wav.astype(np.float32), sr)
        return {"seconds": round(len(wav) / sr, 2)}

    @app.post("/upload_take")
    async def upload_take(request: Request):
        if takes_dir is None:
            return Response("no --takes-dir configured", status_code=400)
        takes_dir.mkdir(parents=True, exist_ok=True)
        data = await request.body()
        ext = "mp4" if "mp4" in (request.headers.get("content-type") or "") else "webm"
        path = takes_dir / f"take_{int(time.time())}.{ext}"
        path.write_bytes(data)
        return {"saved": str(path), "mb": round(len(data) / 1e6, 1)}

    @app.get("/record", response_class=HTMLResponse)
    def record_page():
        return RECORD_HTML

    @app.get("/sample.wav")
    def sample_wav():
        return Response(sample.read_bytes() if sample else b"", media_type="audio/wav")

    @app.get("/", response_class=HTMLResponse)
    def page():
        return """<!doctype html><meta charset=utf-8><title>Live avatar</title>
<body style="background:#0b1020;color:#ddd;font-family:sans-serif;text-align:center">
<h3>Live avatar (local, LivePortrait)</h3>
<img id=v src="/stream" style="max-width:720px;width:95%;border-radius:12px"><br><br>
<button id=b style="font-size:18px;padding:10px 20px">Speak test</button>
<p id=s style="opacity:.7"></p>
<script>
const s=document.getElementById('s'), v=document.getElementById('v');
// If the stream drops (engine restarted, network hiccup) reconnect by itself instead of staying blank.
let lastOk=Date.now();
v.onload=()=>{lastOk=Date.now()};
v.onerror=()=>setTimeout(()=>{v.src='/stream?'+Date.now()},1000);
setInterval(()=>{ if(Date.now()-lastOk>6000){ v.src='/stream?'+Date.now(); lastOk=Date.now(); } },3000);
setInterval(async()=>{try{const r=await (await fetch('/health')).json();s.textContent='streaming at '+r.fps+' fps';lastOk=Date.now()}catch(e){}},2000);
document.getElementById('b').onclick=async()=>{
 const buf=await (await fetch('/sample.wav')).arrayBuffer();
 const a=new Audio(URL.createObjectURL(new Blob([buf],{type:'audio/wav'})));
 await a.play();                                  // audio and mouth start together
 fetch('/speak',{method:'POST',body:buf});
};
</script>"""

    print(f"Live avatar on http://127.0.0.1:{port}/", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--photo", required=True)
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--sample", default=None, help="wav used by the demo button")
    ap.add_argument("--takes-dir", default=None, help="where /record saves talking videos (e.g. data/media/<user>/talk)")
    ap.add_argument("--selftest", default=None, help="write a contact sheet to this path and exit")
    a = ap.parse_args()
    avatar = Avatar((ORIG_CWD / a.photo).resolve())
    if a.selftest:
        selftest(avatar, (ORIG_CWD / a.selftest).resolve())
    else:
        serve(avatar, a.port, (ORIG_CWD / a.sample).resolve() if a.sample else None, (ORIG_CWD / a.takes_dir).resolve() if a.takes_dir else None)
