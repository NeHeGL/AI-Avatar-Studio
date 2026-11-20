"""Local render service: speech WAV in -> talking-head MP4 out (Ditto, full quality). Nothing leaves this machine.

Keeps Ditto's models loaded so each sentence of a reply renders in about 1.3x its own length.

    ml\\realtime\\dittoenv\\Scripts\\python.exe ml\\realtime\\ditto_render_server.py [--port 8771]

    POST /render?source=<photo path>&out=<mp4 path>   body = WAV bytes   ->  {"ok": true, "path": "...", "seconds": 4.2}
    GET  /health
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DITTO = HERE / "ditto"
FFMPEG_BIN = HERE / "ffmpeg" / "ffmpeg-master-latest-win64-gpl" / "bin"
os.environ["PATH"] = str(FFMPEG_BIN) + os.pathsep + os.environ.get("PATH", "")
os.environ.setdefault("PYTHONUTF8", "1")
sys.path.insert(0, str(DITTO))
os.chdir(DITTO)

import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from inference import run, seed_everything
from stream_pipeline_offline import StreamSDK

CFG = "./checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl"
DATA_ROOT = "./checkpoints/ditto_pytorch"
MAX_SIZE = int(os.environ.get("DITTO_MAX_SIZE", "768"))  # longest side of the output picture


def _patch_gpu_resident():
    """Ditto's PyTorch renderer copies the 8 MB feature volume CPU->GPU and the result GPU->CPU->GPU on every frame.
    Keep everything on the GPU: cache the source features there and hand the warp result straight to the decoder."""
    import core.models.decoder as dec
    import core.models.warp_network as wn

    cache = {}

    def warp(self, feature_3d, kp_source, kp_driving):
        hit = cache.get(id(feature_3d))
        if hit is None or hit[0] is not feature_3d:
            cache.clear()
            hit = (feature_3d, torch.from_numpy(feature_3d).to(self.device))
            cache[id(feature_3d)] = hit
        with torch.no_grad(), torch.autocast(device_type=self.device[:4], dtype=torch.float16, enabled=True):
            return self.model(hit[1], torch.from_numpy(kp_source).to(self.device), torch.from_numpy(kp_driving).to(self.device))

    def decode(self, feature):
        with torch.no_grad(), torch.autocast(device_type=self.device[:4], dtype=torch.float16, enabled=True):
            out = self.model(feature if isinstance(feature, torch.Tensor) else torch.from_numpy(feature).to(self.device))
        return (out[0].float().permute(1, 2, 0).clamp(0, 1) * 255).cpu().numpy()

    wn.WarpNetwork.__call__ = warp
    dec.Decoder.__call__ = decode


_patch_gpu_resident()
seed_everything(1024)
print("loading Ditto models...", flush=True)
_t = time.time()
SDK = StreamSDK(CFG, DATA_ROOT)
print(f"models loaded in {time.time() - _t:.1f}s", flush=True)
LOCK = threading.Lock()
app = FastAPI()
TMP = ROOT / "data" / "tmp_render"


@app.get("/health")
def health():
    return {"ok": True, "max_size": MAX_SIZE}


@app.post("/render")
async def render(request: Request, source: str, out: str):
    data = await request.body()
    TMP.mkdir(parents=True, exist_ok=True)
    wav = TMP / f"{uuid.uuid4().hex[:10]}.wav"
    wav.write_bytes(data)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def work():
        with LOCK:
            t = time.time()
            run(SDK, str(wav), source, str(out_path), {"setup_kwargs": {"max_size": MAX_SIZE}})
            return time.time() - t

    try:
        import asyncio

        secs = await asyncio.to_thread(work)
        return {"ok": True, "path": str(out_path), "seconds": round(secs, 2)}
    except Exception as e:  # the app falls back to the puppet on any failure
        return JSONResponse({"ok": False, "error": str(e)[:300]}, status_code=500)
    finally:
        try:
            wav.unlink()
            tmp_mp4 = Path(str(out_path) + ".tmp.mp4")
            if tmp_mp4.exists():
                tmp_mp4.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8771)
    a = ap.parse_args()
    print(f"Ditto render service on http://127.0.0.1:{a.port}/", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")
