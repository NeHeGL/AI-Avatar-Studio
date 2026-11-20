"""Side-by-side test: the same photo + the same speech, rendered by Ditto and by ComfyUI InfiniteTalk, with timings.

    ml\\.venv\\Scripts\\python.exe ml\\compare_talk.py <user_slug> --engine ditto     (needs the studio running: Ditto is its service on :8771)
    ml\\.venv\\Scripts\\python.exe ml\\compare_talk.py <user_slug> --engine comfy     (stop the studio first: InfiniteTalk needs the GPU to itself)
    ml\\.venv\\Scripts\\python.exe ml\\compare_talk.py <user_slug> --html            (builds the page)

Output: data/media/<user>/compare/  (clips, timings.json, index.html)  ->  http://localhost:8765/data/media/<user>/compare/index.html
Nothing here changes the app or the person's profile.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ML = ROOT / "ml"
PY = ML / ".venv" / "Scripts" / "python.exe"
DEFAULT_CLIPS = ["greeting_1", "reaction_agree", "thinking_1"]  # existing speech in the person's cloned voice


def gpu_mb() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    return int(out.strip().splitlines()[0])


def unload_llm() -> None:
    try:
        for m in json.loads(urllib.request.urlopen("http://127.0.0.1:11434/api/ps", timeout=5).read()).get("models", []):
            body = json.dumps({"model": m["name"], "keep_alive": 0}).encode()
            urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:11434/api/generate", data=body, method="POST"), timeout=30).read()
            print("unloaded language model", m["name"], flush=True)
    except Exception:
        pass


def load_timings(d: Path) -> dict:
    f = d / "timings.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def save_timings(d: Path, t: dict) -> None:
    (d / "timings.json").write_text(json.dumps(t, indent=2), encoding="utf-8")


def seconds_of(wav: Path) -> float:
    import soundfile as sf

    return float(sf.info(str(wav)).duration)


def run_ditto(clips, photo: Path, src: Path, d: Path) -> None:
    import urllib.parse

    t = load_timings(d)
    base = "http://127.0.0.1:8771/render?"
    for i, cid in enumerate([clips[0]] + list(clips)):  # the first clip is rendered twice: the first run is only warm-up
        wav = src / f"{cid}.wav"
        out = d / f"{cid}_ditto.mp4"
        url = base + urllib.parse.urlencode({"source": str(photo), "out": str(out)})
        t0 = time.time()
        with urllib.request.urlopen(urllib.request.Request(url, data=wav.read_bytes(), method="POST", headers={"Content-Type": "audio/wav"}), timeout=600) as r:
            ok = json.loads(r.read()).get("ok")
        wall = time.time() - t0
        if i == 0:
            print(f"ditto warm-up: {wall:.1f}s (not counted)", flush=True)
            continue
        print(f"ditto {cid}: {wall:.1f}s for {seconds_of(wav):.1f}s of speech ok={ok}", flush=True)
        t.setdefault(cid, {})["ditto"] = {"seconds": round(wall, 1), "speech": round(seconds_of(wav), 1)}
        save_timings(d, t)


def run_comfy(clips, photo: Path, src: Path, d: Path, size: int, steps: int) -> None:
    unload_llm()
    used = gpu_mb()
    if used > 14000:
        sys.exit(f"GPU already at {used} MiB: stop the studio (Stop Avatar Studio.bat) so InfiniteTalk has the GPU to itself, then re-run.")
    t = load_timings(d)
    prompt = "a man speaking calmly to the camera, head almost still, natural blinking, photorealistic"
    for i, cid in enumerate(clips):
        wav = src / f"{cid}.wav"
        out = d / f"{cid}_comfy.mp4"
        t0 = time.time()
        proc = subprocess.Popen([str(PY), str(ML / "comfy_run.py"), "--image", str(photo), "--audio", str(wav), "--out", str(out),
                                 "--size", str(size), "--steps", str(steps), "--prompt", prompt], cwd=str(ML))
        while proc.poll() is None:
            time.sleep(5)
            if gpu_mb() > 30500:
                proc.kill()
                sys.exit("GPU memory nearly full: aborted to protect the driver.")
        wall = time.time() - t0
        if proc.returncode != 0 or not out.exists():
            print("comfy render failed for", cid, flush=True)
            continue
        note = "includes loading the models" if i == 0 else "models already warm"
        print(f"comfy {cid}: {wall:.0f}s for {seconds_of(wav):.1f}s of speech ({note})", flush=True)
        t.setdefault(cid, {})["comfy"] = {"seconds": round(wall, 1), "speech": round(seconds_of(wav), 1), "note": note}
        save_timings(d, t)


PAGE = """<!doctype html><meta charset="utf-8"><title>Ditto vs InfiniteTalk</title>
<style>
body{font-family:system-ui,sans-serif;background:#10141c;color:#e8ecf4;margin:0;padding:1.2rem}
h1{font-size:1.2rem;margin:0 0 .6rem}.bar{display:flex;gap:.6rem;align-items:center;margin-bottom:1rem;flex-wrap:wrap}
button{background:#25304a;color:#e8ecf4;border:1px solid #3b4a70;border-radius:8px;padding:.45rem .9rem;cursor:pointer;font-size:.95rem}
button.on{background:#3d7bff;border-color:#3d7bff}
.clip{margin:0 0 1.6rem}.row{display:flex;gap:1rem;flex-wrap:wrap}.box{flex:1 1 300px;max-width:520px}
video{width:100%;border-radius:10px;background:#000}.cap{font-size:.85rem;opacity:.85;margin:.3rem 0}
.hide{display:none}
</style>
<h1>Same photo, same voice: Ditto vs InfiniteTalk (ComfyUI)</h1>
<div class="bar">
 <button id="b-both" class="on">Side by side</button><button id="b-ditto">Ditto only</button><button id="b-comfy">InfiniteTalk only</button>
 <button id="play">Play all</button><span class="cap">Videos start together so lips can be compared. Timings are real render times on this machine.</span>
</div>
<div id="clips"></div>
<script>
const DATA=__DATA__;
const box=document.getElementById('clips');
for(const c of DATA){
  const t=c.t||{};
  const cap=(k,l)=>t[k]?`${l}: ${t[k].seconds}s to render ${t[k].speech}s of speech (${(t[k].seconds/t[k].speech).toFixed(1)}x real time${t[k].note?', '+t[k].note:''})`:`${l}: not rendered yet`;
  box.insertAdjacentHTML('beforeend',`<div class="clip"><div class="cap"><b>${c.id}</b></div><div class="row">
   <div class="box" data-e="ditto"><video src="${c.id}_ditto.mp4" controls playsinline></video><div class="cap">${cap('ditto','Ditto')}</div></div>
   <div class="box" data-e="comfy"><video src="${c.id}_comfy.mp4" controls playsinline></video><div class="cap">${cap('comfy','InfiniteTalk')}</div></div></div></div>`);
}
function show(m){for(const b of document.querySelectorAll('.box'))b.classList.toggle('hide',m!=='both'&&b.dataset.e!==m);
 for(const id of ['both','ditto','comfy'])document.getElementById('b-'+id).classList.toggle('on',id===m);}
for(const id of ['both','ditto','comfy'])document.getElementById('b-'+id).onclick=()=>show(id);
document.getElementById('play').onclick=()=>{for(const v of document.querySelectorAll('.box:not(.hide) video')){v.currentTime=0;v.play().catch(()=>{})}};
</script>"""


def build_html(clips, d: Path) -> None:
    t = load_timings(d)
    data = [{"id": c, "t": t.get(c, {})} for c in clips]
    (d / "index.html").write_text(PAGE.replace("__DATA__", json.dumps(data)), encoding="utf-8")
    print("page ->", d / "index.html")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("user")
    ap.add_argument("--engine", choices=["ditto", "comfy"])
    ap.add_argument("--html", action="store_true")
    ap.add_argument("--clips", default=",".join(DEFAULT_CLIPS))
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=4)
    a = ap.parse_args()
    cdir = ROOT / "data" / "media" / a.user / "cinematic"
    d = ROOT / "data" / "media" / a.user / "compare"
    d.mkdir(parents=True, exist_ok=True)
    clips = [c for c in a.clips.split(",") if c]
    photo, src = cdir / "source.jpg", cdir / "src"
    if a.engine == "ditto":
        run_ditto(clips, photo, src, d)
    elif a.engine == "comfy":
        run_comfy(clips, photo, src, d, a.size, a.steps)
    if a.html or a.engine:
        build_html(clips, d)


if __name__ == "__main__":
    main()
