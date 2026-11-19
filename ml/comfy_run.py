"""Generate a talking-head video with InfiniteTalk (Wan2.1) through a local ComfyUI.

usage: python comfy_run.py --image photo.jpg --audio speech.wav --out out.mp4 [--prompt "..."] [--size 640]

Takes the stock Kijai InfiniteTalk "image + audio -> video" workflow (UI format), converts it to ComfyUI's
API format, points it at our local model files and inputs, runs it, and copies the resulting mp4.
Starts ComfyUI itself if it is not already running on 127.0.0.1:8188.
"""
import argparse
import json
import math
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMFY = HERE / "ComfyUI"
PY = HERE / ".venv" / "Scripts" / "python.exe"
WORKFLOW = COMFY / "custom_nodes" / "ComfyUI-WanVideoWrapper" / "example_workflows" / "wanvideo_2_1_14B_I2V_InfiniteTalk_example_03.json"
URL = "http://127.0.0.1:8188"
FPS = 25


def http(path, data=None, timeout=60):
    req = urllib.request.Request(URL + path, data=json.dumps(data).encode() if data is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def server_up():
    try:
        http("/system_stats", timeout=3)
        return True
    except Exception:
        return False


def ensure_server():
    if server_up():
        return None
    log = open(HERE / "out" / "comfyui.log", "w", encoding="utf-8")
    proc = subprocess.Popen([str(PY), "main.py", "--listen", "127.0.0.1", "--port", "8188", "--disable-auto-launch"],
                            cwd=str(COMFY), stdout=log, stderr=subprocess.STDOUT)
    for _ in range(240):
        if server_up():
            return proc
        if proc.poll() is not None:
            raise SystemExit(f"ComfyUI exited early - see {HERE / 'out' / 'comfyui.log'}")
        time.sleep(1)
    raise SystemExit("ComfyUI did not start in 4 minutes")


# ---------------------------------------------------------------- UI workflow -> API prompt
def convert(ui, info):
    nodes = {n["id"]: n for n in ui["nodes"]}
    links = {l[0]: l for l in ui["links"]}  # [id, from_node, from_slot, to_node, to_slot, type]
    set_src = {}
    for n in nodes.values():
        if n["type"] == "SetNode":
            lk = n["inputs"][0].get("link")
            if lk is not None:
                set_src[n["widgets_values"][0]] = (links[lk][1], links[lk][2])

    def resolve(node_id, slot):
        n = nodes[node_id]
        if n["type"] == "GetNode":
            return resolve(*set_src[n["widgets_values"][0]])
        if n["type"] in ("SetNode", "Reroute"):
            return resolve(links[n["inputs"][0]["link"]][1], links[n["inputs"][0]["link"]][2])
        return node_id, slot

    api = {}
    for n in nodes.values():
        t = n["type"]
        if t not in info:
            continue
        spec = info[t]
        order = spec.get("input_order") or {"required": list(spec["input"].get("required", {})), "optional": list(spec["input"].get("optional", {}))}
        names = order.get("required", []) + order.get("optional", [])
        defs = {**spec["input"].get("required", {}), **spec["input"].get("optional", {})}
        ui_inputs = {i["name"]: i for i in n.get("inputs", [])}
        wv = n.get("widgets_values", [])
        inputs = {}
        if isinstance(wv, dict):
            inputs.update({k: v for k, v in wv.items() if k in defs})
            wv = []
        wi = 0
        for name in names:
            d = defs[name]
            kind = d[0]
            is_widget = isinstance(kind, list) or kind in ("INT", "FLOAT", "STRING", "BOOLEAN", "COMBO") or (isinstance(kind, str) and kind not in ("*",) and False)
            cfg = d[1] if len(d) > 1 and isinstance(d[1], dict) else {}
            link = ui_inputs.get(name, {}).get("link")
            if is_widget:
                val = wv[wi] if wi < len(wv) else None
                wi += 1
                if cfg.get("control_after_generate") or name in ("seed", "noise_seed"):
                    wi += 1  # the UI stores the "fixed/randomize" combo right after the seed
                if link is None and val is not None:
                    inputs[name] = val
            if link is not None:
                src, slot = resolve(links[link][1], links[link][2])
                inputs[name] = [str(src), slot]
        api[str(n["id"])] = {"class_type": t, "inputs": inputs}

    # keep only what the output node needs
    outs = [k for k, v in api.items() if v["class_type"] == "VHS_VideoCombine"]
    keep, stack = set(), list(outs)
    while stack:
        k = stack.pop()
        if k in keep or k not in api:
            continue
        keep.add(k)
        for v in api[k]["inputs"].values():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                stack.append(v[0])
    return {k: api[k] for k in keep}


def by_class(api, cls):
    return [(k, v) for k, v in api.items() if v["class_type"] == cls]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default="a person talking naturally to the camera, subtle natural head movement, blinking, soft realistic lighting")
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--dump-api", action="store_true", help="only write the converted API prompt and exit")
    a = ap.parse_args()

    (HERE / "out").mkdir(exist_ok=True)
    proc = ensure_server()
    info = http("/object_info", timeout=120)
    ui = json.load(open(WORKFLOW, encoding="utf-8"))
    api = convert(ui, info)

    # ---- inputs
    inp = COMFY / "input"
    inp.mkdir(exist_ok=True)
    img_name, aud_name = f"avatar_{uuid.uuid4().hex[:6]}{Path(a.image).suffix}", f"speech_{uuid.uuid4().hex[:6]}{Path(a.audio).suffix}"
    shutil.copy(a.image, inp / img_name)
    shutil.copy(a.audio, inp / aud_name)
    import soundfile as sf
    sec = sf.info(a.audio).duration
    frames = max(25, int(math.ceil(sec * FPS)) // 4 * 4 + 1)

    for k, v in by_class(api, "LoadImage"):
        v["inputs"]["image"] = img_name
    for k, v in by_class(api, "LoadAudio"):
        v["inputs"]["audio"] = aud_name
    # models
    for k, v in by_class(api, "WanVideoModelLoader"):
        v["inputs"].update(model="Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors", base_precision="fp16_fast",
                           quantization="fp8_e4m3fn", attention_mode="sdpa")
    for k, v in by_class(api, "MultiTalkModelLoader"):
        v["inputs"]["model"] = "Wan2_1-InfiniTetalk-Single_fp16.safetensors"
    for k, v in by_class(api, "WanVideoVAELoader"):
        v["inputs"]["model_name"] = "Wan2_1_VAE_bf16.safetensors"
    for k, v in by_class(api, "WanVideoLoraSelect"):
        v["inputs"]["lora"] = "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"
    for k, v in by_class(api, "WanVideoTextEncodeCached"):
        v["inputs"].update(model_name="umt5-xxl-enc-fp8_e4m3fn.safetensors", positive_prompt=a.prompt)
    for k, v in by_class(api, "CLIPVisionLoader"):
        v["inputs"]["clip_name"] = "clip_vision_h.safetensors"
    for k, v in by_class(api, "WanVideoSampler"):
        v["inputs"]["steps"] = a.steps
    for k, v in by_class(api, "INTConstant"):
        pass
    # size + length: the three INTConstants are width, height, max_frames; identify by value
    for k, v in by_class(api, "INTConstant"):
        val = v["inputs"].get("value")
        if val == 640:
            v["inputs"]["value"] = a.size
        elif val == 500:
            v["inputs"]["value"] = frames
    # clean speech: skip the optional vocal-separation branch, feed the audio straight in
    audio_src = [k for k, v in by_class(api, "LoadAudio")][0]
    for k, v in by_class(api, "MultiTalkWav2VecEmbeds"):
        v["inputs"]["audio_1"] = [audio_src, 0]
    for k, v in by_class(api, "VHS_VideoCombine"):
        v["inputs"]["save_output"] = True
        v["inputs"]["filename_prefix"] = "avatar_talk"
    for k in [k for k, v in api.items() if v["class_type"].startswith("MelBandRoFormer")]:
        del api[k]

    if a.dump_api:
        json.dump(api, open(HERE / "out" / "api_prompt.json", "w"), indent=1)
        print("wrote out/api_prompt.json with", len(api), "nodes")
        return

    print(f"audio {sec:.1f}s -> {frames} frames at {FPS} fps, {a.size}px", flush=True)
    pid = http("/prompt", {"prompt": api, "client_id": uuid.uuid4().hex})
    if "prompt_id" not in pid:
        raise SystemExit("ComfyUI rejected the workflow:\n" + json.dumps(pid, indent=1)[:3000])
    pid = pid["prompt_id"]
    t0 = time.time()
    while True:
        h = http(f"/history/{pid}")
        if pid in h:
            break
        time.sleep(3)
        if int(time.time() - t0) % 30 == 0:
            print(f"  ...{int(time.time()-t0)}s", flush=True)
    res = h[pid]
    if res.get("status", {}).get("status_str") == "error":
        raise SystemExit("generation failed:\n" + json.dumps(res["status"], indent=1)[:3000])
    for node_out in res["outputs"].values():
        for g in node_out.get("gifs", []):
            src = COMFY / "output" / g.get("subfolder", "") / g["filename"]
            shutil.copy(src, a.out)
            print(f"done in {time.time()-t0:.0f}s -> {a.out}")
            return
    raise SystemExit("no video in outputs: " + json.dumps(res["outputs"])[:500])


if __name__ == "__main__":
    main()
