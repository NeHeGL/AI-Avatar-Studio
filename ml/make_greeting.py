"""Startup greeting: a local LLM writes a line, the cloned voice speaks it, InfiniteTalk renders the video."""
import json, subprocess, sys, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
PY = str(HERE / ".venv" / "Scripts" / "python.exe")
ROOT = HERE.parent
NAME = "Jeff Molofee"
REF_VOICE = sorted((ROOT / "data/media/jeff_molofee/voice").glob("*.ogg"), key=lambda p: p.stat().st_mtime)[-1]
PHOTO = next(
    (ROOT / "data/media/jeff_molofee/scan").glob("angle_100_*.jpg"),
    next((ROOT / "data/media/jeff_molofee/scan").glob("angle_*_*.jpg"), None),
)

PROMPT = ("You are the digital avatar of a person named Jeff. Greet him warmly as the program starts, then share one "
          "interesting, well-known true fact. Write exactly two short spoken sentences, at most 35 words in total, "
          "plain speech, no emojis, no quotation marks, no stage directions.")

def llm_line():
    try:
        req = urllib.request.Request("http://127.0.0.1:11434/api/generate", method="POST",
            data=json.dumps({"model": "gemma-opus-cuda:q4km", "prompt": PROMPT, "stream": False, "keep_alive": 0,
                             "options": {"temperature": 0.9, "num_predict": 90}}).encode())
        text = json.loads(urllib.request.urlopen(req, timeout=180).read())["response"].strip().replace("\n", " ")
        text = text.strip('"').strip()
        if 10 < len(text) < 400:
            return text
    except Exception as e:
        print("LLM unavailable:", e)
    return "Hey, welcome back. Did you know honey never spoils? Archaeologists have found edible honey in ancient Egyptian tombs."

text = llm_line()
(OUT / "greeting.txt").write_text(text, encoding="utf-8")
print("LINE:", text, flush=True)
subprocess.run([PY, str(HERE / "voice_clone.py"), str(REF_VOICE), text, str(OUT / "greeting.wav")], check=True)
subprocess.run([PY, str(HERE / "comfy_run.py"), "--image", str(PHOTO), "--audio", str(OUT / "greeting.wav"),
                "--out", str(OUT / "greeting.mp4")], check=True)
print("GREETING READY", flush=True)
