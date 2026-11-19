"""Talk to your digital twin, out loud. Fully local.

    mic -> voice-activity detection (Silero) -> speech-to-text (Whisper large-v3-turbo)
        -> local LLM (Ollama) -> your cloned voice (Chatterbox), spoken sentence by sentence

usage:
    python converse.py                         # live, with the microphone (use headphones: it can't hear itself)
    python converse.py --selftest clip.wav     # feed a WAV as "what you said"; saves the reply, no mic/speakers
    python converse.py --text "hello there"    # type instead of speaking
    python converse.py --list-devices
Options: --llm <ollama model>  --mic <device index>  --voice <reference .wav/.ogg>  --name "Jeff"
"""
import argparse
import json
import queue
import re
import sys
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OLLAMA = "http://127.0.0.1:11434"
SR_IN = 16000


def log(*a):
    print(*a, flush=True)


def default_voice():
    vs = sorted((ROOT / "data/media/jeff_molofee/voice").glob("*.*"), key=lambda p: p.stat().st_mtime)
    return vs[-1] if vs else None


# ------------------------------------------------------------------ models
class Ears:
    def __init__(self):
        from transformers import pipeline
        t0 = time.time()
        self.asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo",
                            torch_dtype=torch.float16, device="cuda")
        log(f"[ears] whisper ready in {time.time()-t0:.1f}s")

    def transcribe(self, audio16k: np.ndarray) -> str:
        out = self.asr({"raw": audio16k.astype(np.float32), "sampling_rate": SR_IN},
                       generate_kwargs={"language": "en", "task": "transcribe"})
        return out["text"].strip()


class Voice:
    def __init__(self, reference: Path):
        from chatterbox.tts import ChatterboxTTS
        t0 = time.time()
        data, sr = sf.read(str(reference), always_2d=True)
        ref = HERE / "out" / "_converse_ref.wav"
        ref.parent.mkdir(exist_ok=True)
        sf.write(str(ref), data.mean(axis=1).astype(np.float32), sr)
        self.tts = ChatterboxTTS.from_pretrained(device="cuda")
        self.tts.prepare_conditionals(str(ref), exaggeration=0.5)  # learn the voice once
        self.sr = self.tts.sr
        log(f"[voice] cloned voice ready in {time.time()-t0:.1f}s (reference: {reference.name})")

    def speak(self, text: str) -> np.ndarray:
        wav = self.tts.generate(text, cfg_weight=0.5, exaggeration=0.5)
        return wav.squeeze().cpu().numpy().astype(np.float32)


class Brain:
    def __init__(self, model: str, name: str):
        self.model = model
        self.history = []
        facts_file = HERE / "memory" / "facts.txt"
        facts = [l.strip() for l in facts_file.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")] if facts_file.exists() else []
        NL = chr(10)
        known = (("What you know about " + name + " (this is ALL you know):" + NL + "- " + (NL + "- ").join(facts))
                 if facts else f"You have not been told anything about {name}'s life yet.")
        self.system = (
            f"You are the digital twin of {name}: a voice avatar that talks the way {name} does, in the first person. "
            "You are in a spoken conversation, so answer in one to three short, natural sentences. "
            "No lists, no markdown, no emojis, no stage directions." + NL + NL
            + known + NL + NL +
            "STRICT RULES ABOUT TRUTH: You have no memory of past days and no senses. You do not know what the user ate, did, "
            "wore, felt or said earlier unless it appears in the facts above or earlier in THIS conversation. "
            "If asked about anything personal that is not in the facts, say plainly that you do not know or cannot remember, "
            "in a friendly way, and never invent details or pretend to remember. "
            "General knowledge questions (science, history, how things work) you may answer normally and briefly."
        )

    def reply_stream(self, user_text: str):
        self.history.append({"role": "user", "content": user_text})
        msgs = [{"role": "system", "content": self.system}] + self.history[-20:]
        req = urllib.request.Request(OLLAMA + "/api/chat", method="POST", data=json.dumps(
            {"model": self.model, "messages": msgs, "stream": True, "keep_alive": "30m",
             "options": {"temperature": 0.7, "num_predict": 220}}).encode())
        full = []
        with urllib.request.urlopen(req, timeout=300) as r:
            for line in r:
                if not line.strip():
                    continue
                chunk = json.loads(line)
                tok = chunk.get("message", {}).get("content", "")
                if tok:
                    full.append(tok)
                    yield tok
        self.history.append({"role": "assistant", "content": "".join(full).strip()})


SENT_END = re.compile(r"(?<=[.!?])\s+")


def sentences(token_iter, min_chars=24):
    """Group streamed tokens into speakable sentences (speak the first one as soon as it is complete)."""
    buf = ""
    for tok in token_iter:
        buf += tok
        parts = SENT_END.split(buf)
        while len(parts) > 1:
            first = parts[0].strip()
            if len(first) >= min_chars:
                yield first
                buf = " ".join(parts[1:])
                parts = SENT_END.split(buf)
            else:
                break
    rest = buf.strip()
    if rest:
        yield rest


def clean(text):
    return re.sub(r"[*_#`>]+", "", text).replace("\n", " ").strip()


# ------------------------------------------------------------------ one conversational turn
def respond(brain, voice, user_text, play=True, save_to=None):
    t_start = time.time()
    chunks_q = queue.Queue()
    played_first = {"t": None}
    collected = []

    def player():
        while True:
            item = chunks_q.get()
            if item is None:
                return
            if played_first["t"] is None:
                played_first["t"] = time.time()
            collected.append(item)
            if play:
                sd.play(item, voice.sr)
                sd.wait()

    th = threading.Thread(target=player, daemon=True)
    th.start()
    said = []
    first_token = None
    tts_time = 0.0
    audio_secs = 0.0

    def tokens():
        nonlocal first_token
        for tok in brain.reply_stream(user_text):
            if first_token is None:
                first_token = time.time() - t_start
            yield tok

    for sent in sentences(tokens()):
        sent = clean(sent)
        if not sent:
            continue
        said.append(sent)
        t0 = time.time()
        audio = voice.speak(sent)
        tts_time += time.time() - t0
        audio_secs += len(audio) / voice.sr
        chunks_q.put(audio)
        log(f"   twin> {sent}")
    chunks_q.put(None)
    th.join()
    total = time.time() - t_start
    log(f"   [timing] first LLM token {first_token or 0:.2f}s | speech starts after "
        f"{(played_first['t'] - t_start) if played_first['t'] else 0:.2f}s | voice {tts_time:.1f}s to make {audio_secs:.1f}s of audio "
        f"(x{(tts_time / audio_secs) if audio_secs else 0:.2f} real time) | total {total:.1f}s")
    if save_to and collected:
        sf.write(str(save_to), np.concatenate(collected), voice.sr)
    return " ".join(said)


def save_turn(user_text, reply):
    d = HERE / "memory" / "conversations"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f"{datetime.now():%Y-%m-%d}.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"t": datetime.now().isoformat(timespec="seconds"), "you": user_text, "twin": reply}, ensure_ascii=False) + "\n")


# ------------------------------------------------------------------ live microphone loop
def listen_loop(ears, brain, voice, mic):
    from silero_vad import load_silero_vad, VADIterator
    vad = VADIterator(load_silero_vad(), sampling_rate=SR_IN, threshold=0.5, min_silence_duration_ms=800, speech_pad_ms=250)
    q = queue.Queue()
    muted = threading.Event()

    def cb(indata, frames, t, status):
        if not muted.is_set():
            q.put(indata[:, 0].copy())

    log("\nListening. Speak when you are ready — pause for about a second when you finish. Ctrl+C to quit.\n")
    speech, in_speech = [], False
    with sd.InputStream(samplerate=SR_IN, channels=1, dtype="float32", blocksize=512, device=mic, callback=cb):
        while True:
            block = q.get()
            ev = vad(torch.from_numpy(block))
            if in_speech:
                speech.append(block)
            if ev and "start" in ev and not in_speech:
                in_speech, speech = True, [block]
            if ev and "end" in ev and in_speech:
                audio = np.concatenate(speech)
                in_speech, speech = False, []
                vad.reset_states()
                if len(audio) < SR_IN * 0.4:
                    continue
                muted.set()  # half-duplex: do not listen to ourselves
                t0 = time.time()
                text = ears.transcribe(audio)
                log(f"you> {text}   [heard in {time.time()-t0:.2f}s]")
                if text and len(text) > 1:
                    reply = respond(brain, voice, text)
                    save_turn(text, reply)
                while not q.empty():
                    q.get_nowait()
                vad.reset_states()
                muted.clear()
                log("\n(listening)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default="gemma-opus-cuda:q4km")
    ap.add_argument("--mic", type=int, default=None)
    ap.add_argument("--voice", default=None)
    ap.add_argument("--name", default="Jeff")
    ap.add_argument("--selftest", default=None)
    ap.add_argument("--text", default=None)
    ap.add_argument("--list-devices", action="store_true")
    a = ap.parse_args()
    if a.list_devices:
        print(sd.query_devices())
        return
    ref = Path(a.voice) if a.voice else default_voice()
    if not ref:
        raise SystemExit("No voice reference found. Pass --voice <file>.")
    brain = Brain(a.llm, a.name)
    voice = Voice(ref)
    if a.selftest or a.text:
        if a.selftest:
            ears = Ears()
            data, sr = sf.read(a.selftest, always_2d=True)
            mono = data.mean(axis=1).astype(np.float32)
            if sr != SR_IN:
                import torchaudio
                mono = torchaudio.functional.resample(torch.from_numpy(mono), sr, SR_IN).numpy()
            t0 = time.time()
            text = ears.transcribe(mono)
            log(f"you> {text}   [heard in {time.time()-t0:.2f}s]")
        else:
            text = a.text
        out = HERE / "out" / "converse_reply.wav"
        reply = respond(brain, voice, text, play=False, save_to=out)
        save_turn(text, reply)
        log(f"saved reply audio -> {out}")
        return
    ears = Ears()
    try:
        listen_loop(ears, brain, voice, a.mic)
    except KeyboardInterrupt:
        log("\nbye")


if __name__ == "__main__":
    main()
