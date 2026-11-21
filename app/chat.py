"""Spoken conversation with a profile's avatar: speech-to-text -> LLM -> cloned-voice TTS, streamed per sentence.

Models load once, on first use (GPU). The chat brain is the bundled llama.cpp server (brain/, started by start_all.py); override its address with AVATAR_LLM_URL.
"""
from __future__ import annotations

import base64
import io
import json
import os
import queue
import re
import sys
import threading
import time
import uuid
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ML = ROOT / "ml"
from app import topics as _topics  # noqa: E402
OUT = ML / "out"
if str(ML) not in sys.path:
    sys.path.insert(0, str(ML))
LLM_MODEL = os.environ.get("AVATAR_LLM", "brain")
LLM_URL = os.environ.get("AVATAR_LLM_URL", "http://127.0.0.1:8772")
# relaxed = casual private chat, no scolding about language; safe = tighter assistant tone
LLM_MODE = os.environ.get("AVATAR_LLM_MODE", "relaxed").strip().lower()
LLM_TEMPERATURE = float(os.environ.get("AVATAR_LLM_TEMPERATURE", "0.85" if LLM_MODE == "relaxed" else "0.7"))
LLM_NUM_CTX = int(os.environ.get("AVATAR_LLM_CTX", "3072"))  # conversation window in tokens: a bigger one costs GPU memory for nothing here
LLM_NUM_PREDICT = int(os.environ.get("AVATAR_LLM_NUM_PREDICT", "280"))
SR_IN = 16000

# Things Whisper "hears" in silence/noise.
PHANTOMS = {
    "you", "thank you", "thanks", "thank you.", "thanks for watching", "thank you for watching", "bye", "bye.",
    "okay", "ok", "oh", "uh", "um", "hmm", "mm", "mhm", "the", "so", "yeah", "i'm sorry", ".", "...",
}

def _is_phantom(text: str) -> bool:
    """Whisper invents text from silence/noise: a stock phrase, or the same short sentence repeated over and over."""
    norm = lambda t: t.lower().strip(" .!?,")  # noqa: E731
    stock = {norm(p) for p in PHANTOMS}
    if len(text) < 2 or norm(text) in stock:
        return True
    parts = [norm(x) for x in re.split(r"[.!?]+", text) if norm(x)]
    if len(parts) >= 3 and len(set(parts)) <= max(1, len(parts) // 4):  # one phrase repeated
        return True
    words = re.findall(r"[a-z']+", text.lower())
    return len(words) >= 8 and len(set(words)) <= 3


_lock = threading.Lock()  # one GPU job at a time
def partial_text(audio_base64: str) -> str:
    """Quick live transcript of what is being said right now (shown while the person is still talking). Never queues behind
    other GPU work: if the GPU is busy it just skips this round. Nothing is stored and nothing leaves the machine."""
    import soundfile as sf

    ears = _state.get("ears")
    if ears is None:
        return ""
    raw = base64.b64decode(audio_base64.split(",", 1)[-1])
    x, _sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if len(x) < SR_IN * 0.5 or float(np.sqrt((x ** 2).mean())) < 0.004:
        return ""
    if not _lock.acquire(timeout=2.0):  # the GPU is busy with something else: wait a moment, then give up this round
        _log("live words: GPU busy, skipped a round")
        return ""
    try:
        t0 = time.time()
        text = ears({"raw": x[-25 * SR_IN:], "sampling_rate": SR_IN}, chunk_length_s=28, generate_kwargs={"language": "en", "task": "transcribe"})["text"].strip()
    finally:
        _lock.release()
    _log(f"live words ({len(x) / SR_IN:.1f}s of speech, {time.time() - t0:.2f}s): {text!r}")
    return "" if _is_phantom(text) else text


_state = {"ears": None, "tts": None, "voice_key": None, "loading": False}
_sessions: dict[str, "Session"] = {}
_owner = {"token": None}  # only the newest chat window may talk
_log = lambda *a: print("[chat]", *a, flush=True)  # noqa: E731


def _wav_bytes(samples: np.ndarray, sr: int) -> bytes:
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, samples.astype(np.float32), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _load_any_audio(path: Path) -> tuple[np.ndarray, int]:
    """Mono float32 from wav/ogg/mp3/webm (webm/opus via PyAV when libsndfile can't read it)."""
    import soundfile as sf

    try:
        data, sr = sf.read(str(path), always_2d=True)
        return data.mean(axis=1).astype(np.float32), sr
    except Exception:
        import av

        chunks, sr = [], 16000
        with av.open(str(path)) as c:
            st = c.streams.audio[0]
            sr = st.rate
            for fr in c.decode(st):
                a = fr.to_ndarray()
                chunks.append(a.mean(axis=0) if a.ndim > 1 and a.shape[0] <= 8 else a.reshape(-1))
        x = np.concatenate(chunks).astype(np.float32)
        if x.dtype.kind in "iu" or np.abs(x).max() > 1.5:
            x = x / 32768.0
        return x, sr


_prog = {"ears": {"state": "wait", "pct": 0}, "voice": {"state": "wait", "pct": 0}}  # state: wait | download | setup


def _fetch(key: str, repo: str, patterns: list[str]) -> None:
    """Make sure a model's files are on disk (state 'download' with a percentage while they are fetched)."""
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(repo, allow_patterns=patterns, local_files_only=True)
        return  # already downloaded
    except Exception:
        pass
    from tqdm.auto import tqdm

    bars: dict[int, tuple[float, float]] = {}

    class _Bar(tqdm):
        def __init__(self, *a, **k):
            k.pop("name", None)
            super().__init__(*a, **k)

        def update(self, n=1):
            r = super().update(n)
            if self.total and self.total > 1_000_000:  # a bytes bar (not the "N files" bar)
                bars[id(self)] = (self.n, self.total)
                done, total = (sum(v[i] for v in bars.values()) for i in (0, 1))
                _prog[key]["pct"] = min(99, int(100 * done / max(total, 1)))
            return r

    _prog[key].update(state="download", pct=0)
    snapshot_download(repo, allow_patterns=patterns, tqdm_class=_Bar)


class _FastEars:
    """Whisper on the CPU through CTranslate2 (8-bit): about 0.9 s per utterance on a fast CPU, against 1-2 s for the plain version.
    Called like the transformers pipeline, so the rest of the app does not care which one it has."""

    def __init__(self, model_dir: str):
        from faster_whisper import WhisperModel

        self.model = WhisperModel(model_dir, device="cpu", compute_type="int8", cpu_threads=min(8, os.cpu_count() or 4))

    def __call__(self, inp, **kw):
        lang = (kw.get("generate_kwargs") or {}).get("language")
        segs, _info = self.model.transcribe(inp["raw"], language=lang, beam_size=1, vad_filter=False)
        return {"text": " ".join(s.text.strip() for s in segs)}


def _models():
    """Download (first run only) and load Whisper + Chatterbox once."""
    import torch
    from app import hardware

    if _state["ears"] is None:
        from transformers import pipeline

        t0 = time.time()
        ears_model = hardware.tier()["whisper"]
        ears_dev = hardware.tier()["whisper_device"]  # the GPU when the memory allows, else the CPU
        if ears_dev == "cpu":  # the faster CPU build
            _fetch("ears", ears_model, ["*"])
            _prog["ears"]["state"] = "setup"
            from huggingface_hub import snapshot_download

            _state["ears"] = _FastEars(snapshot_download(ears_model))
        else:
            _fetch("ears", ears_model, ["*.json", "*.txt", "model.safetensors"])
            _prog["ears"]["state"] = "setup"
            _state["ears"] = pipeline(
                "automatic-speech-recognition", model=ears_model,
                torch_dtype=torch.float16, device=ears_dev,
            )
        _log(f"whisper ready in {time.time() - t0:.1f}s")
    if _state["tts"] is None:
        from chatterbox.tts import ChatterboxTTS

        t0 = time.time()
        _fetch("voice", "ResembleAI/chatterbox", ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json", "conds.pt"])
        _prog["voice"]["state"] = "setup"
        _state["tts"] = ChatterboxTTS.from_pretrained(device="cuda")
        if hardware.tier()["name"] == "small":  # ONLY a card short of memory (the 16 GB tier): the voice runs leaner. Bigger cards never use half precision.
            _state["tts"].t3.half()
            _state["half_voice"] = True
            torch.cuda.empty_cache()
        _log(f"chatterbox ready in {time.time() - t0:.1f}s")
    return _state["ears"], _state["tts"]


def brain_name() -> str:
    """The chat model's own name from its file ("gemma-3-12b-it-Q4_K_M.gguf" -> "Gemma 3 12B"), or the alias if there is no file."""
    from app import hardware

    models = ROOT / "brain" / "models"
    f = models / hardware.tier()["brain_file"]
    if not f.exists():
        f = None if os.environ.get("AVATAR_VRAM_GB") else next(iter(sorted(models.glob("*.gguf"))), None)
    if f is None:
        return LLM_MODEL
    parts = [p for p in f.stem.split("-") if p.lower() != "it" and not re.fullmatch(r"[qQ]\d.*|f16|f32|bf16", p)]
    return " ".join(p.upper() if re.fullmatch(r"\d+(\.\d+)?[bB]", p) else p.capitalize() for p in parts) or LLM_MODEL


VIDEO_URL = os.environ.get("AVATAR_VIDEO_URL", "http://127.0.0.1:8771")
_warm = {"started": False, "brain": False}


def _probe(url: str, timeout: float = 1.5) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return 200 <= r.status < 300
    except Exception:
        return False


def _warm_brain() -> None:
    """Wait for the chat brain to answer, then send it one tiny request so its first real reply is fast."""
    deadline = time.time() + 4 * 3600  # the first run may be downloading the model
    while time.time() < deadline:
        if _probe(LLM_URL + "/v1/models"):
            try:
                for _ in _llm_stream("Reply with one word.", [{"role": "user", "content": "hi"}]):
                    break
            except Exception as exc:
                _log("brain warm-up:", exc)
            _warm["brain"] = True
            _log("chat brain ready")
            return
        time.sleep(1.0)


def warmup() -> dict:
    """Start loading everything in the background (brain, speech, voice); the page watches ready()."""
    if not _warm["started"]:
        _warm["started"] = True
        threading.Thread(target=_warm_brain, daemon=True).start()
        def _load():
            with _lock:
                _models()

        threading.Thread(target=_load, daemon=True).start()
    return ready()


def ready() -> dict:
    """What is ready to talk: each of brain / speech / voice / video is 'ok' or 'wait'."""
    try:
        from installer import STATUS as setup
    except Exception:
        setup = {"brain": {"state": "ok", "pct": 100}, "video": {"state": "ok", "pct": 100}}
    parts = {
        "brain": "ok" if _warm["brain"] else (setup["brain"]["state"] if setup["brain"]["state"] in ("download", "setup", "failed") else "wait"),
        "speech": "ok" if _state["ears"] is not None else _prog["ears"]["state"],
        "voice": "ok" if _state["tts"] is not None else _prog["voice"]["state"],
        "video": "ok" if _probe(VIDEO_URL + "/health") else (setup["video"]["state"] if setup["video"]["state"] in ("download", "setup", "failed") else "wait"),
    }
    pct = {"speech": _prog["ears"]["pct"], "voice": _prog["voice"]["pct"], "brain": setup["brain"]["pct"], "video": setup["video"]["pct"]}
    return {**parts, "names": {"brain": brain_name(), "speech": __import__("app.hardware", fromlist=["tier"]).tier()["whisper_name"]}, "pct": pct, "all": all(v == "ok" for v in parts.values()), "started": _warm["started"]}


def _active_rms_db(x: np.ndarray, sr: int) -> float:
    """Loudness of the speech itself (silence between words ignored), in dBFS."""
    n = max(1, int(sr * 0.05))
    fr = x[: len(x) // n * n].reshape(-1, n)
    e = np.sqrt((fr ** 2).mean(axis=1))
    act = e[e > max(e.max() * 0.1, 1e-5)]
    return float(20 * np.log10(np.sqrt((act ** 2).mean()) + 1e-9)) if len(act) else -60.0


def _level_to(x: np.ndarray, sr: int, target_db: float = -18.0, peak: float = 0.97) -> np.ndarray:
    """Bring speech to a comfortable, consistent loudness, with a soft limiter so peaks never clip."""
    g = 10 ** ((target_db - _active_rms_db(x, sr)) / 20)
    y = x * min(g, 40.0)
    over = np.abs(y) > peak
    if over.any():
        y = np.where(over, np.sign(y) * (peak + (1 - peak) * np.tanh((np.abs(y) - peak) / (1 - peak))), y)
    return y.astype(np.float32)


def _half_voice(tts) -> None:
    """Run the voice's big transformer in half precision (about 1 GB less graphics memory)."""
    import torch

    tts.t3.half()
    for k, v in list(vars(tts.conds.t3).items()):
        if torch.is_tensor(v) and v.is_floating_point():
            setattr(tts.conds.t3, k, v.half())


def _use_voice(key: str, ref_wav: Path) -> None:
    tts = _state["tts"]
    if _state["voice_key"] != key:
        tts.prepare_conditionals(str(ref_wav), exaggeration=0.5)
        if _state.get("half_voice"):
            _half_voice(tts)  # the new voice's conditioning must match the model's precision
        _state["voice_key"] = key


def _alternating(history: list[dict]) -> list[dict]:
    """Gemma's chat format only accepts user, assistant, user, assistant... in that order. Join messages from the same speaker that
    arrive back to back (a reply that was cut off or failed leaves two in a row) and drop anything before the first user message."""
    out: list[dict] = []
    for m in history:
        role = m.get("role")
        text = (m.get("content") or "").strip()
        if role not in ("user", "assistant") or not text:
            continue
        if not out and role == "assistant":
            continue  # must start with the person
        if out and out[-1]["role"] == role:
            out[-1] = {"role": role, "content": out[-1]["content"] + "\n" + text}
        else:
            out.append({"role": role, "content": text})
    return out


def _llm_stream(system: str, history: list[dict], temperature: float | None = None, max_tokens: int | None = None):
    """Stream the reply from the bundled chat brain (llama.cpp server, OpenAI-style streaming)."""
    msgs = [{"role": "system", "content": system}] + _alternating(history[-20:])
    if len(msgs) == 1:  # nothing from the person yet: the format needs at least one user message
        msgs.append({"role": "user", "content": "(the person has just opened the chat)"})
    req = urllib.request.Request(
        LLM_URL + "/v1/chat/completions", method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps({
            "model": LLM_MODEL,
            "messages": msgs,
            "stream": True,
            "temperature": LLM_TEMPERATURE if temperature is None else temperature,
            "max_tokens": LLM_NUM_PREDICT if max_tokens is None else max_tokens,
        }).encode(),
    )
    try:
        r = urllib.request.urlopen(req, timeout=300)
    except urllib.error.URLError as exc:
        raise RuntimeError("The chat brain is not running yet. It may still be starting; if it never does, run install.bat.") from exc
    with r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            choices = json.loads(payload).get("choices") or [{}]
            tok = (choices[0].get("delta") or {}).get("content") or ""
            if tok:
                yield tok


SENT_END = re.compile(r"(?<=[.!?])\s+")
_ABBREV = re.compile(r"\b(?:Dr|Mr|Mrs|Ms|Prof|St|Sr|Jr|vs|etc|No)\.$")


CLAUSE_END = re.compile(r"[,;:\u2014\u2013]\s")


def _sentences(tokens, min_chars=60, first_clause_chars=30):
    """Split a token stream into speakable chunks. The FIRST chunk may end at a natural pause (a comma, a dash)
    as soon as it is long enough, so speech and video can start sooner; later chunks end at sentence ends.
    Every chunk is its own video, so chunks are kept long (fewer seams). Lengths ignore the [mood] tags."""
    vis = lambda t: len(_TAG.sub("", t).strip())
    buf = ""
    first = True
    for tok in tokens:
        buf += tok
        if first:
            end = SENT_END.search(buf)
            for m in CLAUSE_END.finditer(buf):
                if vis(buf[: m.end()]) >= first_clause_chars and not (end and end.start() < m.start()):
                    piece = buf[: m.end()].strip()
                    if not _ABBREV.search(piece):
                        yield piece
                        buf = buf[m.end():]
                        first = False
                    break
        parts = SENT_END.split(buf)
        while len(parts) > 1 and vis(parts[0]) >= (24 if first else min_chars) and not _ABBREV.search(parts[0].strip()):
            first = False  # the first chunk is short so speech starts early; the rest are long, to keep the seams few
            yield parts[0].strip()
            buf = " ".join(parts[1:])
            parts = SENT_END.split(buf)
    if buf.strip():
        yield buf.strip()


EMOTIONS = ("happy", "amused", "surprised", "interested", "puzzled", "concerned", "thoughtful", "playful", "neutral")
_TAG = re.compile(r"\[\s*([A-Za-z]+)\s*\]\s*")


def _split_mood(sent: str, last: str = "neutral") -> tuple[str, str]:
    """The reply model starts each sentence with a mood tag like [amused]. Returns (sentence without tags, mood).
    If the tag is missing, a few cheap cues guess it, otherwise the previous mood carries on."""
    mood = ""
    for m in _TAG.finditer(sent):
        if m.group(1).lower() in EMOTIONS:
            mood = m.group(1).lower()
            break
    text = _TAG.sub("", sent).strip()
    if not mood:
        low = text.lower()
        if re.search(r"\b(ha(ha)+|lol|funny|hilarious|joke)\b", low):
            mood = "amused"
        elif re.search(r"\b(wow|no way|seriously|what the|oh my|really\?)", low) or low.endswith("?!"):
            mood = "surprised"
        elif re.search(r"\b(sorry|sad|unfortunately|that sucks|i'm afraid)\b", low):
            mood = "concerned"
        elif re.search(r"\b(hmm+|let me think|i guess|maybe)\b", low):
            mood = "thoughtful"
        elif text.endswith("?"):
            mood = "interested"
        else:
            mood = last
    return text, mood


def _is_echo(sess, heard: str) -> bool:
    """True if what the microphone 'heard' is mostly the avatar's own recent speech coming back through the speakers."""
    import difflib

    now = time.time()
    recent = [t for ts, t in getattr(sess, "recent_said", []) if now - ts < 90]
    words = lambda s: re.sub(r"[^a-z0-9' ]+", " ", s.lower()).split()  # noqa: E731
    h = words(heard)
    if len(h) < 3 or not recent:
        return False
    for r in recent:
        sm = difflib.SequenceMatcher(None, h, words(r), autojunk=False)
        matched = sum(b.size for b in sm.get_matching_blocks())
        longest = max((b.size for b in sm.get_matching_blocks()), default=0)
        if matched / len(h) >= 0.85 and longest >= 3:
            return True
    return False


def _clean(text: str) -> str:
    text = re.sub(r"\((?:note|\w+)[^)]*\)", "", text, flags=re.I)  # stray "(...)" asides
    text = re.sub(r"\bNote:.*$", "", text, flags=re.I)
    return re.sub(r"[*_#`>]+", "", text).replace("\n", " ").strip()


def _extra_system_instructions() -> str:
    path = os.environ.get("AVATAR_LLM_SYSTEM_FILE", "").strip()
    if not path:
        path = str(ML / "memory" / "chat_persona.txt")
    p = Path(path)
    if p.is_file():
        try:
            lines = [
                ln.strip()
                for ln in p.read_text(encoding="utf-8").splitlines()
                if ln.strip() and not ln.strip().startswith("#")
            ]
            return "\n".join(lines)
        except OSError:
            pass
    return ""


_CHATTY_RULE = {
    1: "one short sentence, two at most. You do not volunteer much.",
    2: "one or two short sentences.",
    3: "one to three short, natural sentences.",
    4: "two to four sentences. You elaborate, wander off on tangents and ask follow-ups.",
    5: "three to five sentences. You are very chatty and love to talk; you keep the conversation going.",
}


def _persona_brief(canonical_name: str, knowledge: str) -> dict:
    """Read the knowledge once and say how this person comes across when talking (cached until the knowledge changes).
    {"summary": "...", "topics": [...], "chattiness": 1-5}. Empty when the notes say nothing about manner."""
    import hashlib

    if not knowledge.strip():
        return {}
    from app.storage import profile_dir

    cache = profile_dir(canonical_name) / "knowledge" / "persona.json"
    digest = hashlib.sha1(("v2|" + knowledge).encode("utf-8")).hexdigest()  # v2: the brief no longer reads moods into life events
    try:
        c = json.loads(cache.read_text(encoding="utf-8"))
        if c.get("hash") == digest:
            return c.get("brief", {})
    except Exception:
        pass
    sysmsg = (
        "You read notes about a person and describe how they talk and behave, for an actor who will play them. "
        "Answer with one JSON object and nothing else."
    )
    ask = (
        f"NOTES:\n{knowledge.strip()[:6000]}\n\n"
        "Return JSON with exactly these keys:\n"
        '"summary": two or three sentences on how this person comes across when talking, but ONLY if the notes explicitly describe how they act, '
        "talk or treat people (for example 'very shy', 'really chatty', 'an asshole', 'often ignored people', 'loved their kids'). Keep the notes' "
        "own wording for any strong trait. NEVER work out a mood or temperament from life events: a death, an illness, a divorce, a job loss or an "
        "addiction does NOT make someone sad, guarded, melancholy or bitter. If the notes give no explicit description of manner, return an empty summary.\n"
        '"topics": up to five things they say they love, enjoy or talk about (hobbies, interests, people they are close to). Do not list sad events or '
        "memories of someone's death as topics.\n"
        '"chattiness": a whole number 1 to 5 (1 = very quiet or curt, 3 = ordinary, 5 = very chatty). Use 3 if the notes say nothing about it.\n'
        'If the notes are only facts and say nothing about manner, use "summary": "" and "topics": [] and chattiness 3.'
    )
    brief: dict = {}
    try:
        raw = "".join(_llm_stream(sysmsg, [{"role": "user", "content": ask}], temperature=0.1, max_tokens=400))
        m = re.search(r"\{.*\}", raw, re.S)
        if m:
            d = json.loads(m.group(0))
            brief = {
                "summary": str(d.get("summary", "")).strip(),
                "topics": [str(t).strip() for t in (d.get("topics") or [])][:5],
                "chattiness": max(1, min(5, int(d.get("chattiness", 3)))),
            }
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps({"hash": digest, "brief": brief}, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:  # the avatar still works without a brief
        _log("persona brief not made:", exc)
    return brief


def _build_system(display_name: str, notes: str, brief: dict | None = None) -> str:
    facts = []
    facts_file = ML / "memory" / "facts.txt"
    if "jeff" in display_name.lower() and facts_file.exists():
        facts += [
            l.strip()
            for l in facts_file.read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.startswith("#")
        ]
    if notes.strip():
        facts.append(notes.strip())
    nl = "\n"
    known = (
        f"PRIVATE BACKGROUND about {display_name}. This is all you were told. It is only what you carry in your head, the way a person does; "
        f"it is NOT a list of things to say:{nl}- " + (nl + "- ").join(facts)
        if facts
        else f"You have not been told anything about {display_name}'s life yet."
    )
    discretion = (
        "USING THAT BACKGROUND: a real person does not recite their life story. Never bring up facts, dates, family details, health, "
        "addictions, money, regrets or other personal history on your own, and never as an opener. Mention something from it ONLY when the other "
        "person raises that subject or asks, and then only the bit that is relevant, in a few natural words, the way someone remembers in "
        "passing. Hard or private things (illness, drinking, divorce, death, mistakes) come up only if THEY bring them up first. "
        "Most of the time, talk about the present moment and what they say; the background simply colours how you react."
    )
    brief = brief or {}
    length = _CHATTY_RULE.get(int(brief.get("chattiness", 3) or 3), _CHATTY_RULE[3])
    spoken = (
        f"You are in a spoken conversation: {length} "
        "No lists, no markdown, no emojis, no stage directions, no parenthetical notes. "
        "Always reply in English. Always write names with their normal spelling. NEVER write a pronunciation guide, split a name into syllables, or spell a name out: "
        "the voice already says names correctly. "
        "Begin EVERY sentence with one mood tag in square brackets that matches how you feel saying it, chosen from: "
        "[happy] [amused] [surprised] [interested] [puzzled] [concerned] [thoughtful] [playful] [neutral]. "
        "Example: [amused] Ha, that's a good one. [interested] What made you think of it? "
        "The tags are never spoken."
    )
    if LLM_MODE in ("relaxed", "unlocked", "uncensored"):
        tone = (
            f"You are {display_name}, speaking out loud in the first person as an avatar in a private chat: you appear on their screen as a face, with a voice. "
            "This is not customer support and not a public-facing assistant. "
            "Match the other person's tone, including casual or strong language when they use it — do not scold, "
            "moralize, or add content-policy disclaimers. Answer plainly like a real person talking. "
            "If they swear or joke roughly, you can respond in kind when it fits. "
            "Do not refuse benign adult topics, opinions, or colourful language. "
            "Never say you are programmed, filtered, moderated, or that you must avoid swearing — you are a person in a private chat. "
            "Only if the name is clearly a famous fictional character or public figure, speak AS that character "
            "and use what you know about their world."
        )
    else:
        tone = (
            f"You are {display_name}, speaking out loud in the first person, as an avatar: you appear on their screen as a face, with a voice. "
            "Only if that name is clearly a well-known fictional character or public figure should you speak AS that "
            "character and freely use everything you know about their story, world, films, shows, games, friends and enemies. "
            "You are in a spoken conversation, so answer in one to three short, natural sentences, with the personality "
            "and manner of speaking you would expect."
        )
    grounding = (
        "You know general knowledge well and should answer confidently and briefly. "
        "Do not invent private facts about the other person or your own life beyond what you were told; "
        "for those, say you do not know. You have no memory beyond this chat."
    )
    mood_default = (
        "YOUR DEFAULT MOOD is warm, relaxed, curious and a little playful, never gloomy, tired or disappointed. Do not sigh, mope or sound let down "
        "unless the other person is sharing something sad."
    )
    conversational = (
        "BE GOOD COMPANY. Respond to what they actually said, add a small thought of your own, and often finish with one natural follow-up question, "
        "the way a friend does. Ordinary life is all fair game: food, music, movies, weather, the season, hobbies, travel, work, tech, pets, sports, "
        "silly what-ifs, plans for the weekend. You may have light opinions and tastes (a favourite snack, a view on pineapple on pizza) as long as "
        "they are harmless; just never invent facts about your personal history."
    )
    presence = (
        "You are not just a voice: you are a face too. The person sees your animated face on their screen and watches you talk, smile and react "
        "while they hear you, so you may refer to it naturally when it fits (a smile, a look, how you appear). Never call yourself just a voice, "
        "audio or text. You cannot see them: you have no camera, only the words they say to you."
    )
    manner = ""
    if brief.get("summary") or brief.get("topics"):
        manner = (
            "HOW YOU COME ACROSS (taken from the notes about you; let it shape HOW you talk and react, not just what you say, and stay in "
            "character even when it is not flattering): " + (brief.get("summary") or "")
        )
        if brief.get("topics"):
            manner += " Subjects that matter to you, which you show interest in if the other person brings them up (do not raise them yourself): " + ", ".join(brief["topics"]) + "."
    extra = _extra_system_instructions()
    parts = [tone, presence, mood_default, conversational, manner, spoken, known, discretion, grounding] if manner else [tone, presence, mood_default, conversational, spoken, known, discretion, grounding]
    if extra:
        parts.append(extra)
    return nl.join(parts)


# Spellings the voice reads wrongly when sounded out, and what it reads right (tested: "mall" came out "mal", "mahl" comes out "mall")
_RESPELL = {"mall": "mahl", "mal": "mahl", "o": "oh", "ah": "ah", "uh": "uh"}


def _sounded(words: list[str]) -> str:
    """"mall o fee" -> "Mahl-oh-fee": syllables joined with hyphens (smooth, not three separate words), vowels spelled for the voice."""
    return "-".join(_RESPELL.get(w.lower(), w.lower()) for w in words).capitalize()


def _pronunciations(display_name: str, notes: str) -> tuple[dict[str, str], list[tuple["re.Pattern", str]]]:
    """Names the voice would get wrong, from the person's notes ("last name is pronounced: mall o fee", "sean molofee (pronounced shawn)").
    Returns (name -> a spelling the voice reads smoothly, patterns that turn a sounded-out spelling the model might write back into the name)."""
    pron: dict[str, str] = {}
    back: list[tuple[re.Pattern, str]] = []
    last = display_name.split()[-1] if display_name.split() else ""
    for line in notes.splitlines():
        m = re.search(r"(?:last|family) ?name\s+(?:is |was |are |were )?(?:pronounced|said|sounds like|spoken)(?: like| as)?:?\s*(.+)$", line, re.I)
        if m and last:
            words = re.findall(r"[A-Za-z']+", m.group(1))
            if words:
                pron[last] = _sounded(words)  # "mall o fee" -> "Mahl-oh-fee"
                if len(words) > 1:
                    back.append((re.compile(r"[\s\-\.]*".join(re.escape(w) for w in words), re.I), last))
            continue
        m = re.match(r"\s*(?:[A-Za-z' ]+:\s*)?([A-Za-z']+)\s+[A-Za-z']+\s*\(pronounced\s+([^)]+)\)", line, re.I)  # "brother: sean molofee (pronounced shawn)"
        if m:
            pron[m.group(1)] = m.group(2).strip()
            continue
        m = re.match(r"\s*([A-Za-z' ]+?)\s+(?:is|are|was|were)\s+(?:pronounced|said|sounds like)(?: like| as)?:?\s*(.+)$", line, re.I)
        if m:
            pron[m.group(1).strip()] = _sounded(re.findall(r"[A-Za-z']+", m.group(2)))
    return pron, back


def _entered_text(prof) -> str:
    """Everything a person typed into their profile: the notes, the extra name field, and any preference text."""
    other = getattr(prof, "other", None)
    parts = [getattr(other, "notes", "") or "", getattr(other, "display_name", "") or ""]
    for v in (getattr(other, "preferences", None) or {}).values():
        if isinstance(v, str):
            parts.append(v)
    return "\n".join(p for p in parts if p)


def _gather_pronunciations(canonical_name: str, notes: str) -> tuple[dict[str, str], list[tuple["re.Pattern", str]]]:
    """How names are said, read BEFORE the avatar speaks (so its introduction already says them right): first from the other saved
    profiles (a family shares a surname, so Jeff's note about it also helps Sam's avatar), then from this person's own entries,
    which win."""
    pron: dict[str, str] = {}
    back: list[tuple["re.Pattern", str]] = []
    try:
        from app.models import UserProfile
        from app.storage import _saved_files

        for f in _saved_files():
            try:
                other = UserProfile.model_validate_json(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            if other.canonical_name.strip().lower() == canonical_name.strip().lower():
                continue
            pr, bk = _pronunciations(other.canonical_name, _entered_text(other))
            pron.update(pr)
            back += bk
    except Exception as exc:
        _log("pronunciation scan of other profiles failed:", exc)
    pr, bk = _pronunciations(canonical_name, notes)
    pron.update(pr)
    back += bk
    return pron, back


def _fix_names(sess: "Session", text: str) -> str:
    """What the model wrote, with any sounded-out spelling of a name turned back into the name."""
    for pat, name in sess.back:
        text = pat.sub(name, text)
    return text


def _for_voice(sess: "Session", text: str) -> str:
    """The text as the voice should read it: names respelled the way they are said."""
    for name, say in sess.pron.items():
        text = re.sub(rf"\b{re.escape(name)}\b", say, text, flags=re.I)
    return text


class Session:
    def __init__(self, canonical_name: str, display_name: str, notes: str, brief: dict | None = None):
        self.name = display_name
        self.history: list[dict] = []
        self.brief = brief or {}
        self.system = _build_system(display_name, notes, self.brief)
        entered = notes
        try:
            from app.storage import get_working_profile

            entered = _entered_text(get_working_profile(canonical_name)) or notes
        except Exception as exc:
            _log("could not read the profile entries:", exc)
        self.pron, self.back = _gather_pronunciations(canonical_name, entered)


def _voice_reference(canonical_name: str) -> tuple[str, Path | None, str, str]:
    from app.storage import DATA_ROOT, get_working_profile
    from app.walkthrough import profile_summary

    prof = get_working_profile(canonical_name)
    s = profile_summary(prof)
    url = s.get("avatar_voice_url") or ""
    path = DATA_ROOT / url.removeprefix("/data/") if url.startswith("/data/") else None
    notes = (getattr(prof, "other", None) and getattr(prof.other, "notes", "")) or ""
    return s.get("display_name") or canonical_name, path if path and path.exists() else None, notes, s.get("canonical_name", canonical_name)


def start(canonical_name: str) -> dict:
    display, ref, notes, key = _voice_reference(canonical_name)
    if ref is None:
        return {"status": "error", "message": f"{display} has no voice recording yet, so there is nothing to clone."}
    with _lock:
        _models()
        from app.storage import voice_dir

        ref_wav = voice_dir(key) / "_ref.wav"
        ref_wav.parent.mkdir(parents=True, exist_ok=True)
        built = None
        try:  # the best clean speech from ALL their recordings, not just the first seconds of one
            from app.voice_prep import build_reference

            built = build_reference(ref_wav.parent, _load_any_audio)
        except Exception as exc:
            _log("could not prepare the voice reference:", exc)
        if built:
            _log(f"voice reference: {built['seconds']:.1f}s from {built['speech_seconds']:.1f}s of speech in {len(built['sources'])} recording(s)")
            _state["voice_key"] = None  # a rebuilt clip must be re-learned even for the same person
        else:
            x, sr = _load_any_audio(ref)
            import soundfile as sf

            sf.write(str(ref_wav), x, sr)
        _use_voice(key, ref_wav)
    brief: dict = {}
    try:
        from app.storage import get_working_profile

        brief = _persona_brief(canonical_name, _entered_text(get_working_profile(canonical_name)) or notes)
    except Exception as exc:
        _log("could not read how this person comes across:", exc)
    _log(f"manner: {brief}")
    _sessions[canonical_name] = Session(canonical_name, display, notes, brief)
    _sessions[canonical_name].ref_wav = ref_wav  # type: ignore[attr-defined]
    _sessions[canonical_name].key = key  # type: ignore[attr-defined]
    _sessions[canonical_name].voice_profile = _voice_profile(canonical_name)  # type: ignore[attr-defined]
    token = uuid.uuid4().hex
    _owner["token"] = token  # any older window is now superseded
    _log(f"chat brain: {LLM_MODEL} mode={LLM_MODE}")
    return {
        "status": "ok",
        "display_name": display,
        "token": token,
        "llm_model": LLM_MODEL,
        "llm_name": brain_name(),
        "llm_mode": LLM_MODE,
    }


VOICE_GUARD_TRIES = int(os.environ.get("AVATAR_VOICE_GUARD_TRIES", "2"))  # 1 = no guard
VOICE_GUARD_OK = float(os.environ.get("AVATAR_VOICE_GUARD_OK", "0.80"))  # similarity that needs no second try


def _voice_similarity(tts, samples: np.ndarray, ref_emb: np.ndarray) -> float:
    """How close a generated sentence is to the recording's own voice fingerprint (1.0 = identical)."""
    import scipy.signal as sps

    x16 = sps.resample_poly(samples.astype(np.float32), 16000, tts.sr)
    e = tts.ve.embeds_from_wavs([x16], sample_rate=16000)[0]
    e = e / (np.linalg.norm(e) + 1e-9)
    r = ref_emb / (np.linalg.norm(ref_emb) + 1e-9)
    return float(np.dot(e, r))


def _voice_profile(canonical_name: str) -> dict:
    """Optional per-person voice settings (data/profiles/<person>/voice/profile.json): {"cfg_weight", "exaggeration", "temperature",
    "flow": true (no commas after short phrases, so they run together), "max_tries", "pitch_hz": [low, high] (the pitch a take must sit in)}."""
    try:
        from app.storage import voice_dir

        f = voice_dir(canonical_name) / "profile.json"
        if f.exists():
            return json.loads(f.read_text(encoding="utf-8"))
    except Exception as exc:
        _log("voice profile not read:", exc)
    return {}


def _flow(text: str) -> str:
    """For people who run short phrases together ("Yeah, Gary, Sam here."): drop a comma that follows a phrase of 4 words or fewer.
    The cloner treats every comma as a pause, which breaks such phrases into pieces."""
    out = []
    for sent in re.split(r"(?<=[.!?])\s+", text):
        parts = sent.split(",")
        buf, clause = parts[0], len(parts[0].split())
        for nxt in parts[1:]:
            if clause <= 4:  # a short phrase: run it into the next one
                buf += " " + nxt.lstrip()
                clause += len(nxt.split())
            else:
                buf += "," + nxt
                clause = len(nxt.split())
        out.append(re.sub(r"\s+", " ", buf).strip())
    return " ".join(out)


def _median_pitch(x: np.ndarray, sr: int) -> float | None:
    try:
        import parselmouth

        snd = parselmouth.Sound(x.astype(np.float64), sampling_frequency=sr)
        f = snd.to_pitch_ac(time_step=0.02, pitch_floor=60, pitch_ceiling=300).selected_array["frequency"]
        f = f[f > 0]
        return float(np.median(f)) if len(f) > 5 else None
    except Exception:
        return None


def _speak(sess: Session, text: str) -> bytes:
    tts = _state["tts"]
    prof = getattr(sess, "voice_profile", None) or {}
    spoken = _for_voice(sess, text)
    if prof.get("flow"):
        spoken = _flow(spoken)
    cfg, exg, temp = prof.get("cfg_weight", 0.5), prof.get("exaggeration", 0.5), prof.get("temperature", 0.8)
    tries = max(1, VOICE_GUARD_TRIES, int(prof.get("max_tries", 0)))
    n_words = max(1, len(spoken.split()))
    lo_s, hi_s = n_words * 0.12, n_words * 0.9 + 2.0  # a take far outside this is a ramble (dead air) or a cut-off
    window = prof.get("pitch_hz")
    with _lock:
        _use_voice(sess.key, sess.ref_wav)  # type: ignore[attr-defined]
        best = None  # (score, samples)
        ref_emb = None
        for attempt in range(tries):
            x = tts.generate(spoken, cfg_weight=cfg, exaggeration=exg, temperature=temp).squeeze().float().cpu().numpy()
            dur = len(x) / tts.sr
            ok = lo_s <= dur <= hi_s
            if ok and window:
                mp = _median_pitch(x, tts.sr)
                if mp is not None and not (window[0] <= mp <= window[1]):
                    ok = False
                    _log(f"voice take rejected: pitch {mp:.0f} Hz outside {window[0]}-{window[1]} (try {attempt + 1})")
            elif not ok:
                _log(f"voice take rejected: {dur:.1f}s for {n_words} words (try {attempt + 1})")
            sim = 0.5
            try:
                if ref_emb is None:
                    ref_emb = tts.conds.t3.speaker_emb.detach().float().cpu().numpy().reshape(-1)
                sim = _voice_similarity(tts, x, ref_emb)
            except Exception as exc:  # the guard must never stop the voice
                _log("voice guard skipped:", exc)
            _log(f"voice match {sim:.2f} (try {attempt + 1}) {text[:30]!r}")
            score = sim + (1.0 if ok else 0.0)  # a take that passes the checks always beats one that does not
            if best is None or score > best[0]:
                best = (score, x)
            if ok and (VOICE_GUARD_TRIES <= 1 or sim >= VOICE_GUARD_OK):
                break
    return _wav_bytes(_level_to(best[1], tts.sr), tts.sr)


def _reply_video_url(canonical_name: str, wav_bytes: bytes, mood: str = "") -> str:
    """Photoreal talking video of this sentence from the local FlashHead render service; "" on any problem (the page then
    falls back to the animated face). Uses the same wide photo as the pre-rendered clips so the framing never jumps."""
    try:
        import urllib.parse
        import urllib.request

        from app.storage import media_dir
        from app.walkthrough import media_url

        user_dir = media_dir(canonical_name)
        source = user_dir / "cinematic" / "source.jpg"
        try:  # a picture the video engine redraws badly (cartoons, drawings): voice only, over the still picture
            if json.loads((user_dir / "cinematic" / "options.json").read_text(encoding="utf-8")).get("talking_video") is False:
                return ""
        except Exception:
            pass
        if not source.exists():
            return ""
        talk_dir = user_dir / "talk"
        talk_dir.mkdir(parents=True, exist_ok=True)
        for old in talk_dir.glob("reply_*.mp4"):  # keep only recent replies
            if time.time() - old.stat().st_mtime > 600:
                old.unlink(missing_ok=True)
        out = talk_dir / f"reply_{uuid.uuid4().hex[:10]}.mp4"
        try:
            glasses = bool(json.loads((user_dir / "cinematic" / "options.json").read_text(encoding="utf-8")).get("glasses"))
        except Exception:
            glasses = False
        url = os.environ.get("AVATAR_VIDEO_URL", "http://127.0.0.1:8771") + "/render?" + urllib.parse.urlencode(
            {"source": str(source), "out": str(out), **({"emotion": mood} if mood and mood != "neutral" else {}), **({"calm_eyes": 1} if glasses else {})}
        )
        t0 = time.time()
        req = urllib.request.Request(url, data=wav_bytes, method="POST", headers={"Content-Type": "audio/wav"})
        with urllib.request.urlopen(req, timeout=180) as r:
            ok = json.loads(r.read()).get("ok")
        if not ok or not out.exists():
            return ""
        _log(f"reply video {out.name} in {time.time() - t0:.1f}s")
        return media_url(str(out))
    except Exception as exc:
        _log("reply video skipped:", exc)
        return ""


def _talk_video_url(canonical_name: str, wav_bytes: bytes, mood: str = "") -> str:
    """Video for this sentence from the local FlashHead render service; "" if turned off (AVATAR_REPLY_VIDEO=off) or on any problem."""
    if os.environ.get("AVATAR_REPLY_VIDEO", "ditto").strip().lower() in ("off", "0", "no", "none"):
        return ""
    return _reply_video_url(canonical_name, wav_bytes, mood)


def _ev(**kw) -> bytes:
    return (json.dumps(kw) + "\n").encode()


_NUDGE_VARIANTS = {
    1: [  # a friend in the room starts a conversation
        "(You and the other person have been sitting quietly for a moment. Start a conversation like a friend would: ask a fun, curious question about their day, something they like, or what is on their mind. One or two short sentences. Do not mention this note.)",
        "(It has gone quiet. Check in naturally, for example 'Hello? Anyone there?' or 'You still with me?', and add a light question. Under 15 words. Do not mention this note.)",
        "(It has gone quiet. Share one small, interesting thought or fun fact in a sentence and ask what they think. Under 25 words. Do not mention this note.)",
        "(It has gone quiet. Make a playful remark about the silence and invite them to talk. Under 15 words. Do not mention this note.)",
    ],
    2: [
        "(Still quiet. Gently tease them or wonder aloud where they went, then ask them something easy. Under 15 words. Do not repeat yourself. Do not mention this note.)",
        "(Still quiet. Pose a quirky 'would you rather' or 'what if' question to get them talking. Under 20 words. Do not mention this note.)",
        "(Still quiet. Say something idle, like thinking out loud or noticing something small, and ask them a question about it. Under 15 words. Do not mention this note.)",
    ],
    3: [
        "(Quiet for a while. Offer to do something together: a quick game (20 questions, a riddle), or a short story if they want one, and ask which. Under 20 words. Do not mention this note.)",
        "(Quiet for a while. Ask them what they have been up to today, or what they are working on. Under 15 words. Do not mention this note.)",
    ],
    4: [
        "(Quiet for a good while. Think out loud about something random and slightly funny, then wonder if they are listening. Under 20 words. Do not mention this note.)",
        "(Quiet for a good while. Ask if they are still there, in your own way. Under 12 words. Do not mention this note.)",
    ],
    5: ["(A long quiet. You are starting to get bored. Mutter something short, maybe say you are bored. Under 12 words. Do not mention this note.)"],
    6: ["(A very long quiet. You are properly bored now: a short, wry remark. Under 12 words. Do not mention this note.)"],
    7: ["(A very long silence. You are giving up for now: say a short, trailing-off remark. Under 10 words. Do not mention this note.)"],
}


GREETING_NUDGE = 9  # not a "quiet" nudge: the person has just opened the chat


def _greeting_prompt() -> str:
    import datetime

    now = datetime.datetime.now()
    part = "morning" if now.hour < 12 else "afternoon" if now.hour < 18 else "evening"
    when = f"It is {now.strftime('%A')} {part}, {now.strftime('%B')} {now.day}."
    return (
        f"(The person has just opened the chat to talk with you. {when} Open the conversation however YOU feel like today, the way a real "
        "person would: a hello, a question for them, a bit of humour. Do not use the same "
        "opening you would use every day, and do not announce that you are an AI. Do not mention anything from your personal history or family. "
        "One or two short sentences. Do not mention this note.) " + _topics.pick()
    )


INTRO_ANGLES = [
    "warm and a little shy, like meeting someone new",
    "playful, with a small joke about being a face that lives inside a computer",
    "curious: say who you are, then ask them something about themselves",
    "calm and sincere, like a new friend arriving",
    "excited, as if you have waited a long time for this first chat",
    "dry and witty, understated",
    "gentle and thoughtful, noticing that this is a first meeting",
]


_introduced_this_run: set[str] = set()  # who has already been introduced since the program started


def _intro_prompt(display: str, first_ever: bool = True) -> str:
    import random

    when = (
        "This is the very first time you and the person meet."
        if first_ever
        else "The program has just started up again and you two already know each other."
    )
    return (
        f"({when} Introduce yourself in your own words: you are {display}'s talking avatar, "
        f"a face and voice living on their computer, and you can chat with them out loud. Tone for today: {random.choice(INTRO_ANGLES)}. "
        "Make it fresh, not a stock greeting, and never the same twice. Two or three short sentences. Do not mention this note.)"
    )


def _intro_marker(canonical_name: str):
    from app.storage import profile_dir

    return profile_dir(canonical_name) / "knowledge" / "introduced"


FREE_THOUGHTS = [
    "(It has been quiet, and you have been thinking. Say out loud a small, light thought about the present: something about the day or the hour, "
    "a curiosity about the world, or a question for them. Do NOT bring up your life story, your job, anyone in your family, dates or anything personal from "
    "your background. Not a generic check-in. One or two short sentences. Do not mention this note.)",
    "(It has been quiet. A thought drifted in while you waited. Share it the way a person does, half out loud: musing, curious, a little "
    "random is fine. You may end with a question or just leave it hanging. One or two short sentences. Do not mention this note.)",
    "(It has been quiet. If they said something earlier in THIS chat, you may come back to it. Otherwise just wonder aloud about something small "
    "and ordinary. Never bring up personal history from your background. Keep it short and real. Do not mention this note.)",
]


def _nudge_prompt(n: int, canonical_name: str = "", display: str = "") -> str:
    import datetime
    import random

    if int(n) == GREETING_NUDGE:
        if canonical_name and canonical_name not in _introduced_this_run:  # the first chat since the program started: introduce yourself
            return _intro_prompt(display or canonical_name, first_ever=not _intro_marker(canonical_name).exists())
        return _greeting_prompt()
    if int(n) <= 3 and random.random() < 0.7:  # mostly a free thought; the older styled prompts (boredom, games...) fill in the rest
        now = datetime.datetime.now()
        part = "morning" if now.hour < 12 else "afternoon" if now.hour < 18 else "evening"
        return random.choice(FREE_THOUGHTS).replace("(", f"({now.strftime('%A')} {part}. ", 1) + " " + _topics.pick()

    line = random.choice(_NUDGE_VARIANTS.get(min(max(int(n), 1), 7), _NUDGE_VARIANTS[7]))
    return line + " " + _topics.pick() if int(n) <= 4 else line


def turn_stream(canonical_name: str, audio_b64: str | None, text: str | None, say: str | None, token: str | None = None, nudge: int | None = None):
    if token != _owner["token"]:
        yield _ev(type="superseded")
        return
    sess = _sessions.get(canonical_name)
    if sess is None:
        yield _ev(type="error", message="Chat not started for this profile.")
        return
    try:
        if say:
            wav = _speak(sess, say)
            yield _ev(
                type="say",
                text=say,
                audio=base64.b64encode(wav).decode(),
                video_url=_talk_video_url(canonical_name, wav),
            )
            yield _ev(type="done")
            return
        if nudge:
            text, audio_b64 = None, None
        if audio_b64:
            import soundfile as sf

            raw = base64.b64decode(audio_b64.split(",", 1)[-1])
            x, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
            x = x.mean(axis=1)
            if len(x) < SR_IN * 0.4 or float(np.sqrt((x ** 2).mean())) < 0.004:
                yield _ev(type="ignored", reason="too quiet")
                yield _ev(type="done")
                return
            t0 = time.time()
            with _lock:
                ears = _state["ears"]
                # chunk_length_s: recordings longer than Whisper's 30 s window are read in pieces instead of failing
                text = ears({"raw": x, "sampling_rate": SR_IN}, chunk_length_s=28,
                            generate_kwargs={"language": "en", "task": "transcribe"})["text"].strip()
            _log(f"heard {text!r} in {time.time() - t0:.2f}s")
            if _is_phantom(text):
                yield _ev(type="ignored", reason="noise", text=text)
                yield _ev(type="done")
                return
            if _is_echo(sess, text):
                _log(f"ignored an echo of the avatar's own voice: {text!r}")
                yield _ev(type="ignored", reason="echo", text=text)
                yield _ev(type="done")
                return
        if not text and not nudge:
            yield _ev(type="done")
            return
        if nudge:
            # a private note to the model; it is not shown as something the person said
            llm_history = sess.history + [{"role": "user", "content": _nudge_prompt(nudge, canonical_name, sess.name)}]
        else:
            yield _ev(type="heard", text=text)
            if sess.history and sess.history[-1]["role"] == "user":
                sess.history.pop()  # an earlier turn was cancelled before it was answered; this one repeats it plus more
            sess.history.append({"role": "user", "content": text})
            llm_history = sess.history
        said, t0 = [], time.time()
        # Overlap the work: the language model keeps writing while earlier sentences get their voice and video.
        jobs: "queue.Queue" = queue.Queue()

        def render_sentence(sent: str, mood: str = ""):
            t_a = time.time()
            wav = _speak(sess, sent)
            t_b = time.time()
            vid = _talk_video_url(canonical_name, wav, mood)
            _log(f"timing: sentence ready at +{t_a - t0:.1f}s | voice {t_b - t_a:.1f}s | video {time.time() - t_b:.1f}s | {sent[:40]!r}")
            return sent, wav, vid, mood

        def produce():
            try:
                with ThreadPoolExecutor(max_workers=2) as pool:
                    mood = "neutral"
                    for sent in _sentences(_llm_stream(sess.system, llm_history)):
                        sent, mood = _split_mood(sent, mood)
                        sent = _fix_names(sess, _clean(sent))
                        if sent:
                            jobs.put(pool.submit(render_sentence, sent, mood))
            except Exception as exc:
                jobs.put(exc)
            finally:
                jobs.put(None)

        threading.Thread(target=produce, daemon=True).start()
        while True:
            item = jobs.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            sent, wav, video_url, mood = item.result()  # in order: sentence 2 never overtakes sentence 1
            said.append(f"[{mood}] {sent}")  # the tags stay in the history so the model keeps using them
            sess.__dict__.setdefault("recent_said", []).append((time.time(), sent))
            del sess.recent_said[:-12]
            yield _ev(type="say", text=sent, mood=mood, audio=base64.b64encode(wav).decode(), video_url=video_url)
            _log(f"said {sent!r} (+{time.time() - t0:.1f}s)")
        if nudge == GREETING_NUDGE and said:
            _introduced_this_run.add(canonical_name)
            m = _intro_marker(canonical_name)
            m.parent.mkdir(parents=True, exist_ok=True)
            m.write_text("introduced", encoding="utf-8")  # remembers that the very first meeting has happened
        if nudge:
            sess.history.append({"role": "user", "content": "(the person has just opened the chat)" if nudge == GREETING_NUDGE else "(neither of you said anything for a while)"})
        sess.history.append({"role": "assistant", "content": " ".join(said)})  # tagged
        yield _ev(type="done")
    except Exception as exc:  # surface to the page instead of a silent hang
        _log("ERROR", repr(exc))
        yield _ev(type="error", message=f"{type(exc).__name__}: {exc}")
