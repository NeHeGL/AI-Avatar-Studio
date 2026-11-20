"""Spoken conversation with a profile's avatar: speech-to-text -> LLM -> cloned-voice TTS, streamed per sentence.

Models load once, on first use (GPU). The LLM backend is Ollama for now (override with AVATAR_LLM / AVATAR_LLM_URL).
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
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ML = ROOT / "ml"
OUT = ML / "out"
if str(ML) not in sys.path:
    sys.path.insert(0, str(ML))
LLM_MODEL = os.environ.get("AVATAR_LLM", "gemma-opus-cuda:q4km")
LLM_URL = os.environ.get("AVATAR_LLM_URL", "http://127.0.0.1:11434")
# relaxed = casual private chat, no scolding about language; safe = tighter assistant tone
LLM_MODE = os.environ.get("AVATAR_LLM_MODE", "relaxed").strip().lower()
LLM_TEMPERATURE = float(os.environ.get("AVATAR_LLM_TEMPERATURE", "0.85" if LLM_MODE == "relaxed" else "0.7"))
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


def _models():
    """Load Whisper + Chatterbox once."""
    import torch

    if _state["ears"] is None:
        from transformers import pipeline

        t0 = time.time()
        _state["ears"] = pipeline(
            "automatic-speech-recognition", model="openai/whisper-large-v3-turbo",
            torch_dtype=torch.float16, device="cuda",
        )
        _log(f"whisper ready in {time.time() - t0:.1f}s")
    if _state["tts"] is None:
        from chatterbox.tts import ChatterboxTTS

        t0 = time.time()
        _state["tts"] = ChatterboxTTS.from_pretrained(device="cuda")
        _log(f"chatterbox ready in {time.time() - t0:.1f}s")
    return _state["ears"], _state["tts"]


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


def _use_voice(key: str, ref_wav: Path) -> None:
    tts = _state["tts"]
    if _state["voice_key"] != key:
        tts.prepare_conditionals(str(ref_wav), exaggeration=0.5)
        _state["voice_key"] = key


def _ollama_stream(system: str, history: list[dict]):
    msgs = [{"role": "system", "content": system}] + history[-20:]
    req = urllib.request.Request(
        LLM_URL + "/api/chat", method="POST",
        data=json.dumps({
            "model": LLM_MODEL,
            "messages": msgs,
            "stream": True,
            "keep_alive": "30m",
            "options": {"temperature": LLM_TEMPERATURE, "num_predict": LLM_NUM_PREDICT},
        }).encode(),
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        for line in r:
            if line.strip():
                tok = json.loads(line).get("message", {}).get("content", "")
                if tok:
                    yield tok


SENT_END = re.compile(r"(?<=[.!?])\s+")
_ABBREV = re.compile(r"\b(?:Dr|Mr|Mrs|Ms|Prof|St|Sr|Jr|vs|etc|No)\.$")


CLAUSE_END = re.compile(r"[,;:\u2014\u2013]\s")


def _sentences(tokens, min_chars=24, first_clause_chars=18):
    """Split a token stream into speakable chunks. The FIRST chunk may end at a natural pause (a comma, a dash)
    as soon as it is long enough, so speech and video can start sooner; later chunks end at sentence ends."""
    buf = ""
    first = True
    for tok in tokens:
        buf += tok
        if first:
            end = SENT_END.search(buf)
            for m in CLAUSE_END.finditer(buf):
                if m.end() >= first_clause_chars and not (end and end.start() < m.start()):
                    piece = buf[: m.end()].strip()
                    if not _ABBREV.search(piece):
                        yield piece
                        buf = buf[m.end():]
                        first = False
                    break
        parts = SENT_END.split(buf)
        if len(parts) > 1 and len(parts[0].strip()) >= min_chars:
            first = False
        while len(parts) > 1 and len(parts[0].strip()) >= min_chars and not _ABBREV.search(parts[0].strip()):
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


def _build_system(display_name: str, notes: str) -> str:
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
        f"What you know about {display_name} (this is ALL you know):{nl}- " + (nl + "- ").join(facts)
        if facts
        else f"You have not been told anything about {display_name}'s life yet."
    )
    spoken = (
        "You are in a spoken conversation: one to three short, natural sentences. "
        "No lists, no markdown, no emojis, no stage directions, no parenthetical notes. "
        "Always write names with their normal spelling. NEVER write a pronunciation guide, split a name into syllables, or spell a name out: "
        "the voice already says names correctly. "
        "Begin EVERY sentence with one mood tag in square brackets that matches how you feel saying it, chosen from: "
        "[happy] [amused] [surprised] [interested] [puzzled] [concerned] [thoughtful] [playful] [neutral]. "
        "Example: [amused] Ha, that's a good one. [interested] What made you think of it? "
        "Your face acts out the tag, so use it: smile when something is funny or nice, look surprised when shocked, "
        "look interested when curious, puzzled when confused, playful when joking. The tags are never spoken."
    )
    if LLM_MODE in ("relaxed", "unlocked", "uncensored"):
        tone = (
            f"You are {display_name}, speaking out loud in the first person as a voice avatar in a private chat. "
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
            f"You are {display_name}, speaking out loud in the first person, as a voice avatar. "
            "Only if that name is clearly a well-known fictional character or public figure should you speak AS that "
            "character and freely use everything you know about their story, world, films, shows, games, friends and enemies. "
            "You are in a spoken conversation, so answer in one to three short, natural sentences, with the personality "
            "and manner of speaking you would expect."
        )
    grounding = (
        "You know general knowledge well and should answer confidently and briefly. "
        "Do not invent private facts about the other person or your own life beyond what you were told; "
        "for those, say you do not know. You have no senses and no memory beyond this chat."
    )
    extra = _extra_system_instructions()
    parts = [tone, spoken, known, grounding]
    if extra:
        parts.append(extra)
    return nl.join(parts)


def _pronunciations(display_name: str, notes: str) -> tuple[dict[str, str], list[tuple["re.Pattern", str]]]:
    """Names the voice would get wrong, from the person's notes ("last name is pronounced: mall o fee", "sean molofee (pronounced shawn)").
    Returns (name -> a spelling the voice reads smoothly, patterns that turn a sounded-out spelling the model might write back into the name)."""
    pron: dict[str, str] = {}
    back: list[tuple[re.Pattern, str]] = []
    last = display_name.split()[-1] if display_name.split() else ""
    for line in notes.splitlines():
        m = re.search(r"last name is pronounced:?\s*(.+)$", line, re.I)
        if m and last:
            words = re.findall(r"[A-Za-z']+", m.group(1))
            if words:
                pron[last] = "".join(words).capitalize()  # "mall o fee" -> "Mallofee": one word, said smoothly
                if len(words) > 1:
                    back.append((re.compile(r"[\s\-\.]*".join(re.escape(w) for w in words), re.I), last))
            continue
        m = re.match(r"\s*(?:[A-Za-z' ]+:\s*)?([A-Za-z']+)\s+[A-Za-z']+\s*\(pronounced\s+([^)]+)\)", line, re.I)  # "brother: sean molofee (pronounced shawn)"
        if m:
            pron[m.group(1)] = m.group(2).strip()
            continue
        m = re.match(r"\s*([A-Za-z' ]+?)\s+(?:is|are)\s+pronounced:?\s*(.+)$", line, re.I)
        if m:
            pron[m.group(1).strip()] = "".join(re.findall(r"[A-Za-z']+", m.group(2))).capitalize()
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
    def __init__(self, canonical_name: str, display_name: str, notes: str):
        self.name = display_name
        self.history: list[dict] = []
        self.system = _build_system(display_name, notes)
        self.pron, self.back = _pronunciations(display_name, notes)


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
        OUT.mkdir(exist_ok=True)
        x, sr = _load_any_audio(ref)
        ref_wav = OUT / f"_ref_{re.sub(r'[^a-z0-9]+', '_', key.lower())}.wav"
        import soundfile as sf

        sf.write(str(ref_wav), x, sr)
        _use_voice(key, ref_wav)
    _sessions[canonical_name] = Session(canonical_name, display, notes)
    _sessions[canonical_name].ref_wav = ref_wav  # type: ignore[attr-defined]
    _sessions[canonical_name].key = key  # type: ignore[attr-defined]
    token = uuid.uuid4().hex
    _owner["token"] = token  # any older window is now superseded
    _log(f"chat brain: {LLM_MODEL} mode={LLM_MODE}")
    return {
        "status": "ok",
        "display_name": display,
        "token": token,
        "llm_model": LLM_MODEL,
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


def _speak(sess: Session, text: str) -> bytes:
    tts = _state["tts"]
    with _lock:
        _use_voice(sess.key, sess.ref_wav)  # type: ignore[attr-defined]
        best = None
        ref_emb = None
        for attempt in range(max(1, VOICE_GUARD_TRIES)):
            x = tts.generate(_for_voice(sess, text), cfg_weight=0.5, exaggeration=0.5).squeeze().cpu().numpy()
            if VOICE_GUARD_TRIES <= 1:
                best = (1.0, x)
                break
            try:
                if ref_emb is None:
                    ref_emb = tts.conds.t3.speaker_emb.detach().float().cpu().numpy().reshape(-1)
                sim = _voice_similarity(tts, x, ref_emb)
            except Exception as exc:  # the guard must never stop the voice
                _log("voice guard skipped:", exc)
                best = (1.0, x)
                break
            _log(f"voice match {sim:.2f} (try {attempt + 1}) {text[:30]!r}")
            if best is None or sim > best[0]:
                best = (sim, x)
            if sim >= VOICE_GUARD_OK:
                break
    return _wav_bytes(_level_to(best[1], tts.sr), tts.sr)


def _reply_video_url(canonical_name: str, wav_bytes: bytes, mood: str = "") -> str:
    """Photoreal talking video of this sentence from the local FlashHead render service; "" on any problem (the page then
    falls back to the animated face). Uses the same wide photo as the pre-rendered clips so the framing never jumps."""
    try:
        import urllib.parse
        import urllib.request

        from app.capture import _user_media_dir
        from app.storage import MEDIA_DIR
        from app.walkthrough import media_url

        user_dir = MEDIA_DIR / _user_media_dir(canonical_name)
        source = user_dir / "cinematic" / "source.jpg"
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
        "person would: a hello, a thought you had, a question, something about the day or the time, a bit of humour. Do not use the same "
        "opening you would use every day, and do not announce that you are an AI. One or two short sentences. Do not mention this note.)"
    )


FREE_THOUGHTS = [
    "(It has been quiet, and you have been thinking. Say out loud what actually occurred to you: something from what you two talked about, a "
    "memory, a small wonder about the world, an observation about the day or the hour. Not a generic check-in. One or two short sentences. "
    "Do not mention this note.)",
    "(It has been quiet. A thought drifted in while you waited. Share it the way a person does, half out loud: musing, curious, a little "
    "random is fine. You may end with a question or just leave it hanging. One or two short sentences. Do not mention this note.)",
    "(It has been quiet. Something you said earlier, or something they said, has been on your mind. Come back to it naturally, or move to "
    "whatever your mind has wandered to. Keep it short and real. Do not mention this note.)",
]


def _nudge_prompt(n: int) -> str:
    import datetime
    import random

    if int(n) == GREETING_NUDGE:
        return _greeting_prompt()
    if int(n) <= 3 and random.random() < 0.7:  # mostly a free thought; the older styled prompts (boredom, games...) fill in the rest
        now = datetime.datetime.now()
        part = "morning" if now.hour < 12 else "afternoon" if now.hour < 18 else "evening"
        return random.choice(FREE_THOUGHTS).replace("(", f"({now.strftime('%A')} {part}. ", 1)

    return random.choice(_NUDGE_VARIANTS.get(min(max(int(n), 1), 7), _NUDGE_VARIANTS[7]))


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
            llm_history = sess.history + [{"role": "user", "content": _nudge_prompt(nudge)}]
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
                    for sent in _sentences(_ollama_stream(sess.system, llm_history)):
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
        if nudge:
            sess.history.append({"role": "user", "content": "(the person has just opened the chat)" if nudge == GREETING_NUDGE else "(neither of you said anything for a while)"})
        sess.history.append({"role": "assistant", "content": " ".join(said)})  # tagged
        yield _ev(type="done")
    except Exception as exc:  # surface to the page instead of a silent hang
        _log("ERROR", repr(exc))
        yield _ev(type="error", message=f"{type(exc).__name__}: {exc}")
