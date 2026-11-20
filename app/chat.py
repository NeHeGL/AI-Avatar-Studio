"""Spoken conversation with a profile's avatar: speech-to-text -> LLM -> cloned-voice TTS, streamed per sentence.

Models load once, on first use (GPU). The LLM backend is Ollama for now (override with AVATAR_LLM / AVATAR_LLM_URL).
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import sys
import threading
import time
import uuid
import urllib.request
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

_lock = threading.Lock()  # one GPU job at a time
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


def _sentences(tokens, min_chars=24):
    buf = ""
    for tok in tokens:
        buf += tok
        parts = SENT_END.split(buf)
        while len(parts) > 1 and len(parts[0].strip()) >= min_chars and not _ABBREV.search(parts[0].strip()):
            yield parts[0].strip()
            buf = " ".join(parts[1:])
            parts = SENT_END.split(buf)
    if buf.strip():
        yield buf.strip()


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
        "No lists, no markdown, no emojis, no stage directions, no parenthetical notes."
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


class Session:
    def __init__(self, canonical_name: str, display_name: str, notes: str):
        self.name = display_name
        self.history: list[dict] = []
        self.system = _build_system(display_name, notes)


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


def _speak(sess: Session, text: str) -> bytes:
    tts = _state["tts"]
    with _lock:
        _use_voice(sess.key, sess.ref_wav)  # type: ignore[attr-defined]
        wav = tts.generate(text, cfg_weight=0.5, exaggeration=0.5)
    return _wav_bytes(wav.squeeze().cpu().numpy(), tts.sr)


def _ditto_video_url(canonical_name: str, wav_bytes: bytes) -> str:
    """Photoreal talking video of this sentence from the local Ditto render service; "" on any problem (the page then
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
            if time.time() - old.stat().st_mtime > 3600:
                old.unlink(missing_ok=True)
        out = talk_dir / f"reply_{uuid.uuid4().hex[:10]}.mp4"
        url = os.environ.get("AVATAR_DITTO_URL", "http://127.0.0.1:8771") + "/render?" + urllib.parse.urlencode({"source": str(source), "out": str(out)})
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


def _talk_video_url(canonical_name: str, wav_bytes: bytes) -> str:
    """Video for this sentence; empty string if disabled or on failure.
    AVATAR_REPLY_VIDEO=ditto -> local Ditto service (fast, photoreal); AVATAR_PHOTOREAL_VIDEO=1 -> InfiniteTalk (slow)."""
    if os.environ.get("AVATAR_REPLY_VIDEO", "").strip().lower() == "ditto":
        return _ditto_video_url(canonical_name, wav_bytes)
    if os.environ.get("AVATAR_PHOTOREAL_VIDEO", "").strip() != "1":
        return ""
    try:
        from avatar_video import portrait_for_profile, render_talk_video
        from app.capture import _user_media_dir
        from app.storage import MEDIA_DIR
        from app.walkthrough import media_url

        scan_dir = MEDIA_DIR / _user_media_dir(canonical_name) / "scan"
        portrait = portrait_for_profile(scan_dir)
        talk_dir = MEDIA_DIR / _user_media_dir(canonical_name) / "talk"
        talk_dir.mkdir(parents=True, exist_ok=True)
        tag = uuid.uuid4().hex[:10]
        wav_path = talk_dir / f"clip_{tag}.wav"
        mp4_path = talk_dir / f"clip_{tag}.mp4"
        wav_path.write_bytes(wav_bytes)
        t0 = time.time()
        render_talk_video(portrait, wav_path, mp4_path)
        _log(f"talk video {mp4_path.name} in {time.time() - t0:.1f}s")
        return media_url(str(mp4_path))
    except Exception as exc:
        _log("talk video skipped:", exc)
        return ""


def _ev(**kw) -> bytes:
    return (json.dumps(kw) + "\n").encode()


def turn_stream(canonical_name: str, audio_b64: str | None, text: str | None, say: str | None, token: str | None = None):
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
                text = ears({"raw": x, "sampling_rate": SR_IN},
                            generate_kwargs={"language": "en", "task": "transcribe"})["text"].strip()
            _log(f"heard {text!r} in {time.time() - t0:.2f}s")
            if text.lower().strip(" .!?,") in {p.strip(" .!?,") for p in PHANTOMS} or len(text) < 2:
                yield _ev(type="ignored", reason="noise", text=text)
                yield _ev(type="done")
                return
        if not text:
            yield _ev(type="done")
            return
        yield _ev(type="heard", text=text)
        sess.history.append({"role": "user", "content": text})
        said, t0 = [], time.time()
        for sent in _sentences(_ollama_stream(sess.system, sess.history)):
            sent = _clean(sent)
            if not sent:
                continue
            said.append(sent)
            wav = _speak(sess, sent)
            yield _ev(
                type="say",
                text=sent,
                audio=base64.b64encode(wav).decode(),
                video_url=_talk_video_url(canonical_name, wav),
            )
            _log(f"said {sent!r} (+{time.time() - t0:.1f}s)")
        sess.history.append({"role": "assistant", "content": " ".join(said)})
        yield _ev(type="done")
    except Exception as exc:  # surface to the page instead of a silent hang
        _log("ERROR", repr(exc))
        yield _ev(type="error", message=f"{type(exc).__name__}: {exc}")
