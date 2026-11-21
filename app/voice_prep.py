"""Build the clip the voice engine listens to.

The voice engine (Chatterbox) copies a voice from one short reference clip: it uses about the first 6 seconds closely and up to 10 in
all. So what matters is not how much a person recorded but how good the best few seconds are. This module reads EVERY recording in a
person's voice folder, cuts out the stretches of real speech, drops the silence, hiss, clipped (too loud) and too-quiet parts, ranks what
is left, and joins the best into one clean ~12 second reference, best parts first. It is written to voice/_ref.wav.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

SR = 24000  # the voice engine's own rate
TARGET_SEC = 12.0  # enough for the engine's 10 s limit, best parts first
MIN_SEGMENT_SEC = 0.8
PINNED = "reference.wav"  # a clip the person put here is used exactly as it is and is never rebuilt or replaced
GAP_SEC = 0.25  # pauses shorter than this inside speech are kept as part of the same stretch


def _recordings(vdir: Path) -> list[Path]:
    return sorted(
        p for p in vdir.iterdir() if p.is_file() and not p.name.startswith("_") and p.name.lower() != PINNED and p.suffix.lower() in {".wav", ".webm", ".ogg", ".mp3", ".m4a", ".flac"}
    )


def _segments(x: np.ndarray, sr: int) -> list[dict]:
    """Stretches of speech in one recording, each with a quality score."""
    import scipy.signal as sps

    frame = int(sr * 0.02)
    n = len(x) // frame
    if n < 10:
        return []
    db = 20 * np.log10(np.sqrt((x[: n * frame].reshape(n, frame) ** 2).mean(1)) + 1e-9)
    floor = float(np.percentile(db, 10))
    top = float(np.percentile(db, 95))
    if top - floor < 12:  # nothing stands out from the background: no usable speech
        return []
    on = db > max(floor + 12.0, top - 28.0)
    # close short gaps, then collect runs
    gap = int(GAP_SEC / 0.02)
    idx = np.flatnonzero(on)
    if not len(idx):
        return []
    runs, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev > gap:
            runs.append((start, prev))
            start = i
        prev = i
    runs.append((start, prev))
    out = []
    pad = int(0.06 * sr)
    for a, b in runs:
        s0, s1 = max(0, a * frame - pad), min(len(x), (b + 1) * frame + pad)
        seg = x[s0:s1]
        dur = len(seg) / sr
        if dur < MIN_SEGMENT_SEC:
            continue
        clip = float((np.abs(seg) > 0.985).mean())
        seg_db = float(np.percentile(db[a : b + 1], 50))
        snr = seg_db - floor
        quiet_penalty = max(0.0, (-38.0 - seg_db) / 10.0)  # very quiet speech is mostly noise once raised
        score = min(dur, 6.0) + snr / 15.0 - 120.0 * clip - quiet_penalty
        out.append({"x": seg, "dur": dur, "score": score, "clip": clip, "snr": snr})
    return out


def _clean(x: np.ndarray, sr: int) -> np.ndarray:
    """Mono, 24 kHz, low rumble removed."""
    import scipy.signal as sps

    if sr != SR:
        g = np.gcd(int(sr), SR)
        x = sps.resample_poly(x, SR // g, int(sr) // g)
    sos = sps.butter(2, 70, "highpass", fs=SR, output="sos")
    return sps.sosfilt(sos, x).astype(np.float32)


def _fingerprint(files: list[Path]) -> list:
    return [[f.name, f.stat().st_size, int(f.stat().st_mtime)] for f in files]


def build_reference(vdir: Path, load_audio) -> dict | None:
    """The clip the voice engine should copy, for the recordings in vdir. `load_audio(path)` returns (mono float32, sample_rate).

    - A file named reference.wav is the person's own choice: used as it is, never rebuilt, never overwritten.
    - Otherwise the best speech from every recording in the folder (any dropped-in wav/mp3/m4a/ogg/webm/flac) is joined into _ref.wav.
      It is rebuilt only when the recordings change, and the previous build is kept as _ref_prev.wav.
    Returns {"path", "seconds", "speech_seconds", "sources", "rebuilt", "pinned"} or None when there is no usable speech."""
    import json
    import shutil

    import soundfile as sf

    pinned = vdir / PINNED
    if pinned.exists():
        return {"path": pinned, "seconds": 0.0, "speech_seconds": 0.0, "sources": [PINNED], "rebuilt": False, "pinned": True}
    files = _recordings(vdir)
    out, meta = vdir / "_ref.wav", vdir / "_ref.json"
    fp = _fingerprint(files)
    try:
        old = json.loads(meta.read_text(encoding="utf-8"))
        if out.exists() and old.get("fingerprint") == fp:
            return {"path": out, "seconds": old.get("seconds", 0.0), "speech_seconds": old.get("speech_seconds", 0.0), "sources": old.get("sources", []), "rebuilt": False, "pinned": False}
    except Exception:
        pass
    segs: list[dict] = []
    for f in files:
        try:
            x, sr = load_audio(f)
        except Exception:
            continue
        x = _clean(np.asarray(x, dtype=np.float32), sr)
        for s in _segments(x, SR):
            s["src"] = f.name
            segs.append(s)
    total = sum(s["dur"] for s in segs)
    if total < 2.0:
        return None
    segs.sort(key=lambda s: -s["score"])
    chosen, have = [], 0.0
    for s in segs:
        if have >= TARGET_SEC:
            break
        chosen.append(s)
        have += s["dur"]
    gap = np.zeros(int(0.12 * SR), dtype=np.float32)
    parts = []
    for s in chosen:
        seg = s["x"].copy()
        fade = int(0.01 * SR)
        seg[:fade] *= np.linspace(0, 1, fade)
        seg[-fade:] *= np.linspace(1, 0, fade)
        parts.append(seg)
        parts.append(gap)
    y = np.concatenate(parts[:-1]) if parts else np.zeros(1, np.float32)
    # one comfortable, steady loudness with a soft ceiling
    rms = float(np.sqrt((y**2).mean()) + 1e-9)
    y = y * min(10 ** (-20 / 20) / rms, 20.0)
    y = np.tanh(y * 1.1) / np.tanh(1.1) * 0.95
    if out.exists():
        shutil.copyfile(out, vdir / "_ref_prev.wav")  # never lose the clip that was in use
    sf.write(str(out), y.astype(np.float32), SR)
    sources = sorted({s["src"] for s in chosen})
    meta.write_text(json.dumps({"fingerprint": fp, "seconds": len(y) / SR, "speech_seconds": total, "sources": sources}), encoding="utf-8")
    return {"path": out, "seconds": len(y) / SR, "speech_seconds": total, "sources": sources, "rebuilt": True, "pinned": False}


def assess(x: np.ndarray, sr: int) -> dict:
    """Is this recording good enough to copy a voice from? Returns {"verdict": good|usable|bad, "issues": [{"level","text"}], "seconds",
    "speech_seconds", "snr_db", "clipped_pct", "sample_rate"} in plain language."""
    x = np.asarray(x, dtype=np.float32)
    seconds = len(x) / sr if sr else 0.0
    issues: list[dict] = []

    def issue(level: str, text: str) -> None:
        issues.append({"level": level, "text": text})

    xc = _clean(x, sr)
    segs = _segments(xc, SR)
    speech = sum(s["dur"] for s in segs)
    frame = int(SR * 0.02)
    n = len(xc) // frame
    snr = 0.0
    if n >= 10:
        db = 20 * np.log10(np.sqrt((xc[: n * frame].reshape(n, frame) ** 2).mean(1)) + 1e-9)
        snr = float(np.percentile(db, 95) - np.percentile(db, 10))
    clip = float((np.abs(x) > 0.985).mean()) * 100 if len(x) else 0.0
    if speech < 3:
        issue("bad", "There is hardly any clear speech in this recording. Use one where you talk for at least 10 seconds, with little background noise.")
    elif speech < 8:
        issue("warn", f"Only about {speech:.0f} seconds of clear speech. 20 seconds or more gives a better copy of your voice.")
    elif speech < 15:
        issue("warn", f"About {speech:.0f} seconds of clear speech. More is better; 20 seconds or more works best.")
    if clip > 2.0:
        issue("bad", "The recording is distorted: it was too loud and the sound is clipped. Record again with the microphone further away or the volume lower.")
    elif clip > 0.3:
        issue("warn", "The loudest parts are a little clipped (too loud).")
    if speech >= 3:
        if snr < 12:
            issue("bad", "There is too much background noise or echo. Use a quieter room.")
        elif snr < 20:
            issue("warn", "There is some background noise. A quieter room gives a cleaner voice.")
    if sr < 16000:
        issue("warn", f"The recording is phone quality ({sr // 1000} kHz). The voice will sound dull; a recording from a microphone is better.")
    verdict = "bad" if any(i["level"] == "bad" for i in issues) else ("usable" if issues else "good")
    return {"verdict": verdict, "issues": issues, "seconds": round(seconds, 1), "speech_seconds": round(speech, 1), "snr_db": round(snr), "clipped_pct": round(clip, 2), "sample_rate": int(sr)}
