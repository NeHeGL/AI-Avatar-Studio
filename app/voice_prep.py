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
GAP_SEC = 0.25  # pauses shorter than this inside speech are kept as part of the same stretch


def _recordings(vdir: Path) -> list[Path]:
    return sorted(
        p for p in vdir.iterdir() if p.is_file() and not p.name.startswith("_") and p.suffix.lower() in {".wav", ".webm", ".ogg", ".mp3", ".m4a", ".flac"}
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


def build_reference(vdir: Path, load_audio) -> dict | None:
    """Write vdir/_ref.wav from all recordings. `load_audio(path)` returns (mono float32, sample_rate).
    Returns {"path", "seconds", "sources", "speech_seconds"} or None when there is no usable speech."""
    import soundfile as sf

    segs: list[dict] = []
    files = _recordings(vdir)
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
    out = vdir / "_ref.wav"
    sf.write(str(out), y.astype(np.float32), SR)
    return {"path": out, "seconds": len(y) / SR, "speech_seconds": total, "sources": sorted({s["src"] for s in chosen})}
