from __future__ import annotations

import base64
import re
from pathlib import Path
from uuid import uuid4

from app.models import ScanAsset, VoicePhase
from app.profile_logic import reconcile_profile
from app.storage import photo_dir, voice_dir, get_working_profile, save_draft
from app.walkthrough import profile_summary


def decode_data_url(data_url: str) -> tuple[bytes, str]:
    match = re.match(r"^data:([^;]+);base64,(.+)$", data_url.strip(), re.DOTALL)
    if match:
        mime = match.group(1)
        raw = base64.b64decode(match.group(2))
        return raw, mime
    raw = base64.b64decode(data_url)
    return raw, "image/jpeg"


def extension_for_mime(mime: str) -> str:
    if "png" in mime:
        return "png"
    if "webp" in mime:
        return "webp"
    if "webm" in mime:
        return "webm"
    if "ogg" in mime:
        return "ogg"
    return "jpg"


def sniff_audio_ext(raw: bytes, fallback: str) -> str:
    """Pick the extension from the file's real bytes (browsers mislabel Ogg as WebM)."""
    if raw[:4] == b"OggS":
        return "ogg"
    if raw[:4] == bytes([0x1A, 0x45, 0xDF, 0xA3]):
        return "webm"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WAVE":
        return "wav"
    if raw[:3] == b"ID3" or raw[:2] in (bytes([0xFF, 0xFB]), bytes([0xFF, 0xF3])):
        return "mp3"
    if raw[4:8] == b"ftyp":
        return "m4a"
    return fallback


MAX_CLIPS = 5  # recordings per profile, shown as tiles like the photos
MAX_PHOTOS = 5  # kept in step with the Photo page: they all fit on one row


def _decode_image(b64: str):
    import cv2
    import numpy as np

    raw, _mime = decode_data_url(b64)
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("That file is not a picture the app can read.")
    return img


def _data_url(img, quality: int = 92) -> str:
    import cv2

    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def check_photos(images: list[str]) -> dict:
    """Analyze one uploaded photo, or several webcam snaps (the best one wins). Nothing is saved. Returns the verdict, plain-language
    reasons, and the prepared (levelled, centred, zoomed, cropped) picture as a data URL."""
    from app import photo_prep

    try:
        imgs = [_decode_image(b) for b in images if b]
        if not imgs:
            return {"status": "error", "message": "No picture was received."}
        i, r = photo_prep.best_of(imgs)
        return {
            "status": "ok", "verdict": r["verdict"], "ok": r["ok"], "score": r["score"], "issues": r["issues"], "face": r["face"],
            "picked": i, "count": len(imgs), "image": _data_url(r["crop"]) if r["crop"] is not None else None,
        }
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


def photos(canonical_name: str) -> dict:
    """The person's saved face photos, and which one the avatar is made from."""
    import json

    from app.photoreal import face_dir
    from app.profile_logic import face_assets
    from app.walkthrough import media_url

    profile = reconcile_profile(get_working_profile(canonical_name))  # the files on disk are the truth: the tiles and the counts must agree
    pinned = None
    try:
        pinned = json.loads((face_dir(canonical_name) / "options.json").read_text(encoding="utf-8")).get("source_photo")
    except Exception:
        pass
    rows = [{"id": a.id, "name": Path(a.path).name, "url": media_url(a.path), "score": round(a.quality_score * 100), "active": Path(a.path).name == pinned}
            for a in face_assets(profile) if Path(a.path).exists()]
    if rows and not any(r["active"] for r in rows):
        rows[-1]["active"] = True  # nothing pinned: the newest one is used
    return {"status": "ok", "photos": rows, "profile": profile_summary(profile)}


def _pin(canonical_name: str, filename: str) -> None:
    import json

    from app.photoreal import face_dir, invalidate

    invalidate(canonical_name)  # everything made from the old picture goes (and its pin)
    o = face_dir(canonical_name) / "options.json"
    try:
        opts = json.loads(o.read_text(encoding="utf-8"))
    except Exception:
        opts = {}
    opts["source_photo"] = filename
    o.parent.mkdir(parents=True, exist_ok=True)
    o.write_text(json.dumps(opts, indent=1), encoding="utf-8")


def save_photo(canonical_name: str, image: str) -> dict:
    """Keep a prepared photo (as returned by check_photos) as one of the person's faces, and make it the one the avatar uses."""
    from app import photo_prep

    try:
        img = _decode_image(image)
        r = photo_prep.analyze(img)
        if not r["ok"]:
            return {"status": "error", "message": r["issues"][0]["text"] if r["issues"] else "That photo cannot be used."}
        profile = get_working_profile(canonical_name)
        from app.profile_logic import face_assets

        if len(face_assets(profile)) >= MAX_PHOTOS:
            return {"status": "error", "message": f"{MAX_PHOTOS} photos is the most that can be kept. Delete one to add another."}
        d = photo_dir(canonical_name)
        d.mkdir(parents=True, exist_ok=True)
        import cv2

        name = f"portrait_{uuid4().hex[:8]}.jpg"  # the prefix tells the video engine this picture is already framed
        cv2.imwrite(str(d / name), r["crop"], [cv2.IMWRITE_JPEG_QUALITY, 95])
        profile.scan_assets.append(ScanAsset(id=f"photo-{uuid4().hex[:8]}", path=str(d / name), caption="Photo", quality_score=float(r["score"]) / 100.0, tracked=True))
        profile.scan_skipped = False
        profile = reconcile_profile(profile)
        profile.touch()
        save_draft(profile)
        _pin(canonical_name, name)
        return {"status": "ok", "profile": profile_summary(profile), "photo": name}
    except Exception as exc:
        return {"status": "error", "message": f"Save failed: {exc}"}


def select_photo(canonical_name: str, name: str) -> dict:
    d = photo_dir(canonical_name)
    if not (d / Path(name).name).exists():
        return {"status": "error", "message": "That photo is gone."}
    _pin(canonical_name, Path(name).name)
    return photos(canonical_name)


def delete_photo(canonical_name: str, name: str) -> dict:
    profile = get_working_profile(canonical_name)
    keep = [a for a in profile.scan_assets if Path(a.path).name != Path(name).name]
    if len(keep) == len(profile.scan_assets):
        return {"status": "error", "message": "That photo is not in this profile."}
    (photo_dir(canonical_name) / Path(name).name).unlink(missing_ok=True)
    profile.scan_assets = keep
    profile = reconcile_profile(profile)
    profile.touch()
    save_draft(profile)
    cur = photos(canonical_name)
    if cur["photos"]:
        act = next((r for r in cur["photos"] if r["active"]), cur["photos"][-1])
        _pin(canonical_name, act["name"])  # the avatar might have been made from the deleted one
    else:
        from app.photoreal import invalidate

        invalidate(canonical_name)  # no photo left: everything made from it goes too
    return {"status": "ok", "profile": profile_summary(profile), **photos(canonical_name)}


def ingest_other(canonical_name: str, display_name: str, notes: str, force_clear: bool = False) -> dict:
    profile = get_working_profile(canonical_name)
    profile.other.display_name = display_name.strip() or profile.display_name
    if notes.strip() or not (profile.other.notes or "").strip() or force_clear:
        profile.other.notes = notes.strip()  # an empty note never silently replaces existing Knowledge
    profile = reconcile_profile(profile)
    profile.touch()
    save_draft(profile)
    return {"status": "ok", "profile": profile_summary(profile)}


_AUDIO_EXT = {".wav", ".webm", ".ogg", ".mp3", ".m4a", ".flac"}


def _seconds(path: Path) -> float | None:
    try:
        import soundfile as sf

        i = sf.info(str(path))
        return round(i.duration, 1)
    except Exception:
        pass
    try:
        import av

        with av.open(str(path)) as c:
            if c.duration:
                return round(c.duration / 1_000_000, 1)
    except Exception:
        pass
    return None


def voice_list(canonical_name: str) -> dict:
    """Every audio clip in the person's voice folder (recorded in the app or dropped in), newest first."""
    from app.walkthrough import media_url

    d = voice_dir(canonical_name)
    rows = []
    if d.is_dir():
        for f in sorted(d.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if f.is_file() and not f.name.startswith("_") and f.suffix.lower() in _AUDIO_EXT:
                rows.append({"name": f.name, "url": media_url(str(f)), "seconds": _seconds(f), "locked": f.name.lower() == "reference.wav", "recorded": int(f.stat().st_mtime)})
    return {"status": "ok", "clips": rows, "profile": profile_summary(get_working_profile(canonical_name))}


def voice_delete(canonical_name: str, name: str) -> dict:
    d = voice_dir(canonical_name)
    f = d / Path(name).name
    if not f.is_file() or f.name.startswith("_"):
        return {"status": "error", "message": "That clip is not in this profile."}
    f.unlink(missing_ok=True)
    profile = get_working_profile(canonical_name)
    profile.voice_phases = [v for v in profile.voice_phases if Path(v.path).name != f.name]
    profile = reconcile_profile(profile)
    profile.touch()
    save_draft(profile)
    (d / "_ref.json").unlink(missing_ok=True)  # the clean reference is rebuilt from what is left
    return voice_list(canonical_name)


_VOICE_TMP = Path(__file__).resolve().parent.parent / "data" / "tmp_render" / "voice_candidates"


def voice_check(audio_base64: str) -> dict:
    """Analyze one recording (made in the app or uploaded). Nothing is added to the profile yet: the file is held under a token until Save."""
    from app import voice_prep
    from app.chat import _load_any_audio

    try:
        raw, mime = decode_data_url(audio_base64)
        if not raw:
            return {"status": "error", "message": "That file is empty."}
        ext = sniff_audio_ext(raw, extension_for_mime(mime))
        _VOICE_TMP.mkdir(parents=True, exist_ok=True)
        for old in _VOICE_TMP.glob("*"):  # candidates older than a day are cleared
            try:
                if old.stat().st_mtime < __import__("time").time() - 86400:
                    old.unlink()
            except OSError:
                pass
        token = uuid4().hex[:12]
        path = _VOICE_TMP / f"{token}.{ext}"
        path.write_bytes(raw)
        try:
            x, sr = _load_any_audio(path)
        except Exception:
            path.unlink(missing_ok=True)
            return {"status": "error", "message": "That file could not be read as audio. Try a wav, mp3, m4a, ogg or webm file."}
        r = voice_prep.assess(x, sr)
        return {"status": "ok", "token": token, "ext": ext, **r}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


def voice_save(canonical_name: str, token: str) -> dict:
    """Add a checked recording to the person's voice folder and profile."""
    from app import voice_prep
    from app.chat import _load_any_audio

    src = next(iter(_VOICE_TMP.glob(f"{Path(token).name}.*")), None) if token else None
    if src is None or not src.exists():
        return {"status": "error", "message": "That recording is no longer available. Please add it again."}
    x, sr = _load_any_audio(src)
    if voice_prep.assess(x, sr)["verdict"] == "bad":
        return {"status": "error", "message": "That recording cannot be used."}
    profile = get_working_profile(canonical_name)
    d = voice_dir(canonical_name)
    d.mkdir(parents=True, exist_ok=True)
    if len(voice_list(canonical_name)["clips"]) >= MAX_CLIPS:
        return {"status": "error", "message": f"{MAX_CLIPS} recordings is the most that can be kept. Delete one to add another."}
    dest = d / f"{uuid4().hex[:8]}{src.suffix}"
    src.replace(dest)
    profile.voice_phases.append(
        VoicePhase(id=f"voice-{uuid4().hex[:8]}", label="Voice recording", path=str(dest), duration_sec=round(len(x) / sr, 1), quality_score=0.85)
    )
    profile = reconcile_profile(profile)
    profile.touch()
    save_draft(profile)
    (d / "_ref.json").unlink(missing_ok=True)  # the clean reference is rebuilt with the new clip included
    return {"status": "ok", "profile": profile_summary(profile)}
