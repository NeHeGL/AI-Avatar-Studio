from __future__ import annotations

import base64
import re
from pathlib import Path
from uuid import uuid4

from app.models import ScanAsset, VoicePhase
from app.profile_logic import reconcile_profile
from app.storage import media_dir, voice_dir, get_working_profile, normalize_name_key, save_draft
from app.walkthrough import profile_summary


def _user_media_dir(canonical_name: str) -> str:
    return normalize_name_key(canonical_name).replace(" ", "_")


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


def _angle_index_from_asset(asset: ScanAsset) -> int | None:
    name = Path(asset.path).name
    if not name.startswith("angle_"):
        return None
    parts = name.split("_")
    if len(parts) < 2:
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


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

    from app.photoreal import cinematic_dir
    from app.profile_logic import face_assets
    from app.walkthrough import media_url

    profile = get_working_profile(canonical_name)
    pinned = None
    try:
        pinned = json.loads((cinematic_dir(canonical_name) / "options.json").read_text(encoding="utf-8")).get("source_photo")
    except Exception:
        pass
    rows = [{"id": a.id, "name": Path(a.path).name, "url": media_url(a.path), "score": round(a.quality_score * 100), "active": Path(a.path).name == pinned}
            for a in face_assets(profile) if Path(a.path).exists()]
    if rows and not any(r["active"] for r in rows):
        rows[-1]["active"] = True  # nothing pinned: the newest one is used
    return {"status": "ok", "photos": rows, "profile": profile_summary(profile)}


def _pin(canonical_name: str, filename: str) -> None:
    import json

    from app.photoreal import cinematic_dir, invalidate

    invalidate(canonical_name)  # everything made from the old picture goes (and its pin)
    o = cinematic_dir(canonical_name) / "options.json"
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
        d = media_dir(canonical_name) / "scan"
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
    d = media_dir(canonical_name) / "scan"
    if not (d / Path(name).name).exists():
        return {"status": "error", "message": "That photo is gone."}
    _pin(canonical_name, Path(name).name)
    return photos(canonical_name)


def delete_photo(canonical_name: str, name: str) -> dict:
    profile = get_working_profile(canonical_name)
    keep = [a for a in profile.scan_assets if Path(a.path).name != Path(name).name]
    if len(keep) == len(profile.scan_assets):
        return {"status": "error", "message": "That photo is not in this profile."}
    (media_dir(canonical_name) / "scan" / Path(name).name).unlink(missing_ok=True)
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


def ingest_voice(canonical_name: str, audio_base64: str, label: str) -> dict:
    profile = get_working_profile(canonical_name)
    vdir = voice_dir(canonical_name)
    vdir.mkdir(parents=True, exist_ok=True)

    raw, mime = decode_data_url(audio_base64)
    ext = sniff_audio_ext(raw, extension_for_mime(mime))
    fname = f"{uuid4().hex[:8]}.{ext}"
    path = vdir / fname
    path.write_bytes(raw)

    profile.voice_phases.append(
        VoicePhase(
            id=f"voice-{uuid4().hex[:8]}",
            label=label or "Voice sample",
            path=str(path),
            waveform_peaks=[0.2, 0.5, 0.7, 0.9, 0.6, 0.4, 0.8, 0.5, 0.3, 0.6],
            duration_sec=3.0,
            quality_score=0.85,
        )
    )
    profile = reconcile_profile(profile)
    save_draft(profile)
    return {"status": "ok", "profile": profile_summary(profile)}


def ingest_other(canonical_name: str, display_name: str, notes: str) -> dict:
    profile = get_working_profile(canonical_name)
    profile.other.display_name = display_name.strip() or profile.display_name
    profile.other.notes = notes.strip()
    profile = reconcile_profile(profile)
    profile.touch()
    save_draft(profile)
    return {"status": "ok", "profile": profile_summary(profile)}
