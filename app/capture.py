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


def ingest_scan_frames(canonical_name: str, frames: list[dict]) -> dict:
    try:
        profile = get_working_profile(canonical_name)
        scan_dir = media_dir(canonical_name) / "scan"
        scan_dir.mkdir(parents=True, exist_ok=True)

        if not frames:
            return {"status": "error", "message": "No frames to save."}

        def is_body(a: ScanAsset) -> bool:
            return "body" in (a.caption or "").lower() or Path(a.path).name.startswith("body_")

        body_keep = [a for a in profile.scan_assets if is_body(a)]
        merged: dict[int, ScanAsset] = {}
        for a in profile.scan_assets:
            if is_body(a):
                continue
            idx = _angle_index_from_asset(a)
            if idx is not None:
                merged[idx] = a

        # A full scan (many wedges) REPLACES the old face scan — merging by quality kept stale
        # photos from earlier, lower-resolution scans alive (their scores were higher).
        if len([f for f in frames if (f.get("image_base64") or "")]) >= 16:
            merged = {}

        new_count = 0
        for item in frames:
            b64 = item.get("image_base64") or ""
            if not b64 or len(b64) < 50:
                continue
            try:
                raw, mime = decode_data_url(b64)
            except (ValueError, TypeError):
                continue
            if not raw:
                continue
            angle_idx = int(item.get("angle_index", new_count))
            ext = extension_for_mime(mime)
            fname = f"angle_{angle_idx}_{uuid4().hex[:8]}.{ext}"
            path = scan_dir / fname
            path.write_bytes(raw)
            quality = float(item.get("quality_score", 0.9))
            new_asset = ScanAsset(
                id=f"scan-{uuid4().hex[:8]}",
                path=str(path),
                caption=item.get("caption", f"Angle {angle_idx}"),
                quality_score=quality,
                yaw_deg=item.get("yaw_deg"),
                pitch_deg=item.get("pitch_deg"),
                tracked=bool(item.get("tracked", True)),
            )
            new_count += 1
            prev = merged.get(angle_idx)
            if prev is None or new_asset.quality_score > prev.quality_score:
                merged[angle_idx] = new_asset

        if not merged:
            return {"status": "error", "message": "Could not decode any scan images."}

        ordered = [merged[k] for k in sorted(merged.keys())]
        profile.scan_assets = body_keep + ordered

        profile = reconcile_profile(profile)
        save_draft(profile)
        try:  # a rescan means a new photo: everything made from the old one is purged
            from app.photoreal import invalidate

            invalidate(canonical_name)
        except Exception:
            pass
        return {"status": "ok", "profile": profile_summary(profile)}
    except Exception as exc:
        return {"status": "error", "message": f"Save failed: {exc}"}


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


def ingest_body_frames(canonical_name: str, frames: list[dict]) -> dict:
    profile = get_working_profile(canonical_name)
    scan_dir = media_dir(canonical_name) / "scan"
    scan_dir.mkdir(parents=True, exist_ok=True)

    for item in frames:
        raw, mime = decode_data_url(item["image_base64"])
        ext = extension_for_mime(mime)
        fname = f"body_{item.get('angle_index', 0)}_{uuid4().hex[:8]}.{ext}"
        path = scan_dir / fname
        path.write_bytes(raw)
        profile.scan_assets.append(
            ScanAsset(
                id=f"scan-{uuid4().hex[:8]}",
                path=str(path),
                caption=item.get("caption", "Body"),
                quality_score=float(item.get("quality_score", 0.85)),
            )
        )

    profile.touch()
    save_draft(profile)
    return {"status": "ok", "profile": profile_summary(profile)}


def skip_scan(canonical_name: str) -> dict:
    """The person has no face to scan (a cartoon character such as Astro Boy, a pet, a voice-only avatar)."""
    profile = get_working_profile(canonical_name)
    profile.scan_skipped = True
    profile = reconcile_profile(profile)
    profile.touch()
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

def save_skin_atlas(canonical_name: str, atlas_base64: str, layout: dict) -> dict:
    """Store the stitched skin map (PNG) and its mesh layout so the 3D head builder can use them."""
    try:
        import json as _json

        raw, _mime = decode_data_url(atlas_base64)
        if not raw:
            return {"status": "error", "message": "Empty atlas."}
        scan_dir = media_dir(canonical_name) / "scan"
        scan_dir.mkdir(parents=True, exist_ok=True)
        (scan_dir / "skin_atlas.png").write_bytes(raw)
        (scan_dir / "skin_atlas.json").write_text(_json.dumps(layout), encoding="utf-8")
        return {"status": "ok", "bytes": len(raw)}
    except Exception as exc:
        return {"status": "error", "message": f"Atlas save failed: {exc}"}
