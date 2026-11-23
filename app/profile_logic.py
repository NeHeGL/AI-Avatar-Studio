from __future__ import annotations

from pathlib import Path

from app.models import ProfileSection, ScanAsset, UserProfile

MIN_FACE_ANGLES = 1  # one good photo of the face


def pick_avatar_face(assets: list[ScanAsset]) -> ScanAsset | None:
    if not assets:
        return None
    chosen = [a for a in assets if (a.caption or "").startswith("Best photo")]
    if chosen:  # picked by sharpness / neutral face / front-facing, see tools/score_photos
        return chosen[0]
    by_name = [a for a in assets if Path(a.path).name.startswith("angle_100_")]
    if by_name:
        return max(by_name, key=lambda a: a.quality_score)
    exact = [a for a in assets if (a.caption or "").strip().lower() == "center"]
    if exact:
        return max(exact, key=lambda a: a.quality_score)
    centered = [a for a in assets if "center" in (a.caption or "").lower() and "again" not in (a.caption or "").lower()]
    if centered:
        return max(centered, key=lambda a: a.quality_score)
    for a in assets:
        if Path(a.path).name.startswith("angle_8_"):
            return a
    for label in ("Center", "Top", "Right", "Bottom", "Left"):
        hits = [x for x in assets if (x.caption or "").strip().lower() == label.lower()]
        if hits:
            return max(hits, key=lambda a: a.quality_score)
    return max(assets, key=lambda a: a.quality_score)


def face_assets(profile: UserProfile) -> list:
    return [
        a
        for a in profile.scan_assets
        if "body" not in (a.caption or "").lower()
    ]


_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}
_AUDIO_EXT = {".wav", ".webm", ".ogg", ".mp3", ".m4a", ".flac"}


def sync_with_disk(profile: UserProfile) -> UserProfile:
    """Make the profile's lists of photos and recordings match the files that are really in the person's folders, so every screen
    shows the same counts: a photo or clip that was deleted from disk is dropped, and one that was put in the folder by hand (or
    left out of the record by an earlier bug) is added. The files are the truth."""
    from app.models import VoicePhase
    from app.storage import photo_dir, voice_dir

    name = profile.canonical_name
    try:
        sd = photo_dir(name)
        have = {Path(a.path).name: a for a in profile.scan_assets if Path(a.path).exists()}
        if sd.is_dir():
            for f in sorted(sd.iterdir()):
                if f.is_file() and f.suffix.lower() in _IMAGE_EXT and not f.name.startswith(("skin_", "body_")) and f.name not in have:
                    have[f.name] = ScanAsset(id=f"photo-{f.stem[-8:]}", path=str(f), caption="Photo", quality_score=0.7, tracked=True)
        body = [a for a in profile.scan_assets if "body" in (a.caption or "").lower() and Path(a.path).exists()]
        profile.scan_assets = list(have.values()) + [a for a in body if Path(a.path).name not in have]

        vd = voice_dir(name)
        known = {Path(v.path).name: v for v in profile.voice_phases if v.path and Path(v.path).exists()}
        if vd.is_dir():
            for f in sorted(vd.iterdir()):
                if f.is_file() and not f.name.startswith("_") and f.suffix.lower() in _AUDIO_EXT and f.name not in known:
                    known[f.name] = VoicePhase(id=f"voice-{f.stem[:8]}", label="Voice recording", path=str(f), duration_sec=0.0, quality_score=0.85)
        profile.voice_phases = list(known.values()) + [v for v in profile.voice_phases if not v.path]  # placeholders without a file stay as they were
    except Exception:
        pass  # never let a folder problem break loading a profile
    return profile


def reconcile_profile(profile: UserProfile) -> UserProfile:
    sync_with_disk(profile)
    faces = face_assets(profile)
    if len(faces) >= MIN_FACE_ANGLES:
        profile.scan_skipped = False          # a real scan replaces an earlier "no face" choice
    profile.scan_complete = len(faces) >= MIN_FACE_ANGLES or profile.scan_skipped
    profile.voice_complete = any(
        (p.path or "").strip() for p in profile.voice_phases
    )
    profile.facts_complete = bool(
        (profile.facts.notes or "").strip()
        or (
            (profile.facts.display_name or "").strip()
            and profile.facts.display_name.strip() != profile.canonical_name.strip()
        )
    )
    profile.avatar_ready = profile.scan_complete and profile.voice_complete
    return profile


def core_complete(profile: UserProfile) -> bool:
    profile = reconcile_profile(profile)
    return profile.scan_complete and profile.voice_complete


def section_status(profile: UserProfile) -> dict:
    profile = reconcile_profile(profile)
    faces = face_assets(profile)
    return {
        ProfileSection.SCAN.value: {
            "complete": profile.section_complete(ProfileSection.SCAN),
            "count": len(faces),
            "total_assets": len(profile.scan_assets),
        },
        ProfileSection.VOICE.value: {
            "complete": profile.section_complete(ProfileSection.VOICE),
            "count": len(profile.voice_phases),
        },
        ProfileSection.FACTS.value: {
            "complete": profile.section_complete(ProfileSection.FACTS),
            "words": len((profile.facts.notes or "").split()),
        },
        "core_complete": core_complete(profile),
    }