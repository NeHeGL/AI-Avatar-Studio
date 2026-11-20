from __future__ import annotations

from pathlib import Path

from app.models import ProfileSection, ScanAsset, UserProfile

MIN_FACE_ANGLES = 4


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


def reconcile_profile(profile: UserProfile) -> UserProfile:
    faces = face_assets(profile)
    if len(faces) >= MIN_FACE_ANGLES:
        profile.scan_skipped = False          # a real scan replaces an earlier "no face" choice
    profile.scan_complete = len(faces) >= MIN_FACE_ANGLES or profile.scan_skipped
    profile.voice_complete = any(
        (p.path or "").strip() for p in profile.voice_phases
    )
    profile.other_complete = bool(
        (profile.other.notes or "").strip()
        or (
            (profile.other.display_name or "").strip()
            and profile.other.display_name.strip() != profile.canonical_name.strip()
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
        ProfileSection.OTHER.value: {
            "complete": profile.section_complete(ProfileSection.OTHER),
        },
        "core_complete": core_complete(profile),
    }