from __future__ import annotations

from pathlib import Path

from app.models import ProfileSection, UserProfile
from app.profile_logic import (
    face_assets,
    pick_avatar_face,
    reconcile_profile,
    section_status,
)
from app.storage import (
    commit_draft,
    find_by_display_or_canonical,
    get_or_create_draft,
    load_draft,
)


def check_name(name: str) -> dict:
    name = name.strip()
    if not name:
        return {"status": "invalid", "message": "Please enter a name."}
    collision = find_by_display_or_canonical(name)
    if collision:
        collision = reconcile_profile(collision)
        return {
            "status": "exists",
            "message": "I see a user with this name already exists, is this you?",
            "profile_summary": profile_summary(collision),
        }
    return {"status": "new", "message": "Nice to meet you. Let's build your profile."}


def profile_summary(profile: UserProfile) -> dict:
    profile = reconcile_profile(profile)
    sections = section_status(profile)
    faces = face_assets(profile)
    face = pick_avatar_face(faces)
    # Newest recording that actually has a file (the first entry can be a placeholder with no audio).
    voice = next((v for v in reversed(profile.voice_phases) if v.path), None)
    return {
        "canonical_name": profile.canonical_name,
        "display_name": profile.display_name,
        "sections": {
            ProfileSection.SCAN.value: sections[ProfileSection.SCAN.value],
            ProfileSection.VOICE.value: sections[ProfileSection.VOICE.value],
            ProfileSection.OTHER.value: sections[ProfileSection.OTHER.value],
        },
        "core_complete": sections["core_complete"],
        "scan_previews": [
            {"id": a.id, "url": media_url(a.path), "caption": a.caption}
            for a in profile.scan_assets[:6]
        ],
        # Every non-body scan frame (the mesh avatar picks the most frontal one itself).
        "scan_frames": [
            {"id": a.id, "url": media_url(a.path), "caption": a.caption}
            for a in profile.scan_assets
            if "body" not in (a.caption or "").lower() and not Path(a.path).name.startswith("body_")
        ],
        "voice_phases": [
            {
                "id": p.id,
                "label": p.label,
                "url": media_url(p.path) if p.path else "",
                "peaks": p.waveform_peaks,
                "duration_sec": p.duration_sec,
            }
            for p in profile.voice_phases
        ],
        "other_preview": {
            "display_name": profile.other.display_name,
            "notes": profile.other.notes,
            "preferences": profile.other.preferences,
        },
        "avatar_ready": profile.avatar_ready,
        "avatar_face_url": media_url(face.path) if face else "",
        "avatar_voice_url": media_url(voice.path) if voice and voice.path else "",
        "avatar_voice_label": voice.label if voice else "",
    }


def media_url(path: str) -> str:
    if not path:
        return ""
    from pathlib import Path

    data_root = Path(__file__).resolve().parent.parent / "data"
    try:
        rel = Path(path).resolve().relative_to(data_root)
        return f"/data/{rel.as_posix()}"
    except ValueError:
        pass
    # Project folder was renamed/moved: keep what follows the last "data" segment.
    parts = Path(path.replace("\\", "/")).parts
    if "data" in parts:
        i = len(parts) - 1 - parts[::-1].index("data")
        return "/data/" + "/".join(parts[i + 1:])
    return path


def confirm_existing(name: str, is_you: bool, unique_full_name: str | None) -> dict:
    if is_you:
        profile = find_by_display_or_canonical(name)
        if not profile:
            return {"status": "error", "message": "Profile not found."}
        profile = reconcile_profile(profile)
        return {
            "status": "returning",
            "profile": profile_summary(profile),
        }
    if not unique_full_name or len(unique_full_name.strip()) < 3:
        return {
            "status": "need_unique",
            "message": "Please enter a unique or full name (firstname lastname).",
        }
    if find_by_display_or_canonical(unique_full_name):
        return {
            "status": "collision",
            "message": "That name is already saved. Try another full name.",
        }
    return {
        "status": "new",
        "canonical_name": unique_full_name.strip(),
    }


def delete_profile(canonical_name: str) -> dict:
    from app.storage import delete_saved_profile

    if not delete_saved_profile(canonical_name):
        return {"status": "error", "message": "No saved profile to delete."}
    return {"status": "ok", "message": "Profile deleted."}


def save_onboarding(canonical_name: str) -> dict:
    if not load_draft(canonical_name):
        # Nothing was changed since the last save. NEVER make a blank draft and commit it: that replaced a person's whole saved profile
        # (photos, voice and Knowledge) with an empty one every time this ran for someone who already had a saved profile.
        existing = find_by_display_or_canonical(canonical_name)
        if existing:
            existing = reconcile_profile(existing)
            return {
                "status": "avatar",
                "message": "Profile saved.",
                "profile": profile_summary(existing),
                "avatar_mode": "live" if existing.avatar_ready else "placeholder",
            }
        get_or_create_draft(canonical_name)
    profile = commit_draft(canonical_name)
    if not profile:
        return {"status": "error", "message": "Nothing to save."}
    return {
        "status": "avatar",
        "message": "Profile saved.",
        "profile": profile_summary(profile),
        "avatar_mode": "live" if profile.avatar_ready else "placeholder",
    }


def skip_updates(canonical_name: str) -> dict:
    profile = find_by_display_or_canonical(canonical_name)
    if not profile:
        profile = load_draft(canonical_name)
    if not profile:
        return {
            "status": "error",
            "message": "No profile data found yet.",
        }
    profile = reconcile_profile(profile)
    return {
        "status": "avatar",
        "message": "Showing your avatar.",
        "profile": profile_summary(profile),
        "avatar_mode": "live" if profile.avatar_ready else "placeholder",
    }

