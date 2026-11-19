from __future__ import annotations

import os
from pathlib import Path

from app.models import ProfileSection, ScanAsset, UserProfile, VoicePhase
from app.profile_logic import (
    core_complete,
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
    normalize_name_key,
    save_draft,
    save_profile,
)

MIN_SCAN_QUALITY = 0.5
MIN_VOICE_QUALITY = 0.6


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
        # live (animated lip-sync) | photoreal (still) | interactive | puppet — static/avatar-player.js
        "avatar_render": os.environ.get("AVATAR_RENDER", "live"),
        "avatar_video_enabled": os.environ.get("AVATAR_PHOTOREAL_VIDEO", "").strip() == "1",
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


def apply_updates(
    canonical_name: str,
    sections: list[str],
    merge_existing: bool = True,
) -> dict:
    profile = find_by_display_or_canonical(canonical_name)
    if not profile:
        return {"status": "error", "message": "Profile not found."}

    wants = {ProfileSection(s) for s in sections if s in {e.value for e in ProfileSection}}

    dropped: dict[str, int] = {"scan": 0, "voice": 0}

    if ProfileSection.SCAN in wants:
        before = len(profile.scan_assets)
        profile = refine_scan(profile, merge_existing)
        dropped["scan"] = before - len(profile.scan_assets)
        run_capture_stub(
            profile.canonical_name, ProfileSection.SCAN.value, refresh=True
        )
        profile = find_by_display_or_canonical(profile.canonical_name) or profile
    if ProfileSection.VOICE in wants:
        before = len(profile.voice_phases)
        profile = refine_voice(profile, merge_existing)
        dropped["voice"] = before - len(profile.voice_phases)
        run_capture_stub(
            profile.canonical_name, ProfileSection.VOICE.value, refresh=True
        )
        profile = find_by_display_or_canonical(profile.canonical_name) or profile
    if ProfileSection.OTHER in wants:
        profile.other_complete = True
        profile.touch()

    profile.avatar_ready = (
        profile.section_complete(ProfileSection.SCAN)
        and profile.section_complete(ProfileSection.VOICE)
    )
    save_profile(profile)
    return {
        "status": "updated",
        "profile": profile_summary(profile),
        "avatar_mode": "preview" if profile.avatar_ready else "incomplete",
        "dropped": dropped,
    }


def delete_profile(canonical_name: str) -> dict:
    from app.storage import delete_saved_profile

    if not delete_saved_profile(canonical_name):
        return {"status": "error", "message": "No saved profile to delete."}
    return {"status": "ok", "message": "Profile deleted."}


def save_onboarding(canonical_name: str) -> dict:
    if not load_draft(canonical_name):
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


def preview_draft(canonical_name: str) -> dict:
    draft = load_draft(canonical_name)
    if not draft:
        return {"status": "error", "message": "No in-progress data for this name."}
    return {
        "status": "avatar",
        "message": "Preview (not saved yet).",
        "profile": profile_summary(draft),
        "avatar_mode": "live" if draft.avatar_ready else "placeholder",
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


def refine_scan(profile: UserProfile, merge_existing: bool) -> UserProfile:
    kept = [a for a in profile.scan_assets if a.quality_score >= MIN_SCAN_QUALITY]
    dropped = len(profile.scan_assets) - len(kept)
    profile.scan_assets = kept
    profile.scan_complete = len(kept) > 0
    if not merge_existing and dropped:
        pass
    profile.touch()
    return profile


def refine_voice(profile: UserProfile, merge_existing: bool) -> UserProfile:
    kept = [p for p in profile.voice_phases if p.quality_score >= MIN_VOICE_QUALITY]
    profile.voice_phases = kept
    profile.voice_complete = len(kept) > 0
    profile.touch()
    return profile


def run_capture_stub(canonical_name: str, section: str, *, refresh: bool = False) -> dict:
    profile = load_draft(canonical_name) or get_or_create_draft(canonical_name)
    section_enum = ProfileSection(section)
    if section_enum == ProfileSection.SCAN:
        profile.scan_complete = True
        if not profile.scan_assets or refresh:
            from app.storage import MEDIA_DIR
            from uuid import uuid4

            scan_dir = MEDIA_DIR / normalize_name_key(canonical_name).replace(" ", "_") / "scan"
            scan_dir.mkdir(parents=True, exist_ok=True)
            fname = f"capture_{uuid4().hex[:8]}.svg"
            fpath = scan_dir / fname
            fpath.write_text(
                '<svg xmlns="http://www.w3.org/2000/svg" width="320" height="400">'
                '<rect width="100%" height="100%" fill="#6eb5ff"/>'
                '<text x="50%" y="50%" text-anchor="middle" fill="white" '
                'font-family="sans-serif" font-size="18">Updated capture</text></svg>',
                encoding="utf-8",
            )
            profile.scan_assets.append(
                ScanAsset(
                    id=f"scan-{uuid4().hex[:8]}",
                    path=str(fpath),
                    caption="Latest capture",
                    quality_score=0.95,
                )
            )
    elif section_enum == ProfileSection.VOICE:
        profile.voice_complete = True
        if not profile.voice_phases or refresh:
            from uuid import uuid4

            profile.voice_phases.append(
                VoicePhase(
                    id=f"voice-{uuid4().hex[:8]}",
                    label="New capture" if refresh else "Initial capture",
                    path="",
                    waveform_peaks=[0.3, 0.7, 0.9, 0.5, 0.4, 0.6, 0.8],
                    duration_sec=1.8,
                    quality_score=0.9,
                )
            )
    else:
        profile.other_complete = True
    profile.avatar_ready = profile.scan_complete and profile.voice_complete
    save_draft(profile)
    return {"status": "ok", "profile": profile_summary(profile)}