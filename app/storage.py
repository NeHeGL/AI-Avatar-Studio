from __future__ import annotations

import json
import re
from pathlib import Path

from app.models import UserProfile

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
USERS_DIR = DATA_ROOT / "users"
DRAFTS_DIR = DATA_ROOT / "drafts"
MEDIA_DIR = DATA_ROOT / "media"


def normalize_name_key(name: str) -> str:
    cleaned = re.sub(r"\s+", " ", name.strip().lower())
    return cleaned


def profile_path(key: str) -> Path:
    return USERS_DIR / f"{key}.json"


def draft_path(key: str) -> Path:
    return DRAFTS_DIR / f"{key}.json"


def find_by_display_or_canonical(name: str) -> UserProfile | None:
    """Saved profiles only (data/users). Drafts do not count as existing users."""
    key = normalize_name_key(name)
    direct = profile_path(key)
    if direct.exists():
        profile = UserProfile.model_validate_json(direct.read_text(encoding="utf-8"))
        if profile.saved:
            return profile
        return None
    for path in USERS_DIR.glob("*.json"):
        profile = UserProfile.model_validate_json(path.read_text(encoding="utf-8"))
        if not profile.saved:
            continue
        if normalize_name_key(profile.display_name or profile.canonical_name) == key:
            return profile
        if normalize_name_key(profile.canonical_name) == key:
            return profile
    return None


def list_saved_names() -> list[str]:
    """Display names of saved profiles (not drafts), sorted for the name dropdown."""
    names: list[str] = []
    if USERS_DIR.exists():
        for path in USERS_DIR.glob("*.json"):
            try:
                profile = UserProfile.model_validate_json(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if profile.saved:
                names.append(profile.display_name or profile.canonical_name)
    return sorted(set(names), key=str.lower)


def load_draft(name: str) -> UserProfile | None:
    path = draft_path(normalize_name_key(name))
    if not path.exists():
        return None
    return UserProfile.model_validate_json(path.read_text(encoding="utf-8"))


def save_draft(profile: UserProfile) -> UserProfile:
    DRAFTS_DIR.mkdir(parents=True, exist_ok=True)
    key = normalize_name_key(profile.canonical_name)
    profile.saved = False
    profile.touch()
    draft_path(key).write_text(
        profile.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return profile


def delete_draft(name: str) -> None:
    path = draft_path(normalize_name_key(name))
    if path.exists():
        path.unlink()


def get_working_profile(full_name: str) -> UserProfile:
    """Draft if present; otherwise clone saved profile into draft for editing."""
    draft = load_draft(full_name)
    if draft:
        return draft
    saved = find_by_display_or_canonical(full_name)
    if saved:
        working = saved.model_copy(deep=True)
        working.saved = False
        return save_draft(working)
    return get_or_create_draft(full_name)


def delete_saved_profile(full_name: str) -> bool:
    key = normalize_name_key(full_name)
    deleted = False
    path = profile_path(key)
    if path.exists():
        path.unlink()
        deleted = True
    delete_draft(full_name)
    return deleted


def get_or_create_draft(full_name: str) -> UserProfile:
    existing = load_draft(full_name)
    if existing:
        return existing
    profile = UserProfile(
        canonical_name=full_name.strip(),
        display_name=full_name.strip(),
        saved=False,
    )
    return save_draft(profile)


def save_profile(profile: UserProfile) -> UserProfile:
    from app.profile_logic import reconcile_profile

    USERS_DIR.mkdir(parents=True, exist_ok=True)
    key = normalize_name_key(profile.canonical_name)
    profile = reconcile_profile(profile)
    profile.saved = True
    profile.touch()
    profile_path(key).write_text(
        profile.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return profile


def purge_unsaved_user_files() -> None:
    """Move mistaken early saves out of users/ into drafts/."""
    if not USERS_DIR.exists():
        return
    for path in USERS_DIR.glob("*.json"):
        profile = UserProfile.model_validate_json(path.read_text(encoding="utf-8"))
        if profile.saved:
            continue
        save_draft(profile)
        path.unlink()


def commit_draft(full_name: str) -> UserProfile | None:
    draft = load_draft(full_name)
    if not draft:
        return None
    saved = save_profile(draft)
    delete_draft(full_name)
    return saved


def seed_demo_user() -> None:
    """Optional demo data so the returning-user path is testable."""
    key = normalize_name_key("Jeff Molofee")
    if profile_path(key).exists():
        return
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    scan_dir = MEDIA_DIR / "jeff_molofee" / "scan"
    voice_dir = MEDIA_DIR / "jeff_molofee" / "voice"
    scan_dir.mkdir(parents=True, exist_ok=True)
    voice_dir.mkdir(parents=True, exist_ok=True)

    from app.models import OtherData, ScanAsset, VoicePhase

    profile = UserProfile(
        canonical_name="Jeff Molofee",
        display_name="Jeff",
        saved=True,
        scan_complete=True,
        voice_complete=True,
        other_complete=True,
        avatar_ready=True,
        scan_assets=[
            ScanAsset(
                id="s1",
                path=str(scan_dir / "front.svg"),
                caption="Front neutral",
                quality_score=0.92,
            ),
            ScanAsset(
                id="s2",
                path=str(scan_dir / "left.svg"),
                caption="Left 3/4",
                quality_score=0.88,
            ),
            ScanAsset(
                id="s3",
                path=str(scan_dir / "right.svg"),
                caption="Right 3/4",
                quality_score=0.45,
            ),
        ],
        voice_phases=[
            VoicePhase(
                id="v1",
                label="Greeting",
                path="",
                waveform_peaks=[
                    0.2, 0.5, 0.8, 0.6, 0.9, 0.4, 0.3, 0.7, 0.5, 0.2,
                ],
                duration_sec=2.4,
                quality_score=0.91,
            ),
            VoicePhase(
                id="v2",
                label="Story read",
                path="",
                waveform_peaks=[
                    0.1, 0.3, 0.6, 0.4, 0.5, 0.8, 0.7, 0.6, 0.4, 0.2,
                ],
                duration_sec=5.1,
                quality_score=0.55,
            ),
        ],
        other=OtherData(
            display_name="Jeff",
            notes="Prefers casual tone; avatar scale 1.0",
            preferences={"lip_sync": "standard", "idle_motion": True},
        ),
    )
    save_profile(profile)


def ensure_media_placeholders() -> None:
    scan_dir = MEDIA_DIR / "jeff_molofee" / "scan"
    scan_dir.mkdir(parents=True, exist_ok=True)
    svgs = {
        "front.svg": "#4a90d9",
        "left.svg": "#50c878",
        "right.svg": "#e07a4f",
    }
    for name, color in svgs.items():
        path = scan_dir / name
        if not path.exists():
            path.write_text(
                f'<svg xmlns="http://www.w3.org/2000/svg" width="320" height="400">'
                f'<rect width="100%" height="100%" fill="{color}"/>'
                f'<text x="50%" y="50%" text-anchor="middle" fill="white" '
                f'font-family="sans-serif" font-size="20">{name}</text></svg>',
                encoding="utf-8",
            )