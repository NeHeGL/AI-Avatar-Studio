from __future__ import annotations

import json
import re
from pathlib import Path

from app.models import UserProfile

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
# One folder per person: data/profiles/<name>/{knowledge,media,voice}
#   knowledge/  what the avatar knows about the person: profile.json (saved) and draft.json (being edited)
#   media/  scan photos, cinematic source/clips, talking-reply videos
#   voice/  voice recordings, the reference clip and per-person voice settings
PROFILES_DIR = DATA_ROOT / "profiles"
# Old flat layout (data/users, data/drafts, data/media): only read by migrate_legacy_layout()
USERS_DIR = DATA_ROOT / "users"
DRAFTS_DIR = DATA_ROOT / "drafts"
MEDIA_DIR = DATA_ROOT / "media"


def normalize_name_key(name: str) -> str:
    cleaned = re.sub(r"\s+", " ", name.strip().lower())
    return cleaned


def folder_name(name: str) -> str:
    """Folder for a person: the normalized name with spaces as underscores ("jeff molofee" -> "jeff_molofee")."""
    return normalize_name_key(name).replace(" ", "_")


def profile_dir(name: str) -> Path:
    return PROFILES_DIR / folder_name(name)


def media_dir(name: str) -> Path:
    return profile_dir(name) / "media"


def voice_dir(name: str) -> Path:
    return profile_dir(name) / "voice"


def profile_path(key: str) -> Path:
    return profile_dir(key) / "knowledge" / "profile.json"


def draft_path(key: str) -> Path:
    return profile_dir(key) / "knowledge" / "draft.json"


def _saved_files() -> list[Path]:
    return sorted(PROFILES_DIR.glob("*/knowledge/profile.json")) if PROFILES_DIR.exists() else []


def find_by_display_or_canonical(name: str) -> UserProfile | None:
    """Saved profiles only (data/profiles/<name>/knowledge/profile.json). Drafts do not count as existing users."""
    key = normalize_name_key(name)
    direct = profile_path(key)
    if direct.exists():
        profile = UserProfile.model_validate_json(direct.read_text(encoding="utf-8"))
        if profile.saved:
            return profile
        return None
    for path in _saved_files():
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
    for path in _saved_files():
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
    key = normalize_name_key(profile.canonical_name)
    draft_path(key).parent.mkdir(parents=True, exist_ok=True)
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
    """Delete a profile completely: its saved profile, any draft, and its whole folder (knowledge, media, voice)."""
    import shutil

    key = normalize_name_key(full_name)
    deleted = False
    path = profile_path(key)
    if path.exists():
        path.unlink()
        deleted = True
    delete_draft(full_name)
    folder = profile_dir(key)
    if deleted and folder.is_dir() and PROFILES_DIR.resolve() in folder.resolve().parents:  # only ever inside data/profiles
        shutil.rmtree(folder, ignore_errors=True)
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

    key = normalize_name_key(profile.canonical_name)
    profile_path(key).parent.mkdir(parents=True, exist_ok=True)
    profile = reconcile_profile(profile)
    profile.saved = True
    profile.touch()
    profile_path(key).write_text(
        profile.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return profile


def purge_unsaved_user_files() -> None:
    """Turn mistaken early saves (a profile.json not marked saved) into drafts."""
    for path in _saved_files():
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
    scan_dir = media_dir("Jeff Molofee") / "scan"
    scan_dir.mkdir(parents=True, exist_ok=True)
    voice_dir("Jeff Molofee").mkdir(parents=True, exist_ok=True)

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
    scan_dir = media_dir("Jeff Molofee") / "scan"
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

def migrate_legacy_layout() -> None:
    """Move the old flat folders (users/, drafts/, media/) into data/profiles/<name>/{knowledge,media,voice}.

    Moves only, never deletes content. File paths stored inside the profile JSON are repointed to the new place."""
    import shutil

    def mv(src: Path, dst: Path) -> None:
        if src.exists() and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))

    if PROFILES_DIR.exists():  # earlier layout: the knowledge folder was called "data"
        for d in sorted(PROFILES_DIR.iterdir()):
            if d.is_dir() and (d / "data").is_dir() and not (d / "knowledge").exists():
                (d / "data").rename(d / "knowledge")
    if PROFILES_DIR.exists():  # first-generation layout used spaces in the folder names
        for d in sorted(PROFILES_DIR.iterdir()):
            if d.is_dir() and " " in d.name:
                new = PROFILES_DIR / d.name.replace(" ", "_")
                if not new.exists():
                    d.rename(new)
                    for jf in new.glob("knowledge/*.json"):
                        text = jf.read_text(encoding="utf-8")
                        for sep in ("\\\\", "/"):
                            text = text.replace(f"profiles{sep}{d.name}{sep}", f"profiles{sep}{new.name}{sep}")
                        jf.write_text(text, encoding="utf-8")

    def has_proper_twin(key: str) -> bool:
        spaced = key.replace("_", " ")
        return "_" in key and any((d / f"{spaced}.json").exists() for d in (USERS_DIR, DRAFTS_DIR))

    for folder, fname in ((USERS_DIR, "profile.json"), (DRAFTS_DIR, "draft.json")):
        if folder.exists():
            for f in sorted(folder.glob("*.json")):
                key = normalize_name_key(f.stem)
                if not has_proper_twin(key):  # an old underscore-named stub is left behind
                    mv(f, profile_dir(key) / "knowledge" / fname)
    if MEDIA_DIR.exists():
        for m in sorted(MEDIA_DIR.iterdir()):
            if m.is_dir():
                key = m.name.replace("_", " ")
                for sub in sorted(m.iterdir()):
                    mv(sub, voice_dir(key) if sub.name == "voice" else media_dir(key) / sub.name)

    def repoint(value, key: str, hits: list[bool]):
        if isinstance(value, dict):
            return {k: repoint(x, key, hits) for k, x in value.items()}
        if isinstance(value, list):
            return [repoint(x, key, hits) for x in value]
        if isinstance(value, str) and "media" in value:
            parts = re.split(r"[\\/]+", value)
            if "media" in parts:
                i = len(parts) - 1 - parts[::-1].index("media")
                rest = parts[i + 2:]  # what follows media/<person>/
                if rest:
                    new = voice_dir(key).joinpath(*rest[1:]) if rest[0] == "voice" else media_dir(key).joinpath(*rest)
                    if new.exists():
                        hits.append(True)
                        return str(new)
        return value

    if PROFILES_DIR.exists():
        for jf in PROFILES_DIR.glob("*/knowledge/*.json"):
            try:
                raw = json.loads(jf.read_text(encoding="utf-8"))
            except Exception:
                continue
            hits: list[bool] = []
            raw = repoint(raw, jf.parent.parent.name, hits)
            if hits:
                jf.write_text(json.dumps(raw, indent=2), encoding="utf-8")
