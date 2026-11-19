"""Add existing audio clips and photos to a profile (create it if needed).

    python tools/import_media.py --name "Astro Boy" --voice clip1.wav clip2.mp3 --images photo1.jpg photo2.png
    python tools/import_media.py --name "Dad" --images old_photo.jpg --voice voicemail.m4a

- The FIRST image becomes the profile's straight-on "Center" photo (what the avatar video is made from),
  so list your best, most front-facing, sharpest photo first. Others are kept as extra reference views.
- Voice clips are stored as given; the voice cloner wants clean speech, ideally 10+ seconds in total.
- Only use material you have the right to use (your own recordings and photos, or family material).
"""
import argparse
import base64
import mimetypes
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import capture  # noqa: E402
from app.storage import commit_draft, get_or_create_draft, get_working_profile  # noqa: E402
from app.walkthrough import profile_summary  # noqa: E402


def data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def expand(paths):
    out = []
    for p in map(Path, paths or []):
        out += sorted(x for x in p.iterdir() if x.is_file()) if p.is_dir() else [p]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--voice", nargs="*")
    ap.add_argument("--images", nargs="*")
    a = ap.parse_args()

    get_or_create_draft(a.name)
    images = [p for p in expand(a.images) if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")]
    if images:
        frames = []
        for i, p in enumerate(images):
            frames.append({"image_base64": data_url(p), "caption": "Center" if i == 0 else f"Reference {i}",
                           "angle_index": 32 if i == 0 else i - 1, "quality_score": 1.0 if i == 0 else 0.8})
        r = capture.ingest_scan_frames(a.name, frames)
        print("photos:", r.get("status"), len(images), "added", r.get("message", ""))
    for p in expand(a.voice):
        r = capture.ingest_voice(a.name, data_url(p), p.stem)
        print("voice:", p.name, "->", r.get("status"))
    commit_draft(a.name)
    s = profile_summary(get_working_profile(a.name))
    print(f"profile '{s['display_name']}': {len(s['scan_frames'])} photos, voice={'yes' if s['avatar_voice_url'] else 'no'}, "
          f"avatar face={s['avatar_face_url'] or 'none'}")


if __name__ == "__main__":
    main()
