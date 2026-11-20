"""Prepare a person's photo for the photo-real avatar (live idle face, reply videos, optional clips).

Works from whatever the person has: a single photo is enough; a high-resolution wide scan frame is better.
The result is data/media/<user>/cinematic/source.jpg, the one picture every photo-real layer is built from, so the
framing never changes between the idle face, the pre-made clips and the live reply videos.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from app.capture import _user_media_dir
from app.profile_logic import face_assets, pick_avatar_face
from app.storage import MEDIA_DIR, get_working_profile


def cinematic_dir(canonical_name: str) -> Path:
    return MEDIA_DIR / _user_media_dir(canonical_name) / "cinematic"


def _clean_borders(img: np.ndarray) -> np.ndarray:
    """Cut away black bars (camera frame edges) and the black corners left by circular scan crops."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    content = gray > 12
    rows = np.where(content.mean(axis=1) > 0.05)[0]
    cols = np.where(content.mean(axis=0) > 0.05)[0]
    if len(rows) and len(cols):
        img = img[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
    h, w = img.shape[:2]
    # circular crop? the four corners are black while the middle is not: fill the corners so the picture is a full
    # rectangle (keeping the face small enough inside the frame for the face finder to see it)
    k = max(4, min(h, w) // 12)
    corners = [img[:k, :k], img[:k, -k:], img[-k:, :k], img[-k:, -k:]]
    if all(float(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY).mean()) < 14 for c in corners):
        side = min(h, w)
        img = img[(h - side) // 2 : (h - side) // 2 + side, (w - side) // 2 : (w - side) // 2 + side]
        yy, xx = np.mgrid[0:side, 0:side]
        outside = ((xx - side / 2) ** 2 + (yy - side / 2) ** 2 > (side / 2 - 2) ** 2).astype(np.uint8) * 255
        img = cv2.inpaint(img, outside, 6, cv2.INPAINT_TELEA)
    # small photos: enlarge so the later steps have enough pixels to work with (they will be soft; a rescan fixes that)
    h, w = img.shape[:2]
    if min(h, w) < 640:
        f = 720 / min(h, w)
        img = cv2.resize(img, (round(w * f), round(h * f)), interpolation=cv2.INTER_CUBIC)
    return img


def ensure_source(canonical_name: str, force: bool = False) -> Path | None:
    """Return the prepared source photo, creating it from the person's best scan photo if needed."""
    out = cinematic_dir(canonical_name) / "source.jpg"
    if out.exists() and not force:
        return out
    try:
        profile = get_working_profile(canonical_name)
        face = pick_avatar_face(face_assets(profile))
        if face is None or not Path(face.path).exists():
            return None
        img = cv2.imread(str(face.path))
        if img is None:
            return None
        img = _clean_borders(img)
        h, w = img.shape[:2]
        if min(h, w) < 96:
            return None
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        (out.parent / "source_info.json").write_text(
            json.dumps({"from": Path(face.path).name, "width": w, "height": h}, indent=2), encoding="utf-8"
        )
        return out
    except Exception:
        return None


def status(canonical_name: str) -> dict:
    """What the page needs to know: is there a source photo, and a pre-made clip library?"""
    src = ensure_source(canonical_name)
    d = cinematic_dir(canonical_name)
    return {"has_source": bool(src), "has_clips": (d / "manifest.json").exists(), "source_px": _px(d / "source_info.json")}


def _px(info: Path) -> int:
    try:
        j = json.loads(info.read_text(encoding="utf-8"))
        return int(min(j["width"], j["height"]))
    except Exception:
        return 0
