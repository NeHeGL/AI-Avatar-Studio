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


_YUNET = Path(__file__).resolve().parent.parent / "ml" / "realtime" / "models" / "face_detection_yunet_2023mar.onnx"


def _find_face(img: np.ndarray):
    """(x, y, w, h) of the most confident face, or None. A small local detector (YuNet): it only has to say where the head is."""
    try:
        if not _YUNET.exists():
            return None
        h, w = img.shape[:2]
        k = min(1.0, 960 / max(h, w))
        small = cv2.resize(img, (round(w * k), round(h * k))) if k < 1 else img
        det = cv2.FaceDetectorYN.create(str(_YUNET), "", (small.shape[1], small.shape[0]), 0.6, 0.3, 5000)
        _, faces = det.detect(small)
        if faces is None or not len(faces):
            return None
        f = max(faces, key=lambda r: float(r[-1]))
        return tuple(int(v / k) for v in f[:4])
    except Exception:
        return None


def head_in_frame(img: np.ndarray) -> tuple[bool, str]:
    """Is the whole head inside the picture (hair on top, chin below, both sides)?"""
    h, w = img.shape[:2]
    face = _find_face(img)
    if face is None:
        return False, "no face found"
    x, y, fw, fh = face
    if x - 0.25 * fw < 0 or x + 1.25 * fw > w or y - 0.5 * fh < 0 or y + fh > h:  # the chin must be inside; the lower neck may be cut
        return False, "head touches or crosses the edge"
    return True, "ok"


def _frame_head(img: np.ndarray) -> np.ndarray:
    """Tall or wide photos become a square at FULL size (no zooming out, no padding). A portrait keeps the top of the photo and
    loses its bottom, slid just far enough that the top of the head has a little room; a landscape is centred on the face.
    Roughly square photos are left exactly as they are."""
    h, w = img.shape[:2]
    if abs(w / h - 1.0) < 0.08:
        return img
    side = min(w, h)
    face = _find_face(img)
    if h > w:  # portrait: cut off the bottom, not the top
        y0 = 0
        if face:
            x, y, fw, fh = face
            y0 = int(min(max(y - 0.6 * fh - 0.06 * side, 0), h - side))  # top of the hair a little below the frame's top edge
        return img[y0 : y0 + side, 0:side]
    x0 = (w - side) // 2
    if face:
        x, y, fw, fh = face
        x0 = int(min(max(x + fw / 2 - side / 2, 0), w - side))
    return img[0:side, x0 : x0 + side]


def _noise_sigma(img: np.ndarray) -> float:
    """Rough grain level (standard deviation) of a picture."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hp = g - cv2.GaussianBlur(g, (0, 0), 1.5)
    return float(1.4826 * np.median(np.abs(hp - np.median(hp))))


def enhance_photo(img: np.ndarray) -> np.ndarray:
    """Clean up a grainy or colour-cast photo (a scanned print, an old snapshot) so the face model and the background
    cut-out have something smooth to work with. Clean photos are returned untouched."""
    sigma = _noise_sigma(img)
    h, w = img.shape[:2]
    m = img.reshape(-1, 3).mean(axis=0)
    if sigma <= 2.5 and m.max() / max(m.min(), 1.0) < 1.25:  # already clean, no strong colour cast: leave it exactly as it is
        return img
    out = img
    if sigma > 2.5:  # grain: non-local-means denoise (luma harder than colour), blended so skin keeps some texture
        strength = float(min(11.0, 2.5 + 2.6 * sigma))
        den = cv2.fastNlMeansDenoisingColored(out, None, strength, strength * 0.9, 7, 25)
        den = cv2.fastNlMeansDenoisingColored(den, None, strength * 0.4, strength * 0.4, 5, 15)  # a second, lighter pass
        out = cv2.addWeighted(den, 0.85, out, 0.15, 0)
        blur = cv2.GaussianBlur(out, (0, 0), max(1.2, w / 700))
        out = cv2.addWeighted(out, 1.45, blur, -0.45, 0)  # unsharp mask brings back edges (eyes, glasses, hairline)
    # keep the photo's own warmth: only correct a strong colour cast, and only slightly
    mean = out.reshape(-1, 3).mean(axis=0)
    gain = 1.0 + 0.1 * (mean.mean() / np.maximum(mean, 1.0) - 1.0)
    out = np.clip(out.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


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
    img = enhance_photo(_frame_head(img))
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
        head_ok, head_note = head_in_frame(img)
        (out.parent / "source_info.json").write_text(
            json.dumps({"from": Path(face.path).name, "width": w, "height": h, "head_in_frame": head_ok, "head_note": head_note}, indent=2),
            encoding="utf-8",
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
