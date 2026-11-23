"""Score photos of a face and pick the best one to build the avatar from.

For each photo: is a face found, is the whole head in the picture, how sharp, how frontal, are the eyes open, is there a smile, how big.
When NO face can be found in any photo (drawings and cartoons often fail the face detector), it falls back to sharpness, size and shape,
and says so, because the face animation may not work on such pictures.
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2

_MODEL = Path(__file__).resolve().parent.parent / "static" / "vendor" / "mediapipe" / "face_landmarker.task"
_landmarker = None


def _lm():
    global _landmarker
    if _landmarker is None:
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision

        opts = vision.FaceLandmarkerOptions(
            base_options=mpp.BaseOptions(model_asset_path=str(_MODEL)),
            output_face_blendshapes=True, output_facial_transformation_matrixes=True, num_faces=1,
        )
        _landmarker = vision.FaceLandmarker.create_from_options(opts)
    return _landmarker


def _blend_and_pose(img) -> dict:
    """Blendshapes (smile, blink, jaw) and how far the head is turned, from MediaPipe. Empty when it finds no face."""
    try:
        import mediapipe as mp

        small = img
        h, w = img.shape[:2]
        k = min(1.0, 1280 / max(h, w))
        if k < 1:
            small = cv2.resize(img, (round(w * k), round(h * k)))
        r = _lm().detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
        if not r.face_blendshapes:
            return {}
        b = {c.category_name: c.score for c in r.face_blendshapes[0]}
        m = r.facial_transformation_matrixes[0]
        return {
            "smile": (b.get("mouthSmileLeft", 0) + b.get("mouthSmileRight", 0)) / 2,
            "blink": (b.get("eyeBlinkLeft", 0) + b.get("eyeBlinkRight", 0)) / 2,
            "jaw": b.get("jawOpen", 0),
            "turn": abs(float(m[0][2])) + 0.5 * abs(float(m[1][2])),  # 0 = looking straight at the camera
        }
    except Exception:
        return {}


def _sharpness(img) -> float:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    k = min(1.0, 800 / max(h, w))  # compare at a common size, so a big photo is not "sharper" just for being big
    if k < 1:
        g = cv2.resize(g, (round(w * k), round(h * k)))
    v = float(cv2.Laplacian(g, cv2.CV_64F).var())
    return min(1.0, math.log10(1 + v) / 3.0)


def score_photo(img) -> dict:
    """{score 0-100, face, head_in_frame, sharp, smile, blink, turn, px, note}. Higher is better."""
    from app.photoreal import head_in_frame

    h, w = img.shape[:2]
    px = min(h, w)
    sharp = _sharpness(img)
    res = min(1.0, px / 1024)
    ok, note = head_in_frame(img)
    out = {"face": False, "head_in_frame": ok, "sharp": round(sharp, 2), "px": px, "note": note}
    pose = _blend_and_pose(img)
    if not pose:
        # no face found: only sharpness and size can be judged
        out["score"] = round(25 * sharp + 15 * res, 1)  # capped well below any photo with a real face
        out["note"] = "no face found"
        return out
    out.update(face=True, smile=round(pose["smile"], 2), blink=round(pose["blink"], 2), turn=round(pose["turn"], 2))
    frontal = max(0.0, 1.0 - pose["turn"] / 0.4)
    eyes = max(0.0, 1.0 - pose["blink"] / 0.5)
    mouth = 1.0 if pose["jaw"] < 0.25 else 0.4  # a closed or lightly smiling mouth, not mid-word
    score = 35 * frontal + 22 * eyes + 15 * min(1.0, pose["smile"] / 0.5) + 13 * sharp + 8 * res + 7 * mouth
    if not ok:
        score -= 30  # the head touches the edge of the picture
    out["score"] = round(max(0.0, score), 1)
    return out


def rank_photos(paths: list[Path]) -> list[dict]:
    """Score every photo; best first. Each entry has 'path' plus the fields of score_photo()."""
    rows = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        r = score_photo(img)
        r["path"] = str(p)
        rows.append(r)
    rows.sort(key=lambda r: -r["score"])
    return rows

