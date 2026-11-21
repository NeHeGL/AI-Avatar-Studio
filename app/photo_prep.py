"""Look at a photo of a face and prepare it for the avatar: find the face, level it, centre it, zoom to a good size, crop to a square,
and say in plain words whether the photo is good enough (and why not, when it is not).

`analyze(img)` returns one dictionary for a photo:
    ok            True when nothing stops it being used
    verdict       "good" | "usable" (works, but a better photo would help) | "bad" (do not use)
    issues        [{"level": "bad"|"warn", "text": "..."}]  plain-language reasons
    score         0-100, higher is better (used to pick the best of several photos)
    crop          the prepared square picture (BGR numpy array) or None when no face was found
    face          numbers: face height in pixels, head turn, tilt, eyes, sharpness, brightness
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

_MODEL = Path(__file__).resolve().parent.parent / "static" / "vendor" / "mediapipe" / "face_landmarker.task"
_landmarker = None

OUT_PX = 1024  # prepared picture size
FACE_FRACTION = 0.42  # the face (forehead to chin) fills about this much of the picture's height: head, hair and shoulders around it


def _lm():
    global _landmarker
    if _landmarker is None:
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision

        _landmarker = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=mpp.BaseOptions(model_asset_path=str(_MODEL)),
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
                num_faces=2,
                min_face_detection_confidence=0.4,
            )
        )
    return _landmarker


def _detect(img):
    """MediaPipe on a picture reduced to at most 1600 px. Returns (result, scale) where landmark pixels = normalized * original size."""
    import mediapipe as mp

    h, w = img.shape[:2]
    k = min(1.0, 1600 / max(h, w))
    small = cv2.resize(img, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA) if k < 1 else img
    r = _lm().detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
    return r


def _sharpness(gray: np.ndarray) -> float:
    """0-1, how sharp the face is. Variance of the Laplacian at a fixed size (so a big photo is not 'sharper' for being big), measured
    after a light blur: webcam noise and compression speckle look like detail to a plain Laplacian and make a soft picture score as
    sharp, while real edges survive the light blur. Roughly: 0.4+ sharp, 0.3 a little soft, under 0.22 blurry."""
    g = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), 1.2)
    v = float(cv2.Laplacian(g, cv2.CV_32F).var())
    return min(1.0, math.log10(1 + v) / 2.2)


def _issue(level: str, text: str) -> dict:
    return {"level": level, "text": text}


def analyze(img: np.ndarray, make_crop: bool = True) -> dict:
    h, w = img.shape[:2]
    out = {"ok": False, "verdict": "bad", "issues": [], "score": 0, "crop": None, "face": {}}
    issues = out["issues"]
    if min(h, w) < 300:
        issues.append(_issue("bad", f"The photo is very small ({w}x{h}). Use one at least 600 pixels on its short side."))
    r = _detect(img)
    if not r.face_landmarks:
        issues.append(_issue("bad", "No face was found. Use a photo where the face is clearly visible, looking toward the camera, without anything covering it."))
        return out
    faces = []
    for i, lms in enumerate(r.face_landmarks):
        xs = np.array([p.x for p in lms]) * w
        ys = np.array([p.y for p in lms]) * h
        faces.append((float((xs.max() - xs.min()) * (ys.max() - ys.min())), i))
    faces.sort(reverse=True)
    if len(faces) > 1 and faces[1][0] > 0.25 * faces[0][0]:
        issues.append(_issue("warn", "More than one face is in the photo. The largest one is used; a photo of just you is better."))
    fi = faces[0][1]
    lms = r.face_landmarks[fi]
    P = np.array([[p.x * w, p.y * h] for p in lms], dtype=np.float32)
    bs = {c.category_name: c.score for c in r.face_blendshapes[fi]} if r.face_blendshapes else {}
    mat = np.array(r.facial_transformation_matrixes[fi]) if r.facial_transformation_matrixes else None

    # ---- head turn, from the transform matrix (degrees; 0 = looking straight at the camera)
    yaw = pitch = 0.0
    if mat is not None:
        R = mat[:3, :3]
        yaw = math.degrees(math.atan2(R[0, 2], R[2, 2]))
        pitch = math.degrees(math.asin(max(-1.0, min(1.0, -R[1, 2]))))
    # ---- tilt of the head in the picture, from the eye corners (the picture is levelled below)
    le, re = P[33], P[263]
    roll = math.degrees(math.atan2(float(re[1] - le[1]), float(re[0] - le[0])))
    if re[0] < le[0]:  # a mirrored or upside-down face: leave it alone
        roll = 0.0

    # ---- level the picture so the eyes are horizontal, then measure the face again on the levelled picture
    centre = (float(P[:, 0].mean()), float(P[:, 1].mean()))
    if abs(roll) > 1.5 and abs(roll) < 40:
        M = cv2.getRotationMatrix2D(centre, roll, 1.0)
        img2 = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        ones = np.ones((len(P), 1), np.float32)
        P = (np.hstack([P, ones]) @ M.T).astype(np.float32)
    else:
        img2 = img
    x0, x1 = float(P[:, 0].min()), float(P[:, 0].max())
    y0, y1 = float(P[:, 1].min()), float(P[:, 1].max())
    top, chin = float(P[10, 1]), float(P[152, 1])
    face_h = max(chin - top, 1.0)  # forehead to chin
    face_w = x1 - x0
    face_cx = (float(P[234, 0]) + float(P[454, 0])) / 2
    face_cy = (top + chin) / 2

    # ---- crop: a square around the face, centred on it, sized so the face fills FACE_FRACTION of the height (never larger than the photo)
    S = face_h / FACE_FRACTION
    S = min(S, h, w)
    # the person's head sits a little above the middle of a portrait: put the face centre slightly above the crop centre
    cx = face_cx
    cy = face_cy + 0.04 * S
    left = int(round(cx - S / 2))
    topc = int(round(cy - S / 2))
    left = max(0, min(left, w - int(S)))
    topc = max(0, min(topc, h - int(S)))
    side = int(S)
    sq = img2[topc : topc + side, left : left + side]
    if sq.size == 0:
        issues.append(_issue("bad", "The face could not be cropped."))
        return out
    # how much of the crop the face really fills, after the photo's own edges limited the zoom
    fill = face_h / side
    touching = (x0 < 0.02 * w) or (x1 > 0.98 * w) or (top < 0.01 * h) or (chin > 0.99 * h)
    face_px = float(face_h)
    if touching:
        issues.append(_issue("bad", "Part of the face is cut off by the edge of the photo. Use one with the whole head inside the picture."))
    if face_px < 120:
        issues.append(_issue("bad", f"The face is too small in the photo (about {int(face_px)} pixels tall). Take it closer, or use a larger photo."))
    elif face_px < 200:
        issues.append(_issue("warn", f"The face is on the small side ({int(face_px)} pixels tall), so the picture will be a little soft. A closer or larger photo is better."))
    if fill > 0.75 and not touching:
        issues.append(_issue("warn", "The face fills most of the photo, so there is no room for hair and shoulders around it."))

    # ---- focus, light, eyes, head turn
    gface = cv2.cvtColor(img2[max(0, int(top)) : int(chin), max(0, int(x0)) : int(x1)], cv2.COLOR_BGR2GRAY)
    sharp = _sharpness(gface) if gface.size else 0.0
    bright = float(gface.mean()) if gface.size else 0.0
    contrast = float(gface.std()) if gface.size else 0.0
    if sharp < 0.22:
        issues.append(_issue("bad", "The photo is blurry. Use one that is in focus (hold the camera steady, and clean the lens)."))
    elif sharp < 0.32:
        issues.append(_issue("warn", "The photo is a little soft. A sharper one will animate better."))
    if bright < 55:
        issues.append(_issue("bad", "The face is too dark. Use a photo with the light on your face, not behind you."))
    elif bright < 80:
        issues.append(_issue("warn", "The face is on the dark side. More even light will look better."))
    elif bright > 225:
        issues.append(_issue("bad", "The face is washed out by too much light."))
    elif bright > 205:
        issues.append(_issue("warn", "The face is very bright; some detail is lost."))
    if contrast < 18:
        issues.append(_issue("warn", "The face looks flat, with little light and shade."))
    blink = (bs.get("eyeBlinkLeft", 0) + bs.get("eyeBlinkRight", 0)) / 2
    if blink > 0.55:
        issues.append(_issue("bad", "The eyes are closed. Use a photo with your eyes open."))
    elif blink > 0.35:
        issues.append(_issue("warn", "The eyes look half closed."))
    look = max(abs(yaw), abs(pitch) * 1.3)
    if look > 28:
        issues.append(_issue("bad", "The head is turned away. Use a photo where you look straight at the camera."))
    elif look > 14:
        issues.append(_issue("warn", "The head is turned a little. Looking straight at the camera works best."))
    jaw = bs.get("jawOpen", 0)
    if jaw > 0.35:
        issues.append(_issue("warn", "The mouth is wide open. A relaxed face works best as the starting picture."))
    if abs(roll) > 20:
        issues.append(_issue("warn", f"The head is tilted ({abs(roll):.0f} degrees). It has been levelled, but a straight photo is better."))

    # ---- score: frontal, eyes open, sharp, big, well lit
    s = 100.0
    s -= min(40, look * 1.2)
    s -= 30 * min(1, blink / 0.5)
    s -= 35 * max(0.0, 0.42 - sharp) / 0.42
    s -= 25 * max(0.0, 300 - face_px) / 300
    s -= 15 * (max(0.0, 110 - bright) / 110 + max(0.0, bright - 190) / 65)
    s -= 8 * min(1, jaw / 0.4)
    out["score"] = int(max(0, min(100, round(s))))
    out["face"] = {
        "face_px": int(face_px), "yaw": round(yaw, 1), "pitch": round(pitch, 1), "roll": round(roll, 1),
        "blink": round(blink, 2), "sharp": round(sharp, 2), "brightness": round(bright), "fill": round(fill, 2),
    }
    out["verdict"] = "bad" if any(i["level"] == "bad" for i in issues) else ("usable" if issues else "good")
    out["ok"] = out["verdict"] != "bad"
    if make_crop:
        outpx = min(OUT_PX, side) if side < OUT_PX else OUT_PX
        out["crop"] = cv2.resize(sq, (outpx, outpx), interpolation=cv2.INTER_AREA if side > outpx else cv2.INTER_CUBIC)
    return out


def best_of(images: list[np.ndarray]) -> tuple[int, dict]:
    """Analyze several photos (webcam snaps) and return (index of the best one, its analysis). Photos with a verdict of 'bad' only win
    when every photo is bad."""
    results = [analyze(im) for im in images]
    ranked = sorted(range(len(results)), key=lambda i: (results[i]["ok"], results[i]["score"]), reverse=True)
    i = ranked[0]
    return i, results[i]
