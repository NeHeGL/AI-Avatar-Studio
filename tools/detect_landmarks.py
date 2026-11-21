"""Detect MediaPipe face landmarks in one image; print JSON to stdout.

Run with a Python that has mediapipe + opencv (the app's own environment):
    python detect_landmarks.py <image> <face_landmarker.task>

Landmarks are normalized to the ORIGINAL image (x, y in 0..1; z in units of image width,
like MediaPipe's own output). Close-ups that fill the frame confuse the face detector,
so each photo is tried at a few sizes, including zoomed out with padding.
"""
import json
import sys

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision


def main() -> int:
    image_path, model_path = sys.argv[1], sys.argv[2]
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        print(json.dumps({"error": f"cannot read image: {image_path}"}))
        return 1
    h, w = bgr.shape[:2]
    landmarker = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=mpp.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.2,
            min_face_presence_confidence=0.2,
        )
    )
    for size, pad in ((384, 1), (384, 2), (256, 1), (640, 1), (512, 2)):
        k = size / max(w, h)
        sw, sh = max(1, round(w * k)), max(1, round(h * k))
        small = cv2.resize(bgr, (sw, sh), interpolation=cv2.INTER_AREA)
        canvas = np.full((sh * pad, sw * pad, 3), 58, dtype=np.uint8)
        ox, oy = (sw * (pad - 1)) // 2, (sh * (pad - 1)) // 2
        canvas[oy : oy + sh, ox : ox + sw] = small
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not res.face_landmarks:
            continue
        o = (pad - 1) / 2
        pts = [[p.x * pad - o, p.y * pad - o, p.z * pad] for p in res.face_landmarks[0]]
        print(json.dumps({"w": w, "h": h, "variant": f"{size}px pad{pad}", "landmarks": pts}))
        return 0
    print(json.dumps({"error": "no face found"}))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
