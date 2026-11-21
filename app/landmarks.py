"""Server-side face landmarks for the avatar.

Browsers differ (privacy settings, GPU/CPU delegates) in whether MediaPipe can analyse a still
photo — in some, it hangs the whole page. So the server finds the landmarks once with MediaPipe
for Python, caches them next to the photo, and the page just uses the numbers.

Needs a Python with mediapipe + opencv: the app's own environment has both.
Override with the AVATAR_LANDMARK_PYTHON / AVATAR_LANDMARK_MODEL environment variables.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "detect_landmarks.py"
_APP_PY = Path(sys.executable)


def _python() -> Path | None:
    for cand in (os.environ.get("AVATAR_LANDMARK_PYTHON"), str(_APP_PY)):
        if cand and Path(cand).exists():
            return Path(cand)
    return None


def _model() -> Path | None:
    for cand in (os.environ.get("AVATAR_LANDMARK_MODEL"), str(ROOT / "static" / "vendor" / "mediapipe" / "face_landmarker.task")):
        if cand and Path(cand).exists():
            return Path(cand)
    return None
