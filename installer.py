"""Everything install.bat does after the Python environment exists: the chat brain, the video renderer, small model files.

  python installer.py           install whatever is missing (each step is skipped when already done)
  python installer.py --check   exit 0 if everything is installed, 1 (and list what is missing) if not
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
YUNET = ROOT / "ml" / "realtime" / "models" / "face_detection_yunet_2023mar.onnx"
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
FH = ROOT / "ml" / "realtime" / "flashhead"


def missing() -> list[str]:
    out = []
    models = ROOT / "brain" / "models"
    if os.environ.get("AVATAR_VRAM_GB"):  # testing a smaller card: that tier's own model is needed
        sys.path.insert(0, str(ROOT))
        from app import hardware

        have_model = (models / hardware.tier()["brain_file"]).exists()
    else:
        have_model = any(models.glob("*.gguf"))
    if not (ROOT / "brain" / "bin" / "llama-server.exe").exists() or not have_model:
        out.append("chat brain")
    sys.path.insert(0, str(ROOT / "ml" / "realtime"))
    import setup_flashhead

    if not setup_flashhead.complete():
        out.append("video renderer")
    if not YUNET.exists():
        out.append("face detector")
    return out


# What the page shows while the app sets itself up: state is wait | download | setup | ok
STATUS = {"brain": {"state": "wait", "pct": 0}, "video": {"state": "wait", "pct": 0}}


def _install_brain(cb) -> None:
    sys.path.insert(0, str(ROOT / "brain"))
    import setup_brain

    setup_brain.main(cb)


def _install_video(cb) -> None:
    sys.path.insert(0, str(ROOT / "ml" / "realtime"))
    import setup_flashhead

    setup_flashhead.main(sys.executable, cb)


def start_background() -> None:
    """Install whatever is missing in a background thread; STATUS tells the page how each part is getting on."""

    def job() -> None:
        def cb_for(key):
            def cb(state, pct=0):
                STATUS[key].update(state=state, pct=pct)

            return cb

        try:
            if "face detector" in missing():
                YUNET.parent.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(YUNET_URL, YUNET)
            for key, name, fn in (("brain", "chat brain", _install_brain), ("video", "video renderer", _install_video)):
                if name in missing():
                    try:
                        fn(cb_for(key))
                    except BaseException as exc:  # shown on the page; the rest keeps going
                        print(f"[install] {name} failed: {exc}")
                        STATUS[key].update(state="failed", pct=0)
                        continue
                STATUS[key].update(state="ok", pct=100)
        except BaseException as exc:
            print(f"[install] failed: {exc}")

    threading.Thread(target=job, daemon=True, name="installer").start()


def main() -> int:
    if "--check" in sys.argv:
        m = missing()
        if m:
            print("Missing: " + ", ".join(m))
        return 1 if m else 0
    if "face detector" in missing():
        YUNET.parent.mkdir(parents=True, exist_ok=True)
        print("Downloading the face detector...")
        urllib.request.urlretrieve(YUNET_URL, YUNET)
    if "chat brain" in missing():
        print("\n== Chat brain (about 8 GB) ==")
        subprocess.run([sys.executable, str(ROOT / "brain" / "setup_brain.py")], check=True)
    if "video renderer" in missing():
        print("\n== Video renderer (about 8 GB) ==")
        subprocess.run([sys.executable, str(ROOT / "ml" / "realtime" / "setup_flashhead.py"), sys.executable], check=True)
    m = missing()
    print("\nEverything is installed." if not m else "\nStill missing: " + ", ".join(m))
    return 1 if m else 0


if __name__ == "__main__":
    raise SystemExit(main())
