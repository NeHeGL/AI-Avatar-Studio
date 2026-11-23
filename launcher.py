"""AI Avatar Studio.exe: unpacks the app next to itself, sets up Python and the packages the first time, then starts it.

The .exe carries the app's own files (payload.zip) and the `uv` tool (uv.exe), so nothing has to be installed first, not even Python. It is
built by build_standalone.py. Put it in the folder (or on the drive) where you want the app and its downloads to live (about 30 GB in all): it
creates an "AI Avatar Studio" folder beside itself.

  "AI Avatar Studio.exe"              set up whatever is missing, then start the app
  "AI Avatar Studio.exe" --check      unpack and report what would be done; installs nothing, starts nothing
  "AI Avatar Studio.exe" --dry-run    also create the Python environment and resolve the packages, without downloading them
  "AI Avatar Studio.exe" --install-only   set up the Python environment and packages, but do not start the app
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

FROZEN = bool(getattr(sys, "frozen", False))
BUNDLE = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
EXE_DIR = Path(sys.executable).resolve().parent if FROZEN else Path(__file__).resolve().parent
PYTHON_VERSION = "3.11"
TORCH = ["torch==2.11.0", "torchaudio==2.11.0", "torchvision==0.26.0"]
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"  # CUDA build: RTX 20-series through 50-series
MUST_IMPORT = ("fastapi", "torch", "qwen_tts", "transformers", "faster_whisper")


def say(msg: str = "") -> None:
    print(msg, flush=True)


def app_dir() -> Path:
    if (EXE_DIR / "start_all.py").exists():  # the exe sits inside an existing copy of the app (a development checkout)
        return EXE_DIR
    return EXE_DIR / "AI Avatar Studio"


def payload() -> zipfile.ZipFile | None:
    z = BUNDLE / "payload.zip"
    return zipfile.ZipFile(z) if z.exists() else None


def unpack(app: Path) -> str:
    """Write the app's files into `app` (only when they are not already this version). Never touches data/, brain/models or ml/.venv."""
    z = payload()
    if z is None:
        return "dev"
    with z:
        version = json.loads(z.read("release.json"))["version"]
        marker = app / ".release"
        if marker.exists() and marker.read_text(encoding="utf-8").strip() == version:
            say(f"  [OK] The app files are already version {version}.")
            return version
        say(f"  Unpacking the app (version {version}) into {app} ...")
        n = 0
        for info in z.infolist():
            if info.is_dir() or info.filename == "release.json":
                continue
            dest = app / info.filename
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(z.read(info))
            n += 1
        marker.write_text(version, encoding="utf-8")
        say(f"  [OK] {n} files unpacked.")
    return version


def uv_exe() -> str:
    bundled = BUNDLE / "uv.exe"
    if bundled.exists():
        return str(bundled)
    found = shutil.which("uv")
    if not found:
        raise SystemExit("The uv tool is missing from this build.")
    return found


def run(cmd: list[str], cwd: Path, what: str) -> None:
    rc = subprocess.run(cmd, cwd=cwd).returncode
    if rc != 0:
        raise SystemExit(f"\n[ERROR] {what} failed (code {rc}). Check your internet connection and try again; it picks up where it left off.")


def env_ready(app: Path) -> bool:
    py = app / "ml" / ".venv" / "Scripts" / "python.exe"
    if not py.exists():
        return False
    code = "import importlib.util as u,sys;sys.exit(0 if all(u.find_spec(m) for m in %r) else 1)" % (MUST_IMPORT,)
    return subprocess.run([str(py), "-c", code], capture_output=True).returncode == 0


def bootstrap(app: Path, dry: bool) -> None:
    """Python 3.11 and every package the app needs, in one environment (ml/.venv). Each step is skipped when already done."""
    uv = uv_exe()
    venv = app / "ml" / ".venv"
    py = venv / "Scripts" / "python.exe"
    flag = ["--dry-run"] if dry else []
    say("  [1/4] Python " + PYTHON_VERSION + " and the environment (ml\\.venv) ...")
    if not py.exists():
        run([uv, "venv", "--python", PYTHON_VERSION, "--seed", str(venv)], app, "Creating the Python environment")
    else:
        say("        already there.")
    base = [uv, "pip", "install", "--python", str(py)] + flag
    say("  [2/4] PyTorch (about 3 GB, one time) ...")
    have_torch = subprocess.run([str(py), "-c", "import torch"], capture_output=True).returncode == 0
    if have_torch:
        say("        already installed.")
    else:
        run(base + TORCH + ["--index-url", TORCH_INDEX], app, "Installing PyTorch")
    say("  [3/4] The app's Python packages ...")
    run(base + ["-r", "requirements.txt"], app, "Installing the Python packages")
    run(base + ["--no-deps", "qwen-tts==0.1.1"], app, "Installing the voice package")
    run(base + ["--no-deps", "faster-whisper"], app, "Installing the speech package")
    say("  [4/4] The chat model, video engine and face detector (about 22 GB) are downloaded by the app itself, with progress on its start screen.")


def stop_old_servers() -> None:
    ps = "foreach($p in 8765,8770,8771,8772){Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }}"
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True)


def start(app: Path) -> int:
    env = dict(os.environ, AVATAR_WINDOW="app", AVATAR_LLM_MODE="relaxed")
    say("Closing any old AI Avatar Studio servers first ...")
    stop_old_servers()
    say("Starting AI Avatar Studio (the chat model and video engine start in the background) ...")
    return subprocess.run([str(app / "ml" / ".venv" / "Scripts" / "python.exe"), "start_all.py"], cwd=app, env=env).returncode


def main() -> int:
    args = set(sys.argv[1:])
    say("=" * 60)
    say("  AI Avatar Studio")
    say("=" * 60)
    app = app_dir()
    version = unpack(app)
    if "--check" in args:
        say(f"\nCheck only: app folder {app} (version {version}); Python environment ready: {env_ready(app)}; uv: {uv_exe()}")
        return 0
    if not env_ready(app) or "--dry-run" in args:
        bootstrap(app, dry="--dry-run" in args)
        if "--dry-run" in args:
            say("\nDry run finished: the environment resolves. Nothing was downloaded or started.")
            return 0
    if "--install-only" in args:
        say("\nInstall finished: the environment is ready. Start the app by running this program again.")
        return 0
    rc = start(app)
    if rc != 0:
        say(f"\nAI Avatar Studio stopped with an error (code {rc}). See the messages above.")
        if sys.stdin and sys.stdin.isatty():
            input("Press Enter to close ...")
    return rc


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except SystemExit as e:
        if isinstance(e.code, str):
            say(e.code)
            if sys.stdin and sys.stdin.isatty():
                input("Press Enter to close ...")
            raise SystemExit(1)
        raise
