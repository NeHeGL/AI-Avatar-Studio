"""Build the release: dist/AI-Avatar-Studio-<version>-windows-x64.zip, holding AI-Avatar-Studio.exe and README.md (the one file a GitHub release carries).

  python build_standalone.py [version]        (or double-click "Build Standalone.bat")

The .exe is launcher.py packed with PyInstaller. Inside it: the app's own files (payload.zip) and the `uv` tool, which sets up Python 3.11 and the
packages on first run. It holds none of your people, chats, voice samples or logs, and none of the large downloads (the Python packages, about
3 GB of PyTorch, and the chat model, video engine and voice, about 22 GB): those are fetched when it is first run.
The GitHub "Release" workflow runs this and attaches the zip to the release (the source code is the repository itself). Needs `uv` (https://docs.astral.sh/uv/) on the PATH.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIST = ROOT / "dist"
BUILD = ROOT / "build"
NAME = "AI-Avatar-Studio"

FILES = [
    "installer.py", "run.py", "start_all.py", "requirements.txt", "README.md", "LICENSE",
    "brain/setup_brain.py",
    "ml/realtime/flashhead_render_server.py", "ml/realtime/setup_flashhead.py",
]
DIRS = ["app", "static", "ml/realtime/flashhead_stubs"]
SKIP_PARTS = {"__pycache__"}
SKIP_SUFFIX = {".pyc"}


def payload(version: str) -> Path:
    BUILD.mkdir(exist_ok=True)
    out = BUILD / "payload.zip"
    paths = [ROOT / f for f in FILES if (ROOT / f).exists()]
    for d in DIRS:
        paths += [p for p in sorted((ROOT / d).rglob("*")) if p.is_file() and not (SKIP_PARTS & set(p.parts)) and p.suffix not in SKIP_SUFFIX]
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            z.write(p, p.relative_to(ROOT).as_posix())
        z.writestr("release.json", json.dumps({"version": version, "built": int(time.time())}))
    print(f"payload: {len(paths)} files, {out.stat().st_size / 1e6:.1f} MB")
    return out


def main() -> int:
    version = sys.argv[1] if len(sys.argv) > 1 else "dev"
    uv = shutil.which("uv")
    if not uv:
        print("uv is needed to build (https://docs.astral.sh/uv/).")
        return 1
    pay = payload(version)
    DIST.mkdir(exist_ok=True)
    cmd = [
        uv, "tool", "run", "--python", "3.11", "pyinstaller", "--noconfirm", "--clean", "--onefile", "--console", "--name", NAME,
        "--distpath", str(DIST), "--workpath", str(BUILD / "pyinstaller"), "--specpath", str(BUILD),
        "--add-data", f"{pay};.", "--add-binary", f"{uv};.", str(ROOT / "launcher.py"),
    ]
    if subprocess.run(cmd, cwd=ROOT).returncode != 0:
        return 1
    shutil.copyfile(ROOT / "README.md", DIST / "README.md")
    exe = DIST / f"{NAME}.exe"
    zip_path = DIST / f"{NAME}-{version}-windows-x64.zip"  # what the GitHub release carries: the .exe and the README, in one zip
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(exe, exe.name)
        z.write(DIST / "README.md", "README.md")
    print(f"\nBuilt {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB), holding {exe.name} ({exe.stat().st_size / 1e6:.1f} MB) and README.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
