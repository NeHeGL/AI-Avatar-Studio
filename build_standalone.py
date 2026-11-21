"""Build the downloadable release: dist/Avatar-Studio-<version>.zip

  python build_standalone.py [version]        (or double-click "Build Standalone.bat")

The zip holds the app, the launchers and the installers, in one "Avatar AI Studio" folder. It holds none of your
people, chats, voice samples or logs, and none of the large downloads: the first launch (or install.bat) fetches the
Python environments, the chat brain and the video renderer, about 14 GB in total.
The GitHub "Release" workflow runs this and attaches the zip to the release.
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIST = ROOT / "dist"

FILES = [
    "install.bat", "installer.py", "Start Avatar AI Studio App.bat", "Start Avatar AI Studio Web.bat", "Stop Avatar AI Studio.bat",
    "run.py", "start_all.py", "requirements.txt", "README.md",
    "brain/setup_brain.py",
    "ml/voice_clone.py",
    "ml/realtime/flashhead_render_server.py", "ml/realtime/setup_flashhead.py",
]
DIRS = ["app", "static", "tools", "ml/realtime/flashhead_stubs"]
SKIP_PARTS = {"__pycache__"}
SKIP_SUFFIX = {".pyc"}


def main() -> int:
    version = sys.argv[1] if len(sys.argv) > 1 else "dev"
    DIST.mkdir(exist_ok=True)
    out = DIST / f"Avatar-Studio-{version}.zip"
    paths: list[Path] = []
    for f in FILES:
        p = ROOT / f
        if p.exists():
            paths.append(p)
    for d in DIRS:
        paths += [p for p in sorted((ROOT / d).rglob("*")) if p.is_file() and not (SKIP_PARTS & set(p.parts)) and p.suffix not in SKIP_SUFFIX]
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            z.write(p, Path("Avatar AI Studio") / p.relative_to(ROOT))
    print(f"Built {out} ({len(paths)} files, {out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
