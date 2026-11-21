"""Start AI Avatar Studio as ONE thing.

The studio is three programs, each in its own Python environment (they use different libraries, so they cannot share one process):
  - the app           http://localhost:8765  (this process: pages, chat, voice)
  - the live face     http://localhost:8770  (real-time face + the saved waiting animations)
  - the video renderer http://localhost:8771  (talking video of each spoken sentence)

This launcher runs the app in its own window, starts the other two hidden (their output goes to data/tmp_render/*.log), restarts
either one if it crashes, and closes everything together when this window is closed (a Windows "job object" ties them to it).
"""
from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import threading
import time
from ctypes import wintypes
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LOGS = ROOT / "data" / "tmp_render"
BRAIN = ROOT / "brain"
def _brain_model() -> Path:
    """The model that fits this card, if it is there; otherwise any model the person put in brain/models."""
    from app import hardware

    # a model the person chose themselves wins: any file named custom-*.gguf in brain/models (their own pick, e.g. an uncensored one)
    mine = next(iter(sorted((BRAIN / "models").glob("custom-*.gguf"))), None)
    if mine is not None and not os.environ.get("AVATAR_VRAM_GB"):
        return mine
    want = BRAIN / "models" / hardware.tier()["brain_file"]
    if want.exists() or os.environ.get("AVATAR_VRAM_GB"):  # a simulated smaller card must not fall back to a model that will not fit it
        return want
    return next(iter(sorted((BRAIN / "models").glob("*.gguf"))), want)


BRAIN_PORT = int(os.environ.get("AVATAR_BRAIN_PORT", "8772"))
# (name, program, arguments, log file, file that must exist)
CHILDREN = [
    ("chat brain", BRAIN / "bin" / "llama-server.exe",
     lambda: ["-m", _brain_model(), "--alias", "brain", "--host", "127.0.0.1", "--port", BRAIN_PORT, "-c", os.environ.get("AVATAR_LLM_CTX", "4096"), "-ngl", "99", "-np", "1", "--jinja",
                       # no "thinking" out loud: some models (Gemma 4) write their reasoning first, and the avatar would speak it
                       "--reasoning", "off", "--reasoning-format", "deepseek", "--chat-template-kwargs", '{"enable_thinking": false}'],
     "brain.out.log", lambda: _brain_model()),
    ("video renderer", Path(sys.executable), [ROOT / "ml" / "realtime" / "flashhead_render_server.py"], "flashhead.out.log",
     ROOT / "ml" / "realtime" / "flashhead_render_server.py"),
]

# ---- a Windows job object: when this process ends for any reason, everything in the job is closed too ----


class _IO(ctypes.Structure):
    _fields_ = [(n, ctypes.c_ulonglong) for n in ("r", "w", "o", "rb", "wb", "ob")]


class _BASIC(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64), ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t), ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD),
    ]


class _EXT(ctypes.Structure):
    _fields_ = [("Basic", _BASIC), ("Io", _IO), ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]


def _make_job():
    if os.name != "nt":
        return None
    k32 = ctypes.windll.kernel32
    job = k32.CreateJobObjectW(None, None)
    info = _EXT()
    info.Basic.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info))
    return job


JOB = _make_job()
_procs: dict[str, subprocess.Popen] = {}
_stop = threading.Event()


def _start(name: str, exe: Path, args, log: str, needs, quiet: bool = False) -> subprocess.Popen | None:
    args = args() if callable(args) else args
    needs = needs() if callable(needs) else needs
    if not exe.exists() or not needs.exists():
        if quiet:
            return None
        print(f"[{name}] not found ({exe if not exe.exists() else needs}); skipping" + ("  -> run install.bat" if name == "chat brain" else ""))
        return None
    LOGS.mkdir(parents=True, exist_ok=True)
    out = open(LOGS / log, "ab", buffering=0)
    out.write(f"\n=== {time.strftime('%H:%M:%S')} starting {name} ===\n".encode())
    flags = 0x08000000  # CREATE_NO_WINDOW: no console window of its own
    env = dict(os.environ)
    from app import hardware

    if hardware.tier()["name"] == "small":
        env["AVATAR_LEAN"] = "1"  # only the small (16 GB) tier runs the lean settings
    p = subprocess.Popen([str(exe), *map(str, args)], cwd=str(ROOT), env=env, stdout=out, stderr=subprocess.STDOUT, creationflags=flags)
    if JOB:
        ctypes.windll.kernel32.AssignProcessToJobObject(JOB, wintypes.HANDLE(int(p._handle)))
    print(f"[{name}] started (pid {p.pid}), log: data/tmp_render/{log}")
    return p


def _watch() -> None:
    """Restart a child that has died (at most 6 times in 10 minutes, so a broken one does not spin)."""
    history: dict[str, list[float]] = {c[0]: [] for c in CHILDREN}
    while not _stop.wait(5.0):
        for name, exe, script, log, needs in CHILDREN:
            p = _procs.get(name)
            if p is None:  # was not installed at launch: start it as soon as the installer has put it in place
                _procs[name] = _start(name, exe, script, log, needs, quiet=True)
                continue
            if p.poll() is not None:
                now = time.time()
                history[name] = [t for t in history[name] if now - t < 600]
                if len(history[name]) >= 6:
                    continue
                history[name].append(now)
                print(f"[{name}] stopped (exit {p.returncode}); restarting")
                _procs[name] = _start(name, exe, script, log, needs)


def _shutdown() -> None:
    _stop.set()
    for name, p in _procs.items():
        if p is not None and p.poll() is None:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True)


def _clear_temp_renders() -> None:
    """Start clean: drop last session's temp renders/logs and old reply videos so no stale video can be shown."""
    import shutil

    if LOGS.exists():
        for f in LOGS.iterdir():
            try:
                shutil.rmtree(f) if f.is_dir() else f.unlink()
            except OSError:
                pass
    for f in (ROOT / "data" / "profiles").glob("*/media/talk/reply_*.mp4"):
        f.unlink(missing_ok=True)


if __name__ == "__main__":
    _clear_temp_renders()
    print("AI Avatar Studio: starting the live face and the video renderer in the background...")
    for name, exe, script, log, needs in CHILDREN:
        _procs[name] = _start(name, exe, script, log, needs)
    threading.Thread(target=_watch, daemon=True).start()
    import installer

    installer.start_background()  # anything not installed yet (chat brain, video renderer) is fetched now, in the background
    try:
        from app.main import run

        run()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"\nFailed to start AI Avatar Studio:\n  {exc}\n")
        input("Press Enter to close...")
        raise SystemExit(1) from exc
    finally:
        _shutdown()
