"""Start Avatar Studio as ONE thing.

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
CHILDREN = [
    ("live face", ROOT / "ml" / "realtime" / "lpenv" / "Scripts" / "python.exe", ROOT / "ml" / "realtime" / "idle_server.py", "idle.out.log"),
    ("video renderer", ROOT / "ml" / "realtime" / "dittoenv" / "Scripts" / "python.exe", ROOT / "ml" / "realtime" / "ditto_render_server.py", "ditto.out.log"),
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


def _start(name: str, exe: Path, script: Path, log: str) -> subprocess.Popen | None:
    if not exe.exists() or not script.exists():
        print(f"[{name}] not found ({exe if not exe.exists() else script}); skipping")
        return None
    LOGS.mkdir(parents=True, exist_ok=True)
    out = open(LOGS / log, "ab", buffering=0)
    out.write(f"\n=== {time.strftime('%H:%M:%S')} starting {name} ===\n".encode())
    flags = 0x08000000  # CREATE_NO_WINDOW: no console window of its own
    p = subprocess.Popen([str(exe), str(script)], cwd=str(ROOT), stdout=out, stderr=subprocess.STDOUT, creationflags=flags)
    if JOB:
        ctypes.windll.kernel32.AssignProcessToJobObject(JOB, wintypes.HANDLE(int(p._handle)))
    print(f"[{name}] started (pid {p.pid}), log: data/tmp_render/{log}")
    return p


def _watch() -> None:
    """Restart a child that has died (at most 6 times in 10 minutes, so a broken one does not spin)."""
    history: dict[str, list[float]] = {c[0]: [] for c in CHILDREN}
    while not _stop.wait(5.0):
        for name, exe, script, log in CHILDREN:
            p = _procs.get(name)
            if p is not None and p.poll() is not None:
                now = time.time()
                history[name] = [t for t in history[name] if now - t < 600]
                if len(history[name]) >= 6:
                    continue
                history[name].append(now)
                print(f"[{name}] stopped (exit {p.returncode}); restarting")
                _procs[name] = _start(name, exe, script, log)


def _shutdown() -> None:
    _stop.set()
    for name, p in _procs.items():
        if p is not None and p.poll() is None:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True)


if __name__ == "__main__":
    print("Avatar Studio: starting the live face and the video renderer in the background...")
    for name, exe, script, log in CHILDREN:
        _procs[name] = _start(name, exe, script, log)
    threading.Thread(target=_watch, daemon=True).start()
    try:
        from app.main import run

        run()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"\nFailed to start Avatar Studio:\n  {exc}\n")
        input("Press Enter to close...")
        raise SystemExit(1) from exc
    finally:
        _shutdown()
