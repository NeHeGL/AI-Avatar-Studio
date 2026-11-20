from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import capture, chat, walkthrough
from app.storage import list_saved_names
from app.storage import DATA_ROOT, purge_unsaved_user_files, seed_demo_user

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "static"

_launch_browser = False
_launch_url = ""


def open_in_browser(url: str) -> bool:
    if sys.platform == "win32":
        try:
            os.startfile(url)  # type: ignore[attr-defined]
            return True
        except OSError:
            pass
        try:
            subprocess.run(
                ["cmd", "/c", "start", "", url],
                check=False,
                creationflags=subprocess.CREATE_NO_WINDOW
                if hasattr(subprocess, "CREATE_NO_WINDOW")
                else 0,
            )
            return True
        except OSError:
            pass
    return bool(webbrowser.open(url))


def wait_until_serving(url: str, timeout_sec: float = 45.0) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.5) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, TimeoutError, OSError):
            time.sleep(0.2)
    return False


app = FastAPI(title="Avatar Walkthrough")


@app.middleware("http")
async def _prefer_localhost(request: Request, call_next):
    """Browsers that resist fingerprinting (Firefox with 'resistFingerprinting') exempt `localhost` but not `127.0.0.1`,
    and there the face tracker fails with "index out of bounds". Send page loads from 127.0.0.1 to localhost.
    Only browser page navigations are redirected; scripts calling the API on 127.0.0.1 are untouched."""
    host = request.headers.get("host", "")
    if (
        host.startswith("127.0.0.1")
        and request.method == "GET"
        and "text/html" in request.headers.get("accept", "")
    ):
        return RedirectResponse(str(request.url).replace("127.0.0.1", "localhost", 1), status_code=307)
    return await call_next(request)


class NoCacheStatic(StaticFiles):
    """Always revalidate (ETag → fast 304), so edited JS/CSS/media are picked up by every
    browser, including embedded ones that ignore a plain reload."""

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


app.mount("/static", NoCacheStatic(directory=STATIC), name="static")
app.mount("/data", NoCacheStatic(directory=DATA_ROOT), name="data")


class NameBody(BaseModel):
    name: str


class ConfirmBody(BaseModel):
    name: str
    is_you: bool
    unique_full_name: str | None = None


class UpdateBody(BaseModel):
    canonical_name: str
    sections: list[str]
    merge_existing: bool = True


class SkipBody(BaseModel):
    canonical_name: str


class CaptureBody(BaseModel):
    canonical_name: str
    section: str


class ScanFramesBody(BaseModel):
    canonical_name: str
    frames: list[dict]


class VoiceBody(BaseModel):
    canonical_name: str
    audio_base64: str
    label: str = "Voice sample"


class SkinAtlasBody(BaseModel):
    canonical_name: str
    atlas_base64: str
    layout: dict


@app.post("/api/capture/skin-atlas")
def api_skin_atlas(body: SkinAtlasBody) -> dict:
    return capture.save_skin_atlas(body.canonical_name, body.atlas_base64, body.layout)


class StitchBody(BaseModel):
    canonical_name: str
    tone: list[float] | None = None


@app.post("/api/capture/stitch-skin")
def api_stitch_skin(body: StitchBody) -> dict:
    """Blend the saved scan photos into one seamless skin texture."""
    try:
        from app import skin_stitch

        return skin_stitch.stitch(body.canonical_name, body.tone)
    except Exception as exc:  # surface to the page instead of a hang
        return {"status": "error", "message": f"{type(exc).__name__}: {exc}"}


class StitchStartBody(BaseModel):
    canonical_name: str
    layout: dict


class StitchAddBody(BaseModel):
    canonical_name: str
    image_base64: str
    landmarks: list[list[float]]


class StitchFinishBody(BaseModel):
    canonical_name: str
    tone: list[float] | None = None


def _scan_dir(name: str):
    from app.capture import _user_media_dir
    from app.storage import MEDIA_DIR

    return MEDIA_DIR / _user_media_dir(name) / "scan"


@app.post("/api/stitch/start")
def api_stitch_start(body: StitchStartBody) -> dict:
    """Begin a live skin-stitching session for this scan (layout = mesh points + triangles from the calibration frame)."""
    import json as _json

    from app import skin_live

    d = _scan_dir(body.canonical_name)
    d.mkdir(parents=True, exist_ok=True)
    (d / "skin_atlas.json").write_text(_json.dumps(body.layout), encoding="utf-8")  # also used by the 3D head builder
    return skin_live.start(body.canonical_name, body.layout)


@app.post("/api/stitch/add")
def api_stitch_add(body: StitchAddBody) -> dict:
    import base64 as _b64

    from app import skin_live

    raw = _b64.b64decode(body.image_base64.split(",", 1)[-1])
    return skin_live.add(body.canonical_name, raw, body.landmarks)


@app.get("/api/stitch/preview")
def api_stitch_preview(canonical_name: str, tone: str = "") -> Response:
    from app import skin_live

    try:
        tg = [float(x) for x in tone.split(",")] if tone else None
    except ValueError:
        tg = None
    got = skin_live.preview(canonical_name, tg)
    if got is None:
        return Response(status_code=404)
    import base64 as _b64

    jpeg, version, cov, tri = got
    return Response(content=jpeg, media_type="image/jpeg",
                    headers={"X-Version": str(version), "X-Coverage": f"{cov:.4f}", "X-TriCov": _b64.b64encode(tri).decode(),
                             "Cache-Control": "no-store"})


@app.post("/api/stitch/finish")
def api_stitch_finish(body: StitchFinishBody) -> dict:
    from app import skin_live

    out = skin_live.finish(body.canonical_name, _scan_dir(body.canonical_name), body.tone)
    if out.get("status") == "ok":
        out["preview_url"] = walkthrough.media_url(str(_scan_dir(body.canonical_name) / "skin_stitched_preview.jpg"))
        out["url"] = walkthrough.media_url(str(_scan_dir(body.canonical_name) / "skin_stitched.png"))
    return out


@app.get("/api/stitch/status")
def api_stitch_status(canonical_name: str) -> dict:
    from app import skin_live

    s = skin_live.SESSIONS.get(canonical_name)
    return {"hires": s.hires if s else "none", "photos": s.live.photos if s else 0}


class OtherBody(BaseModel):
    canonical_name: str
    display_name: str = ""
    notes: str = ""


@app.on_event("startup")
def startup() -> None:
    purge_unsaved_user_files()
    if os.environ.get("AVATAR_SEED_DEMO", "").strip() == "1":
        seed_demo_user()
    if _launch_browser and _launch_url:

        def _open() -> None:
            if not wait_until_serving(_launch_url):
                print(
                    f"\nServer did not start in time.\n"
                    f"Fix any errors above, then open:\n  {_launch_url}\n"
                )
                return
            if not open_in_browser(_launch_url):
                print(f"\nCould not open a browser automatically.\nOpen: {_launch_url}\n")

        threading.Thread(target=_open, daemon=True).start()


@app.get("/api/build")
def api_build() -> dict:
    """Newest modification time of the front-end files, so an open page can tell when it is out of date."""
    newest = 0.0
    for f in STATIC.rglob("*"):
        if f.suffix in (".js", ".css", ".html"):
            newest = max(newest, f.stat().st_mtime)
    return {"build": newest}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/users")
def api_users() -> dict:
    return {"users": list_saved_names()}


@app.post("/api/walkthrough/check-name")
def api_check_name(body: NameBody) -> dict:
    return walkthrough.check_name(body.name)


@app.post("/api/walkthrough/confirm")
def api_confirm(body: ConfirmBody) -> dict:
    return walkthrough.confirm_existing(
        body.name, body.is_you, body.unique_full_name
    )


@app.post("/api/walkthrough/apply-updates")
def api_apply_updates(body: UpdateBody) -> dict:
    return walkthrough.apply_updates(
        body.canonical_name, body.sections, body.merge_existing
    )


@app.post("/api/walkthrough/skip-updates")
def api_skip_updates(body: SkipBody) -> dict:
    return walkthrough.skip_updates(body.canonical_name)


@app.post("/api/walkthrough/save-profile")
def api_save_profile(body: SkipBody) -> dict:
    return walkthrough.save_onboarding(body.canonical_name)


@app.post("/api/walkthrough/delete-profile")
def api_delete_profile(body: SkipBody) -> dict:
    return walkthrough.delete_profile(body.canonical_name)


@app.post("/api/walkthrough/capture-stub")
def api_capture_stub(body: CaptureBody) -> dict:
    return walkthrough.run_capture_stub(body.canonical_name, body.section)


@app.post("/api/capture/scan-frames")
def api_scan_frames(body: ScanFramesBody) -> dict:
    return capture.ingest_scan_frames(body.canonical_name, body.frames)


@app.post("/api/capture/skip-scan")
def api_skip_scan(body: SkipBody) -> dict:
    return capture.skip_scan(body.canonical_name)


@app.post("/api/capture/voice")
def api_voice(body: VoiceBody) -> dict:
    return capture.ingest_voice(body.canonical_name, body.audio_base64, body.label)


@app.post("/api/capture/other")
def api_other(body: OtherBody) -> dict:
    return capture.ingest_other(body.canonical_name, body.display_name, body.notes)


@app.post("/api/capture/body-frames")
def api_body_frames(body: ScanFramesBody) -> dict:
    return capture.ingest_body_frames(body.canonical_name, body.frames)


class ReplyDoneBody(BaseModel):
    video_url: str


@app.post("/api/chat/reply-played")
def api_reply_played(body: ReplyDoneBody) -> dict:
    """A reply video has been played: delete it (only generated reply_*.mp4 files inside a talk folder)."""
    from pathlib import Path

    data_root = (Path(__file__).resolve().parent.parent / "data").resolve()
    rel = body.video_url.split("?", 1)[0]
    if not rel.startswith("/data/"):
        return {"ok": False}
    path = (data_root / rel[len("/data/"):]).resolve()
    if data_root in path.parents and path.parent.name == "talk" and path.name.startswith("reply_") and path.suffix == ".mp4":
        path.unlink(missing_ok=True)
        return {"ok": True}
    return {"ok": False}


class ChatStartBody(BaseModel):
    canonical_name: str


class ChatTurnBody(BaseModel):
    canonical_name: str
    audio_base64: str | None = None
    text: str | None = None
    say: str | None = None
    token: str | None = None
    nudge: int | None = None  # 1..4: the model speaks up because the room went quiet


class PrepareBody(BaseModel):
    canonical_name: str


@app.post("/api/photoreal/prepare")
def api_photoreal_prepare(body: PrepareBody) -> dict:
    """Make sure this person has a prepared photo for the photo-real layers; says what else exists (clip library)."""
    from app import photoreal

    return photoreal.status(body.canonical_name)


@app.get("/api/chat/brain")
def api_chat_brain() -> dict:
    """Which Ollama model and tone the running server loaded (no GPU warmup)."""
    return {"llm_model": chat.LLM_MODEL, "llm_mode": chat.LLM_MODE}


@app.post("/api/chat/start")
def api_chat_start(body: ChatStartBody) -> dict:
    """Load the speech models (first call takes a while) and prepare this profile's voice + persona."""
    return chat.start(body.canonical_name)


class ChatPartialBody(BaseModel):
    audio_base64: str


@app.post("/api/chat/partial")
def api_chat_partial(body: ChatPartialBody) -> dict:
    """Live words while the person is still speaking (local Whisper on the audio so far)."""
    try:
        return {"text": chat.partial_text(body.audio_base64)}
    except Exception:
        return {"text": ""}


@app.post("/api/chat/turn")
def api_chat_turn(body: ChatTurnBody) -> StreamingResponse:
    """NDJSON stream: heard -> say (one per sentence, with WAV audio) -> done."""
    return StreamingResponse(
        chat.turn_stream(body.canonical_name, body.audio_base64, body.text, body.say, body.token, body.nudge),
        media_type="application/x-ndjson",
    )


def run(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    global _launch_browser, _launch_url
    import uvicorn

    _launch_url = f"http://{'localhost' if host == '127.0.0.1' else host}:{port}/"
    _launch_browser = open_browser
    print(f"\nAvatar Studio — {_launch_url}")
    print("Keep this window open while you use the app (Ctrl+C to quit).")
    if open_browser:
        print("Your browser will open after the server is ready.\n")
    else:
        print("Open that URL in your browser.\n")
    uvicorn.run("app.main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    run()