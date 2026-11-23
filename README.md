# AI Avatar Studio

**Talk to someone you miss.** AI Avatar Studio builds a talking avatar of a person from a few photos, a voice recording and what you can tell it about them, then lets you have a spoken conversation with it. Everything runs on your own PC. Nothing is sent to anyone.

I made this because I want to bring people back from the dead, even just for a conversation. You tell it about the person, give it their face and their voice, and it tries to be them: it knows their family, their stories, how they talked, and it answers out loud, with their face, in their voice.

## What it does

- **Learns the person.** A Knowledge page holds everything you know about them (birthday, family, interests, how their name is pronounced, how they spoke). The avatar uses it in conversation and works out how the person comes across from what you wrote.
- **Faces.** Add a photo with your webcam or by upload. The program finds the face, levels it, centres and zooms it, crops it to the exact framing the video engine uses, and tells you plainly if the photo is blurry, too dark, turned away or has closed eyes. With the webcam it takes the shot itself when you look straight ahead with your eyes open and in focus, and keeps the best. Keep up to **5 photos** per person and swap between them with one click.
- **Voices.** Record a short passage or upload audio (up to **5 recordings** per person). The program picks the clearest speech from all of them and builds one clean reference clip to copy the voice from. It attempts to recreate their voice; how close it gets depends on the recording. A voice you like can be locked so it never changes.
- **Talks back.** You speak (or type), it listens, thinks, and answers with synthesized speech and a lip-synced video of the face.
- **Private.** The chat model, the voice model, the speech recognition and the video model all run locally. Profiles live in `data/profiles/<name>/` and are never part of the repository.

## Keeping it interactive

A conversation feels dead if every reply has a pause in front of it. A lot of the work here is hiding or removing delays, though it is not always possible:

- Every model is loaded at start-up (with a progress screen), so the first reply is not slow.
- Replies are spoken sentence by sentence while the model is still writing, with a short first sentence so the voice starts early.
- Speech recognition runs live while you talk, with strict turn-taking so the avatar does not hear itself.
- While it thinks, the face is never frozen: a bank of calm "waiting" animations plays, and every clip starts and ends on the same still frame, so clips join without a visible jump.
- Models are chosen to fit your graphics card, so nothing swaps out of memory mid-conversation.

## Hardware

| Graphics memory | Chat model | Speech recognition |
|---|---|---|
| **32 GB** (preferred) | Gemma 3 12B | Whisper large-v3 turbo on the GPU |
| 20 to 28 GB | Gemma 3 12B | Whisper small on the GPU |
| **16 GB** (works) | Gemma 3 4B (smaller) | Whisper distil-small on the CPU |

Windows 10/11, an NVIDIA card (RTX 20-series or newer), Python 3.11, and about 25 GB of free disk space. The first start downloads about 14 GB of models, once.

## Install and run

1. Install Python 3.11 (tick "Add Python to PATH").
2. Double-click **Start AI Avatar Studio App.bat** (own window) or **Start AI Avatar Studio Web.bat** (browser tab). If anything is missing, the first launch runs **install.bat** for you.

**Stop AI Avatar Studio.bat** closes everything. **Build Standalone.bat** makes the release zip in `dist/`.

### Use your own chat model

Put a chat-tuned `.gguf` file in `brain/models/` and name it `custom-<anything>.gguf`. It is used instead of the built-in choice. Delete it to go back.

## How it is built

| Part | What it uses |
|---|---|
| Server and pages | Python, FastAPI, plain HTML/CSS/JavaScript (no framework) |
| Chat | llama.cpp's server running a Gemma model |
| Voice | Qwen3-TTS 1.7B voice cloning |
| Hearing | Whisper (GPU) or faster-whisper (CPU) |
| Face video | SoulX-FlashHead-Lite, a lip-sync video model |
| Photo checks | MediaPipe face landmarks and OpenCV, in the browser and on the server |

| Folder | What |
|---|---|
| `app/` | The Python server: pages, profiles, photo and voice checks, chat |
| `static/` | The web pages |
| `brain/` | The chat model and its server (downloaded by the installer) |
| `ml/` | Voice and video code and the Python environment |
| `data/profiles/<name>/` | One folder per person: `photo/`, `voice/`, `knowledge/` (and `cache/`, made by the app and safe to delete) (git-ignored) |

## Limits

- The video model redraws the face for every frame, so it works best on real photos and badly on cartoons.
- Voice cloning from a short, noisy recording will not sound like the person. More clean speech helps a lot.
- It is a likeness built from what you give it, not the person. Use it with respect, and only with the voice and face of people you have the right to use.

## About

Built by Jeff Molofee (NeHe). I love AI and I love tech, and I'm looking for work.
