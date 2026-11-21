# Avatar AI Studio

Make a talking avatar of a person from a few photos and a voice sample, then chat with it by voice. Everything runs on your own PC.

## Install and run

1. Install Python 3.11 (tick "Add Python to PATH").
2. Double-click **Start Avatar AI Studio App.bat** (own window) or **Start Avatar AI Studio Web.bat** (browser tab).
   If anything is missing, the first launch runs **install.bat** for you. It sets up the Python environments, the chat brain and the video renderer (about 14 GB, one time).

**Stop Avatar AI Studio.bat** closes everything. **Build Standalone.bat** makes the release zip in `dist/`.

## Needs

Windows 10/11, an NVIDIA graphics card (RTX 20-series or newer, 12 GB or more of video memory recommended), about 25 GB of free disk space.

## Where things live

| Folder | What |
|---|---|
| `app/` | The Python server: pages, profiles, chat, voice |
| `static/` | The web pages (HTML, CSS, JavaScript) |
| `brain/` | The chat model and its server (downloaded by the installer) |
| `ml/` | Speech and voice code, the video renderer, and the Python environment |
| `tools/` | Face-landmark helpers |
| `data/profiles/<name>/` | One folder per person: `knowledge/` (who they are, their people and life), `media/`, `voice/` |

## Use your own chat model

The chat brain is a standard Gemma 3 model chosen for your graphics card. To use a different one (for example an uncensored model), put its `.gguf` file in `brain/models/` and name it `custom-<anything>.gguf`. A file with that prefix is used instead of the built-in choice. Delete it to go back.

