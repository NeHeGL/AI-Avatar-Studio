@echo off
REM Voice cloning stack (Chatterbox, MIT) in an isolated Python 3.11 env.
cd /d "%~dp0"
uv pip install --python .venv\Scripts\python.exe torch torchaudio --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv\Scripts\python.exe chatterbox-tts soundfile imageio-ffmpeg
