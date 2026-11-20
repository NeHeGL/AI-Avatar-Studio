@echo off
cd /d "%~dp0"
rem Chat brain (Ollama). Use any local model name from: ollama list
rem gemma3:12b leaves room on the GPU for the video renderer; gemma3:27b is smarter but fills the card.
set AVATAR_LLM=gemma3:12b
set AVATAR_LLM_MODE=relaxed
rem set AVATAR_LLM=your-uncensored-model:tag
rem Replies are rendered as a real video of the person by the local Ditto service (started below).
set AVATAR_REPLY_VIDEO=ditto
echo Starting the local video renderer...
start "Ditto render service" /min "ml\realtime\dittoenv\Scripts\python.exe" "ml\realtime\ditto_render_server.py"
echo Starting Avatar Studio...
"ml\.venv\Scripts\python.exe" run.py
if errorlevel 1 (
  echo.
  echo If you see "python is not recognized", install Python or use: py run.py
  pause
)
