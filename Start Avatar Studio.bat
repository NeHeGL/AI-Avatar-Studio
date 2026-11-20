@echo off
cd /d "%~dp0"
rem Chat brain (Ollama). Use any local model name from: ollama list
rem gemma3:12b leaves room on the GPU for the video renderer; gemma3:27b is smarter but fills the card.
set AVATAR_LLM=gemma3:12b
set AVATAR_LLM_MODE=relaxed
rem set AVATAR_LLM=your-uncensored-model:tag
rem Replies are rendered as a real video of the person by the local FlashHead-Lite service (started below).
echo Closing any old Avatar Studio servers first...
powershell -NoProfile -Command "foreach($p in 8765,8770,8771){Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }}"
timeout /t 2 /nobreak >nul
echo Starting Avatar Studio (the live face and the video renderer start in the background)...
"ml\.venv\Scripts\python.exe" start_all.py
if errorlevel 1 (
  echo.
  echo If you see "python is not recognized", install Python or use: py run.py
  pause
)
