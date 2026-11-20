@echo off
cd /d "%~dp0"
rem Chat brain (Ollama). Use any local model name from: ollama list
set AVATAR_LLM=gemma3:27b
set AVATAR_LLM_MODE=relaxed
rem set AVATAR_LLM=your-uncensored-model:tag
echo Starting Avatar Studio...
"ml\.venv\Scripts\python.exe" run.py
if errorlevel 1 (
  echo.
  echo If you see "python is not recognized", install Python or use: py run.py
  pause
)