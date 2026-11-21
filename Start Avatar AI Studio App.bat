@echo off
setlocal
title Avatar AI Studio
cd /d "%~dp0"

:: Opens in its own window (like a desktop app). Use "Start Avatar AI Studio Web.bat" for a browser tab.
set AVATAR_WINDOW=app
set AVATAR_LLM_MODE=relaxed

:: -- Python environment missing? Set it up first. (The chat brain, video renderer and models are then downloaded by the app itself, with progress on its start screen.)
set "NEED="
if not exist "ml\.venv\Scripts\python.exe" set NEED=1
if not defined NEED "ml\.venv\Scripts\python.exe" -c "import importlib.util as u,sys;sys.exit(0 if all(u.find_spec(m) for m in ('fastapi','torch','chatterbox','transformers')) else 1)" >nul 2>&1
if not defined NEED if errorlevel 1 set NEED=1
if defined NEED (
    echo  [INFO] Avatar AI Studio is not fully installed yet. Running the installer...
    echo.
    set AVATAR_AUTO_INSTALL=1
    set AVATAR_BASE_ONLY=1
    call "%~dp0install.bat"
    set AVATAR_AUTO_INSTALL=
    set AVATAR_BASE_ONLY=
    if errorlevel 1 (
        echo  [ERROR] Installation failed. Fix the errors above and try again.
        pause
        exit /b 1
    )
)

echo Closing any old Avatar AI Studio servers first...
powershell -NoProfile -Command "foreach($p in 8765,8770,8771,8772){Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }}"
timeout /t 2 /nobreak >nul
echo Starting Avatar AI Studio (chat brain and video renderer start in the background)...
"ml\.venv\Scripts\python.exe" start_all.py
if errorlevel 1 (
    echo.
    echo  Avatar AI Studio stopped with an error. See the messages above.
    pause
)
