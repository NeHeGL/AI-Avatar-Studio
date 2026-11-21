@echo off
setlocal
title Avatar AI Studio - Installer
cd /d "%~dp0"

echo.
echo  ============================================================
echo   Avatar AI Studio - Installer
echo  ============================================================
echo.

:: -- Python 3.11 (3.10 - 3.12 work) ----------------------------
set "PYBASE="
py -3.11 --version >nul 2>&1 && set "PYBASE=py -3.11"
if not defined PYBASE (
    python --version >nul 2>&1 && set "PYBASE=python"
)
if not defined PYBASE (
    echo  [ERROR] Python not found. Install Python 3.11 from https://www.python.org/downloads/
    echo          and tick "Add Python to PATH" during setup.
    pause
    exit /b 1
)
echo  [OK] Using: %PYBASE%

:: -- App environment ------------------------------------------
if not exist "ml\.venv\Scripts\python.exe" (
    echo  [1/4] Creating the Python environment (ml\.venv)...
    %PYBASE% -m venv ml\.venv
    if errorlevel 1 ( echo  [ERROR] Could not create ml\.venv & pause & exit /b 1 )
) else (
    echo  [1/4] Python environment already exists.
)
set "PY=ml\.venv\Scripts\python.exe"
"%PY%" -m pip install --upgrade pip --quiet

:: -- PyTorch (CUDA build; works on RTX 20-series through 50-series) ----
"%PY%" -c "import torch" >nul 2>&1
if errorlevel 1 (
    echo  [2/4] Installing PyTorch (about 3 GB, one time)...
    "%PY%" -m pip install torch==2.11.0 torchaudio==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
    if errorlevel 1 ( echo  [ERROR] PyTorch install failed. Check your internet connection. & pause & exit /b 1 )
) else (
    echo  [2/4] PyTorch already installed.
)

:: -- Python packages ------------------------------------------
echo  [3/4] Installing Python packages...
"%PY%" -m pip install -r requirements.txt
if errorlevel 1 ( echo  [ERROR] Package install failed. See above. & pause & exit /b 1 )
"%PY%" -m pip install --no-deps chatterbox-tts==0.1.7
if errorlevel 1 ( echo  [ERROR] Voice package install failed. See above. & pause & exit /b 1 )
"%PY%" -m pip install --no-deps faster-whisper
if errorlevel 1 ( echo  [ERROR] Speech package install failed. See above. & pause & exit /b 1 )

:: -- Chat brain, video renderer, face detector ----------------
:: (When started from a Start bat, the app downloads these itself, with progress on its start screen.)
if defined AVATAR_BASE_ONLY goto :done
echo  [4/4] Chat brain, video renderer and face detector (about 14 GB, one time)...
"%PY%" installer.py
if errorlevel 1 ( echo  [ERROR] Some downloads failed. Run install.bat again to resume. & pause & exit /b 1 )

:done
echo.
echo  ============================================================
echo   Installation complete!
echo   Start with:  "Start Avatar AI Studio App.bat"  (own window)
echo           or:  "Start Avatar AI Studio Web.bat"  (browser tab)
echo  ============================================================
echo.
if not defined AVATAR_AUTO_INSTALL pause
