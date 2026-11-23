@echo off
cd /d "%~dp0"
:: Builds dist\AI Avatar Studio.exe and dist\README.md (needs uv: https://docs.astral.sh/uv/)
uv run --no-project --python 3.11 build_standalone.py %1
pause
