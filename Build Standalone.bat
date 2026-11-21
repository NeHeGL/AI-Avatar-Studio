@echo off
cd /d "%~dp0"
if exist "ml\.venv\Scripts\python.exe" ("ml\.venv\Scripts\python.exe" build_standalone.py) else (python build_standalone.py)
pause
