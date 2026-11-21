@echo off
rem Closes the app, the live face and the video renderer.
powershell -NoProfile -Command "foreach($p in 8765,8770,8771,8772){Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }}"
echo Avatar AI Studio stopped.
timeout /t 2 /nobreak >nul
