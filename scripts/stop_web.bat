@echo off
REM Stop the detached AIoT web server started by launch_dashboard.vbs
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_web.ps1" -Stop
echo.
echo Server stopped. Press any key to close.
pause >nul