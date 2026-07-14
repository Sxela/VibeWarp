@echo off
REM VibeWarp launcher — run install.bat once first.
REM All arguments are forwarded to `python -m vibewarp`, e.g.:
REM   run.bat --warpfusion-settings settings.txt --video input.mp4

setlocal

REM ffmpeg.exe is copied next to this script by install.bat
set "PATH=%~dp0;%PATH%"

if not exist "%~dp0env\Scripts\activate.bat" (
    echo Environment not found. Run install.bat first.
    pause
    exit /b 1
)

call "%~dp0env\Scripts\activate"
python -m vibewarp %*
