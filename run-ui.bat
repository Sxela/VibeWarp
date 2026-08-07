@echo off
REM VibeWarp Svelte web UI launcher — run install.bat once first.
REM Serves on http://localhost:7860 and opens a browser.

setlocal

REM ffmpeg.exe is copied next to this script by install.bat
set "PATH=%~dp0;%PATH%"

if not exist "%~dp0env\Scripts\activate.bat" (
    echo Environment not found. Run install.bat first.
    pause
    exit /b 1
)

call "%~dp0env\Scripts\activate"

REM Install/update dependencies when an existing environment predates a
REM required package. torchsde is needed by the DPM++ SDE samplers.
python -c "import fastapi, uvicorn, torchsde" 2>nul
if errorlevel 1 (
    echo Installing or updating VibeWarp UI dependencies...
    python -m pip install -e "%~dp0.[ui]"
)

python -m vibewarp.web %*
