@echo off
REM VibeWarp one-click installer (Windows).
REM Downloads a private embedded Python (see py_ver below), pip, ffmpeg, a venv
REM and installs VibeWarp + PyTorch. Safe to re-run: finished steps are
REM skipped, so a failed download just needs another run.
REM Everything lands next to this script; nothing touches system Python.

setlocal

REM ---- knobs -------------------------------------------------------------
REM PyTorch wheel index — change cu128 to match your CUDA driver if needed
REM (see https://pytorch.org/get-started/locally/).
set "torch_index=https://download.pytorch.org/whl/cu128"

REM Python version the package is validated against. py_tag is the embeddable
REM zip/._pth suffix (e.g. 3.14.4 -> 314). The pinned deps in pyproject
REM require Python >= 3.11, so this must not drop below that.
set "py_ver=3.14.4"
set "py_tag=314"
set "python_url=https://www.python.org/ftp/python/%py_ver%/python-%py_ver%-embed-amd64.zip"
set "pip_url=https://bootstrap.pypa.io/get-pip.py"
set "ffmpeg_url=https://github.com/GyanD/codexffmpeg/releases/download/6.0/ffmpeg-6.0-full_build.zip"
REM Update these hashes whenever one of the pinned downloads changes.
set "python_sha256=CDA80A9B1E75C0F1B4F9872CA1B417F0D19BCE32FACC811AEA9180E70FAD5FB9"
set "pip_sha256=A341E1A43E38001C551A1508A73FF23636A11970B61D901D9A1CAD2A18F57055"
set "ffmpeg_sha256=F5DF7A970919AA0DF2A4F6DF4EF0BBE85E6B7C41D343556E7210837A63B3133C"
set "virtualenv_ver=21.6.1"
REM -------------------------------------------------------------------------

set "python_zip=%~dp0python.zip"
set "python_dir=%~dp0python"
set "scripts_dir=%~dp0python\Scripts"
set "lib_dir=%~dp0python\Lib\site-packages"
set "pip_py=%~dp0get-pip.py"
set "venv_dir=%~dp0env"
set "ffmpeg_zip=%~dp0ffmpeg-6.0-full_build.zip"
set "ffmpeg_dir=%~dp0ffmpeg-6.0-full_build"

echo Checking ffmpeg download...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\download_verified.ps1" -Url "%ffmpeg_url%" -Destination "%ffmpeg_zip%" -Sha256 "%ffmpeg_sha256%"
if errorlevel 1 goto :failed

if not exist "%ffmpeg_dir%\ffmpeg-6.0-full_build\bin\ffmpeg.exe" (
    echo Extracting ffmpeg...
    powershell -NoProfile -Command "Expand-Archive -LiteralPath '%ffmpeg_zip%' -DestinationPath '%ffmpeg_dir%' -Force"
    if errorlevel 1 goto :failed
)
if not exist "%ffmpeg_dir%\ffmpeg-6.0-full_build\bin\ffprobe.exe" (
    echo Repairing incomplete ffmpeg extraction...
    powershell -NoProfile -Command "Expand-Archive -LiteralPath '%ffmpeg_zip%' -DestinationPath '%ffmpeg_dir%' -Force"
    if errorlevel 1 goto :failed
)
copy /Y "%ffmpeg_dir%\ffmpeg-6.0-full_build\bin\ffmpeg.exe" "%~dp0" >nul
if errorlevel 1 goto :failed
copy /Y "%ffmpeg_dir%\ffmpeg-6.0-full_build\bin\ffprobe.exe" "%~dp0" >nul
if errorlevel 1 goto :failed

echo Checking Python %py_ver% download...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\download_verified.ps1" -Url "%python_url%" -Destination "%python_zip%" -Sha256 "%python_sha256%"
if errorlevel 1 goto :failed

if not exist "%python_dir%\python.exe" (
    echo Extracting Python %py_ver%...
    powershell -NoProfile -Command "Expand-Archive -LiteralPath '%python_zip%' -DestinationPath '%python_dir%' -Force"
    if errorlevel 1 goto :failed
)

REM Allow the embedded interpreter to see pip-installed packages
(
echo python%py_tag%.zip
echo Lib\site-packages
echo .
) > "%python_dir%\python%py_tag%._pth"

set "PATH=%python_dir%;%scripts_dir%;%lib_dir%;%PATH%"

if not exist "%python_dir%\Lib\site-packages\pip" (
    echo Installing pip...
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\download_verified.ps1" -Url "%pip_url%" -Destination "%pip_py%" -Sha256 "%pip_sha256%"
    if errorlevel 1 goto :failed
    "%python_dir%\python" "%pip_py%"
    if errorlevel 1 goto :failed
)

if not exist "%python_dir%\Lib\site-packages\virtualenv" (
    echo Installing virtualenv %virtualenv_ver%...
    call "%python_dir%\python" -m pip install "virtualenv==%virtualenv_ver%"
    if errorlevel 1 goto :failed
)

if not exist "%venv_dir%" (
    echo Creating virtual environment...
    call "%python_dir%\python" -m virtualenv --python="%python_dir%\python.exe" "%venv_dir%"
    if errorlevel 1 goto :failed
)

echo Activating virtual environment...
call "%venv_dir%\Scripts\activate"
if errorlevel 1 goto :failed

echo Installing PyTorch from %torch_index% ...
call python -m pip install "torch==2.11.0" "torchvision==0.26.0" --index-url "%torch_index%"
if errorlevel 1 goto :failed

echo Installing VibeWarp...
call python -m pip install -e "%~dp0."
if errorlevel 1 goto :failed

echo.
echo -----------------------------------------------------------------
echo Install complete.
echo.
echo Next steps:
echo   1. Download models for your settings file:
echo        run.bat --download-models --warpfusion-settings path\to\settings.txt
echo   2. Render:
echo        run.bat --warpfusion-settings path\to\settings.txt --video input.mp4
echo -----------------------------------------------------------------
pause
exit /b 0

:failed
echo.
echo Installation failed. Review the error above; no unverified download was installed.
pause
exit /b 1
