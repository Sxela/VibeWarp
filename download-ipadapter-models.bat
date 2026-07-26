@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem Download all regular/Plus IP-Adapter checkpoints supported by VibeWarp and
rem both CLIP vision encoders. Completed files are skipped. Interrupted files
rem remain as *.part and resume on the next run.
rem
rem Optional environment variables:
rem   HF_TOKEN                    Hugging Face token
rem   VIBEWARP_MODELS_DIR         Model root (default: <script>\models)
rem   VIBEWARP_DOWNLOAD_DRY_RUN=1 Print targets without downloading

if defined VIBEWARP_MODELS_DIR (
    set "MODELS_ROOT=%VIBEWARP_MODELS_DIR%"
) else (
    set "MODELS_ROOT=%~dp0models"
)
set "HF_BASE=https://huggingface.co/h94/IP-Adapter/resolve/main"

where curl.exe >nul 2>nul
if errorlevel 1 (
    echo Error: curl.exe is required. Current Windows 10/11 installations include it.
    exit /b 1
)

if not exist "%MODELS_ROOT%\controlnet" mkdir "%MODELS_ROOT%\controlnet"
if errorlevel 1 exit /b 1
if not exist "%MODELS_ROOT%\controlnet\clip_vision" mkdir "%MODELS_ROOT%\controlnet\clip_vision"
if errorlevel 1 exit /b 1

echo.
echo IP-Adapter destination:
echo   %MODELS_ROOT%\controlnet
echo CLIP vision destination:
echo   %MODELS_ROOT%\controlnet\clip_vision
echo.

rem Each item is "remote path|path relative to MODELS_ROOT".
for %%P in (
    "models/ip-adapter_sd15.safetensors|controlnet\ip-adapter_sd15.safetensors"
    "models/ip-adapter_sd15_light.safetensors|controlnet\ip-adapter_sd15_light.safetensors"
    "models/ip-adapter-plus_sd15.safetensors|controlnet\ip-adapter-plus_sd15.safetensors"
    "models/ip-adapter-plus-face_sd15.safetensors|controlnet\ip-adapter-plus-face_sd15.safetensors"
    "models/ip-adapter-full-face_sd15.safetensors|controlnet\ip-adapter-full-face_sd15.safetensors"
    "models/ip-adapter_sd15_vit-G.safetensors|controlnet\ip-adapter_sd15_vit-G.safetensors"
    "sdxl_models/ip-adapter_sdxl.bin|controlnet\ip-adapter_sdxl.bin"
    "sdxl_models/ip-adapter_sdxl_vit-h.bin|controlnet\ip-adapter_sdxl_vit-h.bin"
    "sdxl_models/ip-adapter-plus_sdxl_vit-h.bin|controlnet\ip-adapter-plus_sdxl_vit-h.bin"
    "sdxl_models/ip-adapter-plus-face_sdxl_vit-h.bin|controlnet\ip-adapter-plus-face_sdxl_vit-h.bin"
    "models/image_encoder/model.safetensors|controlnet\clip_vision\clip_vision_vit_h.safetensors"
    "sdxl_models/image_encoder/model.safetensors|controlnet\clip_vision\clip_vision_vit_bigg.safetensors"
) do (
    for /f "tokens=1,2 delims=|" %%A in ("%%~P") do (
        set "URL=!HF_BASE!/%%A?download=true"
        set "TARGET=!MODELS_ROOT!\%%B"
        set "PART=!TARGET!.part"
        for %%N in ("!TARGET!") do set "NAME=%%~nxN"

        if exist "!TARGET!" (
            echo Skipping existing !NAME!
        ) else if defined VIBEWARP_DOWNLOAD_DRY_RUN (
            echo Would download !NAME!
            echo   !URL!
        ) else (
            if exist "!PART!" (
                for %%S in ("!PART!") do echo Resuming !NAME! from %%~zS bytes
            ) else (
                echo Downloading !NAME!
            )

            if defined HF_TOKEN (
                curl.exe --fail --location --show-error --retry 5 --retry-delay 3 --retry-all-errors --continue-at - -H "Authorization: Bearer %HF_TOKEN%" --output "!PART!" "!URL!"
            ) else (
                curl.exe --fail --location --show-error --retry 5 --retry-delay 3 --retry-all-errors --continue-at - --output "!PART!" "!URL!"
            )
            if errorlevel 1 (
                echo Error downloading !NAME!. The partial file was kept for resume.
                exit /b 1
            )

            move /Y "!PART!" "!TARGET!" >nul
            if errorlevel 1 (
                echo Error finalizing !NAME!.
                exit /b 1
            )
            echo Saved !TARGET!
        )
    )
)

echo.
echo IP-Adapter downloads complete.
echo Set the UI CLIP vision path to:
echo   %MODELS_ROOT%\controlnet\clip_vision
exit /b 0
