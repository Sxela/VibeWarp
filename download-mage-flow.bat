@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem Windows counterpart to download-mage-flow.sh. Files are downloaded into
rem the current working directory. Existing files are skipped and *.part files
rem resume on the next run.
rem
rem Optional environment variables:
rem   HF_TOKEN                    Hugging Face token
rem   HF_REPO                     Source repository
rem   HF_REVISION                 Source revision
rem   VIBEWARP_DOWNLOAD_DRY_RUN=1 Print targets without downloading

set "REPO=Comfy-Org/Mage-Flow"
if defined HF_REPO set "REPO=%HF_REPO%"
set "REVISION=main"
if defined HF_REVISION set "REVISION=%HF_REVISION%"
set "DESTINATION=%CD%"

where curl.exe >nul 2>nul
if errorlevel 1 (
    echo Error: curl.exe is required. Current Windows 10/11 installations include it.
    exit /b 1
)

echo Downloading 4 Mage-Flow files to %DESTINATION%
for %%P in (
    "diffusion_models/mage_flow_edit_int8_convrot.safetensors"
    "diffusion_models/mage_flow_edit_turbo_int8_convrot.safetensors"
    "text_encoders/qwen3vl_4b_bf16.safetensors"
    "vae/mage_flow_vae_bf16.safetensors"
) do (
    set "REMOTE_PATH=%%~P"
    set "TARGET=%DESTINATION%\%%~nxP"
    set "PART=!TARGET!.part"
    set "URL=https://huggingface.co/%REPO%/resolve/%REVISION%/!REMOTE_PATH!?download=true"

    if exist "!TARGET!" (
        echo Skipping existing %%~nxP
    ) else if defined VIBEWARP_DOWNLOAD_DRY_RUN (
        echo Would download %%~nxP
        echo   !URL!
    ) else (
        if exist "!PART!" (
            for %%S in ("!PART!") do echo Resuming %%~nxP from %%~zS bytes
        ) else (
            echo Downloading %%~nxP
        )

        if defined HF_TOKEN (
            curl.exe --fail --location --show-error --retry 5 --retry-delay 3 --retry-all-errors --continue-at - -H "Authorization: Bearer %HF_TOKEN%" --output "!PART!" "!URL!"
        ) else (
            curl.exe --fail --location --show-error --retry 5 --retry-delay 3 --retry-all-errors --continue-at - --output "!PART!" "!URL!"
        )
        if errorlevel 1 (
            echo Error downloading %%~nxP. The partial file was kept for resume.
            exit /b 1
        )

        move /Y "!PART!" "!TARGET!" >nul
        if errorlevel 1 (
            echo Error finalizing %%~nxP.
            exit /b 1
        )
        echo Saved !TARGET!
    )
)

echo Mage-Flow model downloads complete.
exit /b 0
