@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem Windows counterpart to download-qwen-image-edit.sh. Downloads every Qwen
rem ConvRot checkpoint plus the encoder, VAE, Lightning LoRA, and Q5_K_M GGUF
rem into the current working directory. Existing files are skipped and *.part
rem files resume on the next run.
rem
rem Optional environment variables:
rem   HF_TOKEN                    Hugging Face token
rem   HF_REPO                     ConvRot source repository
rem   HF_REVISION                 ConvRot source revision
rem   VIBEWARP_DOWNLOAD_DRY_RUN=1 Print targets without downloading

set "REPO=Comfy-Org/Qwen-Image-Edit_ComfyUI"
if defined HF_REPO set "REPO=%HF_REPO%"
set "REVISION=main"
if defined HF_REVISION set "REVISION=%HF_REVISION%"
set "DESTINATION=%CD%"
set "API_URL=https://huggingface.co/api/models/%REPO%/revision/%REVISION%"
set "METADATA_FILE=%TEMP%\vibewarp-qwen-%RANDOM%-%RANDOM%.json"
set "FILE_LIST=%TEMP%\vibewarp-qwen-%RANDOM%-%RANDOM%.txt"

where curl.exe >nul 2>nul
if errorlevel 1 (
    echo Error: curl.exe is required. Current Windows 10/11 installations include it.
    exit /b 1
)
where powershell.exe >nul 2>nul
if errorlevel 1 (
    echo Error: Windows PowerShell is required to read Hugging Face metadata.
    exit /b 1
)

echo Reading file list from https://huggingface.co/%REPO% ...
if defined HF_TOKEN (
    curl.exe --fail --silent --show-error --location -H "Authorization: Bearer %HF_TOKEN%" --output "%METADATA_FILE%" "%API_URL%"
) else (
    curl.exe --fail --silent --show-error --location --output "%METADATA_FILE%" "%API_URL%"
)
if errorlevel 1 (
    if exist "%METADATA_FILE%" del /Q "%METADATA_FILE%"
    echo Error reading Hugging Face metadata.
    exit /b 1
)

set "QWEN_METADATA_FILE=%METADATA_FILE%"
set "QWEN_FILE_LIST=%FILE_LIST%"
powershell.exe -NoProfile -Command "$metadata = Get-Content -Raw -LiteralPath $env:QWEN_METADATA_FILE | ConvertFrom-Json; $files = @($metadata.siblings.rfilename | Where-Object { $_ -match '(^|/)qwen[^/]*convrot[^/]*[.]safetensors$' }); $files | Set-Content -Encoding ascii -LiteralPath $env:QWEN_FILE_LIST; if ($files.Count -eq 0) { exit 2 }"
if errorlevel 1 (
    del /Q "%METADATA_FILE%" >nul 2>nul
    if exist "%FILE_LIST%" del /Q "%FILE_LIST%"
    echo Error: no Qwen ConvRot safetensors files were found in %REPO%@%REVISION%.
    exit /b 1
)

set /A CONVROT_COUNT=0
for /F "usebackq delims=" %%P in ("%FILE_LIST%") do set /A CONVROT_COUNT+=1
echo Found !CONVROT_COUNT! ConvRot file(s); downloading complete Qwen set to %DESTINATION%

for /F "usebackq delims=" %%P in ("%FILE_LIST%") do (
    set "REMOTE_PATH=%%P"
    for %%N in ("%%P") do set "FILENAME=%%~nxN"
    set "TARGET=%DESTINATION%\!FILENAME!"
    set "PART=!TARGET!.part"
    set "URL=https://huggingface.co/%REPO%/resolve/%REVISION%/!REMOTE_PATH!?download=true"

    if exist "!TARGET!" (
        echo Skipping existing !FILENAME!
    ) else if defined VIBEWARP_DOWNLOAD_DRY_RUN (
        echo Would download !FILENAME!
        echo   !URL!
    ) else (
        if exist "!PART!" (
            for %%S in ("!PART!") do echo Resuming !FILENAME! from %%~zS bytes
        ) else (
            echo Downloading !FILENAME!
        )

        if defined HF_TOKEN (
            curl.exe --fail --location --show-error --retry 5 --retry-delay 3 --retry-all-errors --continue-at - -H "Authorization: Bearer %HF_TOKEN%" --output "!PART!" "!URL!"
        ) else (
            curl.exe --fail --location --show-error --retry 5 --retry-delay 3 --retry-all-errors --continue-at - --output "!PART!" "!URL!"
        )
        if errorlevel 1 (
            del /Q "%METADATA_FILE%" "%FILE_LIST%" >nul 2>nul
            echo Error downloading !FILENAME!. The partial file was kept for resume.
            exit /b 1
        )

        move /Y "!PART!" "!TARGET!" >nul
        if errorlevel 1 (
            del /Q "%METADATA_FILE%" "%FILE_LIST%" >nul 2>nul
            echo Error finalizing !FILENAME!.
            exit /b 1
        )
        echo Saved !TARGET!
    )
)

rem Each item is "repository|revision|remote path".
for %%P in (
    "Comfy-Org/HunyuanVideo_1.5_repackaged|main|split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"
    "Comfy-Org/Qwen-Image_ComfyUI|main|split_files/vae/qwen_image_vae.safetensors"
    "lightx2v/Qwen-Image-Edit-2511-Lightning|main|Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors"
    "vantagewithai/Qwen-Image-Edit-2511-GGUF|main|Qwen-Image-Edit-2511-Q5_K_M.gguf"
) do (
    for /F "tokens=1,2,* delims=|" %%A in ("%%~P") do (
        set "SOURCE_REPO=%%A"
        set "SOURCE_REVISION=%%B"
        set "REMOTE_PATH=%%C"
        for %%N in ("%%C") do set "FILENAME=%%~nxN"
        set "TARGET=%DESTINATION%\!FILENAME!"
        set "PART=!TARGET!.part"
        set "URL=https://huggingface.co/!SOURCE_REPO!/resolve/!SOURCE_REVISION!/!REMOTE_PATH!?download=true"

        if exist "!TARGET!" (
            echo Skipping existing !FILENAME!
        ) else if defined VIBEWARP_DOWNLOAD_DRY_RUN (
            echo Would download !FILENAME!
            echo   !URL!
        ) else (
            if exist "!PART!" (
                for %%S in ("!PART!") do echo Resuming !FILENAME! from %%~zS bytes
            ) else (
                echo Downloading !FILENAME!
            )

            if defined HF_TOKEN (
                curl.exe --fail --location --show-error --retry 5 --retry-delay 3 --retry-all-errors --continue-at - -H "Authorization: Bearer %HF_TOKEN%" --output "!PART!" "!URL!"
            ) else (
                curl.exe --fail --location --show-error --retry 5 --retry-delay 3 --retry-all-errors --continue-at - --output "!PART!" "!URL!"
            )
            if errorlevel 1 (
                del /Q "%METADATA_FILE%" "%FILE_LIST%" >nul 2>nul
                echo Error downloading !FILENAME!. The partial file was kept for resume.
                exit /b 1
            )

            move /Y "!PART!" "!TARGET!" >nul
            if errorlevel 1 (
                del /Q "%METADATA_FILE%" "%FILE_LIST%" >nul 2>nul
                echo Error finalizing !FILENAME!.
                exit /b 1
            )
            echo Saved !TARGET!
        )
    )
)

del /Q "%METADATA_FILE%" "%FILE_LIST%" >nul 2>nul
echo Qwen model downloads complete.
exit /b 0
