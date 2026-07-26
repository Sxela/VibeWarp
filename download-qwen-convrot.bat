@echo off
rem Backward-compatible entry point matching download-qwen-convrot.sh.
call "%~dp0download-qwen-image-edit.bat" %*
exit /b %ERRORLEVEL%
