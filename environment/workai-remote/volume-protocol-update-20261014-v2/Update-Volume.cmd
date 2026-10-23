@echo off
setlocal
powershell.exe -NoLogo -NoProfile -ExecutionPolicy RemoteSigned -File "%~dp0Update-Volume.ps1" -PromptForRoot %*
set "WORKAI_UPDATE_EXIT=%ERRORLEVEL%"
pause
exit /b %WORKAI_UPDATE_EXIT%
