@echo off
setlocal
echo WorkAI research source export
echo This reads selected model source files and creates a new ZIP on Desktop.
echo It does not run COMSOL, change pairing, or modify installed files.
echo.
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0Export-WorkAIResearch.ps1"
if errorlevel 1 (
  echo.
  echo EXPORT FAILED. Please send the error text, not any private key.
)
echo.
pause
