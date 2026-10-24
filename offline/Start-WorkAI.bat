@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONUTF8=1
set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=1
"%~dp0runtime\python.exe" -m offline.app %*
if errorlevel 1 pause
