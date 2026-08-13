@echo off
setlocal
set "ROOT=%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%scripts\install-windows.ps1" -Profile NvidiaCuda %*
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
  echo.
  echo NVIDIA GPU install failed with exit code %EXIT_CODE%.
  echo See logs\install-windows-nvidia-*.log for details.
)
exit /b %EXIT_CODE%
