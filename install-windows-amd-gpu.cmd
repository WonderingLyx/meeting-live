@echo off
setlocal
set "ROOT=%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%scripts\install-windows.ps1" -Profile AmdRocm %*
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
  echo.
  echo AMD GPU install failed with exit code %EXIT_CODE%.
  echo See logs\install-windows-rocm-*.log for details.
  echo.
  echo Press any key to close this window.
  pause >nul
)
exit /b %EXIT_CODE%
