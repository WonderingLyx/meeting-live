@echo off
setlocal
set "ROOT=%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%scripts\repair-face-runtime.ps1" %*
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
  echo.
  echo Face runtime repair failed with exit code %EXIT_CODE%.
  echo.
  echo Press any key to close this window.
  pause >nul
)
exit /b %EXIT_CODE%
