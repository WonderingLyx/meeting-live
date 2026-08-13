@echo off
setlocal
set "ROOT=%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%scripts\install-windows.ps1" %*
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
  echo.
  echo Install failed with exit code %EXIT_CODE%.
  echo See logs\install-windows-*.log for details.
)
exit /b %EXIT_CODE%
