@echo off
setlocal
set "ROOT=%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%start-windows.ps1" %*
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
  echo.
  echo Server exited with code %EXIT_CODE%.
)
exit /b %EXIT_CODE%
