@echo off
setlocal
set "ROOT=%~dp0"
echo %* | findstr /I /C:"-Profile" >nul
if "%ERRORLEVEL%"=="0" (
  echo install-windows.cmd is CPU-only. Use install-windows-amd-gpu.cmd or install-windows-nvidia-gpu.cmd for GPU installs.
  exit /b 2
)
powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%scripts\install-windows.ps1" -Profile Cpu %*
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
  echo.
  echo Install failed with exit code %EXIT_CODE%.
  echo See logs\install-windows-*.log for details.
  echo.
  echo Press any key to close this window.
  pause >nul
)
exit /b %EXIT_CODE%
