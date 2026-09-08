@echo off
setlocal EnableExtensions EnableDelayedExpansion
set "ROOT=%~dp0"
echo %* | findstr /I /C:"-Profile" >nul
if "%ERRORLEVEL%"=="0" (
  echo install-windows-nvidia-gpu.cmd always installs the NVIDIA CUDA profile. Do not pass -Profile.
  exit /b 2
)
powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%scripts\install-windows.ps1" -Profile NvidiaCuda %*
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
  echo.
  echo NVIDIA GPU install failed with exit code %EXIT_CODE%.
  echo See logs\install-windows-nvidia-*.log for details.
  set "LATEST_LOG="
  for /f "delims=" %%L in ('dir /b /o-d "%ROOT%logs\install-windows-nvidia-*.log" 2^>nul') do if not defined LATEST_LOG set "LATEST_LOG=%ROOT%logs\%%L"
  if defined LATEST_LOG echo Latest log: "!LATEST_LOG!"
  echo.
  echo Press any key to close this window.
  pause >nul
)
exit /b %EXIT_CODE%
