@echo off
setlocal EnableExtensions EnableDelayedExpansion
set "ROOT=%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%start-windows.ps1" %*
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
  echo.
  echo Server exited with code %EXIT_CODE%.
  set "LATEST_LOG="
  for /f "delims=" %%F in ('dir /b /a-d /o-d "%ROOT%logs\start-windows-*.log" 2^>nul') do if not defined LATEST_LOG set "LATEST_LOG=%ROOT%logs\%%F"
  if defined LATEST_LOG (
    echo Latest log: "!LATEST_LOG!"
  ) else (
    echo See logs\start-windows-*.log for details.
  )
  echo.
  echo Press any key to close this window.
  pause >nul
)
exit /b %EXIT_CODE%
