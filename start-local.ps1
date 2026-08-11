$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Root ".conda\python.exe"
$FfmpegBin = Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0-full_build\bin"

if (-not (Test-Path $Python)) {
    throw "Python environment not found: $Python"
}

if (Test-Path $FfmpegBin) {
    $env:PATH = "$FfmpegBin;$env:PATH"
}

$env:HF_HUB_DISABLE_XET = "1"
$env:ASR_LOAD_TIMEOUT_SEC = "3600"

Set-Location $Root
& $Python "main.py"
