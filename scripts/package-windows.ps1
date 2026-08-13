param(
    [string]$OutputDir = "dist-packages",
    [string]$PackageName = "",
    [switch]$SkipFrontendBuild
)

$ErrorActionPreference = "Stop"

function Write-Step($Message) {
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Invoke-External($Exe, [string[]]$ArgumentList, $FailureMessage) {
    & $Exe @ArgumentList
    if ($LASTEXITCODE -ne 0) {
        throw "$FailureMessage (exit code $LASTEXITCODE)"
    }
}

$scriptDir = $PSScriptRoot
if (-not $scriptDir -and $PSCommandPath) {
    $scriptDir = Split-Path -Parent $PSCommandPath
}
if (-not $scriptDir -and $MyInvocation.MyCommand.Path) {
    $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
}
if (-not $scriptDir) {
    throw "Unable to resolve package script directory."
}
$root = (Resolve-Path (Join-Path $scriptDir "..")).Path
Set-Location $root

if (-not $PackageName) {
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $PackageName = "matrix-live-diarizer-windows-$stamp"
}

Write-Step "Preparing frontend build"
if (-not $SkipFrontendBuild) {
    $npm = Get-Command "npm" -ErrorAction SilentlyContinue
    if (-not $npm) {
        throw "npm was not found. Install Node.js, or pass -SkipFrontendBuild when web\dist already exists."
    }
    Push-Location "web"
    try {
        Invoke-External $npm.Source @("install", "--registry=https://registry.npmmirror.com/") "npm install failed"
        Invoke-External $npm.Source @("run", "build") "npm run build failed"
    } finally {
        Pop-Location
    }
}
if (-not (Test-Path -LiteralPath "web\dist\index.html")) {
    throw "web\dist\index.html not found. Build frontend before packaging."
}

Write-Step "Creating package workspace"
$outRoot = Join-Path $root $OutputDir
New-Item -ItemType Directory -Force -Path $outRoot | Out-Null
$stage = Join-Path $outRoot $PackageName
if (Test-Path -LiteralPath $stage) {
    Remove-Item -LiteralPath $stage -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $stage | Out-Null

$items = @(
    "app",
    "config",
    "docs",
    "engine",
    "scripts",
    "web\dist",
    "web\package.json",
    "web\package-lock.json",
    ".env.example",
    "install-windows-amd-gpu.cmd",
    "install-windows-nvidia-gpu.cmd",
    "install-windows.cmd",
    "package-windows.cmd",
    "start-windows.cmd",
    "start-windows.ps1",
    "main.py",
    "pyproject.toml",
    "requirements.txt",
    "README.md",
    "README.en.md",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md"
)

foreach ($item in $items) {
    if (-not (Test-Path -LiteralPath $item)) {
        continue
    }
    $target = Join-Path $stage $item
    $targetParent = Split-Path -Parent $target
    New-Item -ItemType Directory -Force -Path $targetParent | Out-Null
    Copy-Item -LiteralPath $item -Destination $target -Recurse -Force
}

$excludedPaths = @(
    ".env",
    ".env.local",
    "config\model-settings.json",
    ".download-cache",
    ".venv",
    ".venv-win",
    ".venv-rocm-win",
    ".venv-nvidia-win",
    ".conda",
    "web\node_modules",
    "data",
    "models",
    "logs",
    "uploads",
    ".pytest_cache"
)

foreach ($item in $excludedPaths) {
    $path = Join-Path $stage $item
    if (Test-Path -LiteralPath $path) {
        Remove-Item -LiteralPath $path -Recurse -Force
    }
}

Get-ChildItem -LiteralPath $stage -Recurse -Force -Directory -Filter "__pycache__" |
    Remove-Item -Recurse -Force
Get-ChildItem -LiteralPath $stage -Recurse -Force -File -Filter "*.pyc" |
    Remove-Item -Force
Get-ChildItem -LiteralPath $stage -Recurse -Force -File -Filter "*.pyo" |
    Remove-Item -Force

New-Item -ItemType Directory -Force -Path (Join-Path $stage "data") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $stage "models") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $stage "logs") | Out-Null

Write-Step "Compressing package"
$zipPath = Join-Path $outRoot "$PackageName.zip"
if (Test-Path -LiteralPath $zipPath) {
    Remove-Item -LiteralPath $zipPath -Force
}
Compress-Archive -Path (Join-Path $stage "*") -DestinationPath $zipPath -Force
if (-not (Test-Path -LiteralPath $zipPath)) {
    throw "Package zip was not created: $zipPath"
}

Write-Host ""
Write-Host "Package created:" -ForegroundColor Green
Write-Host $zipPath
