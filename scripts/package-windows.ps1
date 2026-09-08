param(
    [string]$OutputDir = "dist-packages",
    [string]$PackageName = "",
    [ValidateSet("Auto", "Cpu", "AmdRocm", "NvidiaCuda")]
    [string]$Profile = "Auto",
    [switch]$IncludeRocmCache,
    [switch]$RequireRocmCache,
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

$RocmCacheRelativePath = ".download-cache\rocm-win-7.2.1"
$RequiredRocmArtifacts = @(
    @{ Name = "rocm_sdk_core-7.2.1-py3-none-win_amd64.whl"; Expected = 644793492L },
    @{ Name = "rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl"; Expected = 232840013L },
    @{ Name = "rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl"; Expected = 489964648L },
    @{ Name = "rocm-7.2.1.tar.gz"; Expected = 15940L },
    @{ Name = "torch-2.9.1+rocm7.2.1-cp312-cp312-win_amd64.whl"; Expected = 821065907L },
    @{ Name = "torchaudio-2.9.1+rocm7.2.1-cp312-cp312-win_amd64.whl"; Expected = 514708L },
    @{ Name = "torchvision-0.24.1+rocm7.2.1-cp312-cp312-win_amd64.whl"; Expected = 1863312L }
)

function Test-RocmCacheComplete($Root) {
    $cache = Join-Path $Root $RocmCacheRelativePath
    $missing = @()
    foreach ($artifact in $RequiredRocmArtifacts) {
        $path = Join-Path $cache $artifact.Name
        if (-not (Test-Path -LiteralPath $path)) {
            $missing += $artifact.Name
            continue
        }
        $actual = (Get-Item -LiteralPath $path).Length
        if ($actual -ne [int64]$artifact.Expected) {
            $missing += ("{0} expected={1} actual={2}" -f $artifact.Name, $artifact.Expected, $actual)
        }
    }
    return @{
        Cache = $cache
        Complete = ($missing.Count -eq 0)
        Missing = $missing
    }
}

function Test-PythonRuntimeAssetComplete($Root) {
    $pythonAssetDir = Join-Path $Root "offline\python"
    if (Test-Path -LiteralPath $pythonAssetDir) {
        $runtimeArchive = Get-ChildItem -LiteralPath $pythonAssetDir -File -Recurse -ErrorAction SilentlyContinue |
            Where-Object {
                $_.Name -match "(?i)^python\.3\.12.*\.nupkg$" -or
                $_.Name -match "(?i)^python-3\.12.*(amd64|x64).*\.(zip|nupkg)$"
            } |
            Sort-Object LastWriteTime -Descending |
            Select-Object -First 1
        if ($runtimeArchive -and $runtimeArchive.Length -gt 10MB) {
            return @{
                Complete = $true
                Path = $runtimeArchive.FullName
                Reason = "offline portable Python runtime archive"
            }
        }
    }

    return @{
        Complete = $false
        Path = $pythonAssetDir
        Reason = "missing offline\python\python.3.12.x.nupkg"
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

$effectiveProfile = $Profile
if ($effectiveProfile -eq "Auto") {
    if ($PackageName -match "(?i)(amd|rocm)") {
        $effectiveProfile = "AmdRocm"
    } elseif ($PackageName -match "(?i)nvidia") {
        $effectiveProfile = "NvidiaCuda"
    } else {
        $effectiveProfile = "Cpu"
    }
}
if ($effectiveProfile -eq "AmdRocm") {
    $IncludeRocmCache = $true
    $RequireRocmCache = $true
}

$rocmCacheState = $null
if ($IncludeRocmCache) {
    $rocmCacheState = Test-RocmCacheComplete $root
    if (-not $rocmCacheState.Complete) {
        $message = "ROCm offline cache is incomplete: $($rocmCacheState.Cache). Missing: $($rocmCacheState.Missing -join ', ')"
        if ($RequireRocmCache) {
            throw $message
        }
        Write-Warning $message
    }
}
$pythonAssetState = Test-PythonRuntimeAssetComplete $root
if (-not $pythonAssetState.Complete) {
    throw "Windows package requires a project-local portable Python 3.12 runtime archive. Put python.3.12.x.nupkg under offline\python. Checked: $($pythonAssetState.Path)"
}
Write-Host "Included Python runtime asset: $($pythonAssetState.Path) ($($pythonAssetState.Reason))" -ForegroundColor Green

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
    $removedStage = $false
    for ($attempt = 1; $attempt -le 3; $attempt++) {
        try {
            Remove-Item -LiteralPath $stage -Recurse -Force -ErrorAction Stop
            $removedStage = $true
            break
        } catch {
            Start-Sleep -Seconds 2
        }
    }
    if (-not $removedStage) {
        $fallbackStage = Join-Path $outRoot ("_stage-{0}-{1}" -f $PackageName, (Get-Date -Format "yyyyMMdd-HHmmss"))
        Write-Warning "Package workspace is in use: $stage. Using temporary workspace: $fallbackStage"
        $stage = $fallbackStage
    }
}
New-Item -ItemType Directory -Force -Path $stage | Out-Null

$items = @(
    "app",
    "config",
    "docs",
    "engine",
    "scripts",
    "offline",
    "web\dist",
    "web\package.json",
    "web\package-lock.json",
    ".env.example",
    "install-windows-amd-gpu.cmd",
    "install-windows-nvidia-gpu.cmd",
    "install-windows.cmd",
    "package-windows.cmd",
    "repair-face-runtime.cmd",
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
    ".runtime",
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

$stagePythonAssets = Join-Path $stage "offline\python"
if (Test-Path -LiteralPath $stagePythonAssets) {
    Get-ChildItem -LiteralPath $stagePythonAssets -File -Recurse -ErrorAction SilentlyContinue |
        Where-Object { $_.Extension -ieq ".exe" } |
        Remove-Item -Force
}

Get-ChildItem -LiteralPath $stage -Recurse -Force -Directory -Filter "__pycache__" |
    Remove-Item -Recurse -Force
Get-ChildItem -LiteralPath $stage -Recurse -Force -File -Filter "*.pyc" |
    Remove-Item -Force
Get-ChildItem -LiteralPath $stage -Recurse -Force -File -Filter "*.pyo" |
    Remove-Item -Force

if ($IncludeRocmCache) {
    if ($rocmCacheState -and $rocmCacheState.Complete) {
        $targetCache = Join-Path $stage $RocmCacheRelativePath
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $targetCache) | Out-Null
        Copy-Item -LiteralPath $rocmCacheState.Cache -Destination $targetCache -Recurse -Force
        Write-Host "Included ROCm offline cache: $RocmCacheRelativePath" -ForegroundColor Green
    }
}

New-Item -ItemType Directory -Force -Path (Join-Path $stage "data") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $stage "models") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $stage "logs") | Out-Null

Write-Step "Compressing package"
$zipPath = Join-Path $outRoot "$PackageName.zip"
if (Test-Path -LiteralPath $zipPath) {
    $removed = $false
    for ($attempt = 1; $attempt -le 5; $attempt++) {
        try {
            Remove-Item -LiteralPath $zipPath -Force -ErrorAction Stop
            $removed = $true
            break
        } catch {
            Start-Sleep -Seconds 2
        }
    }
    if (-not $removed) {
        throw "Package zip is in use and cannot be overwritten: $zipPath. Close File Explorer/archive tools using it, or use a different -PackageName."
    }
}
Compress-Archive -Path (Join-Path $stage "*") -DestinationPath $zipPath -Force
if (-not (Test-Path -LiteralPath $zipPath)) {
    throw "Package zip was not created: $zipPath"
}

Write-Host ""
Write-Host "Package created:" -ForegroundColor Green
Write-Host $zipPath
