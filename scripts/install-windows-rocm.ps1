param(
    [string]$PythonExe = "",
    [string]$VenvPath = ".venv-rocm-win",
    [string]$WheelCache = ".download-cache\rocm-win-7.2.1",
    [switch]$InstallBuildTools,
    [switch]$RecreateVenv,
    [switch]$ForceRocmRuntime,
    [switch]$ForceTorch,
    [switch]$ForceDeps,
    [switch]$ForceFrontendBuild,
    [switch]$SkipFrontendBuild,
    [switch]$SkipGpuCheck,
    [switch]$StartServer
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "Continue"
$env:PIP_DISABLE_PIP_VERSION_CHECK = "1"

$RocmVersion = "7.2.1"
$TorchVersion = "2.9.1+rocm7.2.1"
$TorchVisionVersion = "0.24.1+rocm7.2.1"
$RocmBaseUrl = "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1"

$RocmRuntimeArtifacts = @(
    @{ Package = "rocm-sdk-core"; VersionPrefix = "7.2.1"; File = "rocm_sdk_core-7.2.1-py3-none-win_amd64.whl"; Url = "$RocmBaseUrl/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl" },
    @{ Package = "rocm-sdk-devel"; VersionPrefix = "7.2.1"; File = "rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl"; Url = "$RocmBaseUrl/rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl" },
    @{ Package = "rocm-sdk-libraries-custom"; VersionPrefix = "7.2.1"; File = "rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl"; Url = "$RocmBaseUrl/rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl" },
    @{ Package = "rocm"; VersionPrefix = "7.2.1"; File = "rocm-7.2.1.tar.gz"; Url = "$RocmBaseUrl/rocm-7.2.1.tar.gz" }
)

$TorchArtifacts = @(
    @{ Package = "torch"; VersionPrefix = $TorchVersion; File = "torch-2.9.1+rocm7.2.1-cp312-cp312-win_amd64.whl"; Url = "$RocmBaseUrl/torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl" },
    @{ Package = "torchaudio"; VersionPrefix = $TorchVersion; File = "torchaudio-2.9.1+rocm7.2.1-cp312-cp312-win_amd64.whl"; Url = "$RocmBaseUrl/torchaudio-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl" },
    @{ Package = "torchvision"; VersionPrefix = $TorchVisionVersion; File = "torchvision-0.24.1+rocm7.2.1-cp312-cp312-win_amd64.whl"; Url = "$RocmBaseUrl/torchvision-0.24.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl" }
)

function Write-Step($Message) {
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Write-Ok($Message) {
    Write-Host "[OK] $Message" -ForegroundColor Green
}

function Write-Warn($Message) {
    Write-Host "[WARN] $Message" -ForegroundColor Yellow
}

function Invoke-External($Exe, [string[]]$ArgumentList, $FailureMessage) {
    & $Exe @ArgumentList
    if ($LASTEXITCODE -ne 0) {
        throw "$FailureMessage (exit code $LASTEXITCODE)"
    }
}

function Normalize-CommandPath($Value) {
    if (-not $Value) {
        return ""
    }
    return $Value.Trim().Trim("'").Trim('"')
}

function Resolve-ProjectRoot {
    $scriptDir = Split-Path -Parent $MyInvocation.ScriptName
    return (Resolve-Path (Join-Path $scriptDir "..")).Path
}

function Initialize-InstallLog {
    $logDir = Join-Path (Resolve-Path ".").Path "logs"
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    $logPath = Join-Path $logDir ("install-windows-rocm-{0}.log" -f (Get-Date -Format "yyyyMMdd-HHmmss"))
    try {
        Start-Transcript -LiteralPath $logPath -Append | Out-Null
        Write-Host "Log: $logPath"
    } catch {
        Write-Warn "Unable to start transcript: $($_.Exception.Message)"
    }
}

function Assert-Windows {
    if (-not $IsWindows -and $env:OS -ne "Windows_NT") {
        throw "This script is for Windows only."
    }
}

function Assert-AMDGraphics($SkipCheck) {
    if ($SkipCheck) {
        Write-Warn "GPU check skipped."
        return
    }
    $names = @(Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name } | Where-Object { $_ })
    $summary = if ($names.Count) { $names -join " / " } else { "none" }
    $amd = $names | Where-Object { $_ -match "AMD|Radeon|Ryzen AI" }
    if (-not $amd) {
        throw "No AMD Radeon/Ryzen AI GPU detected. Detected: $summary. Run this on the AMD Windows machine, or pass -SkipGpuCheck only if you know the GPU is supported."
    }
    Write-Ok "Detected GPU: $($amd -join ' / ')"
}

function Test-Python312Candidate($Exe) {
    $Exe = Normalize-CommandPath $Exe
    if (-not $Exe) {
        return $null
    }
    try {
        $result = & $Exe -c "import sys; print(sys.executable if sys.version_info[:2] == (3, 12) else '')" 2>$null
        if ($LASTEXITCODE -eq 0 -and $result) {
            return ($result | Select-Object -First 1).Trim()
        }
    } catch {
        return $null
    }
    return $null
}

function Find-Python312($Preferred) {
    $candidates = @()
    if ($Preferred) {
        $candidates += $Preferred
    }
    if ($env:PYTHON312) {
        $candidates += $env:PYTHON312
    }

    $commonPaths = @(
        (Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"),
        (Join-Path $env:ProgramFiles "Python312\python.exe"),
        (Join-Path ${env:ProgramFiles(x86)} "Python312\python.exe")
    )
    foreach ($path in $commonPaths) {
        if ($path -and (Test-Path -LiteralPath $path)) {
            $candidates += $path
        }
    }

    $pyLauncher = Get-Command "py" -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        try {
            $py312 = & $pyLauncher.Source -3.12 -c "import sys; print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0 -and $py312) {
                $candidates += ($py312 | Select-Object -First 1)
            }
        } catch {
        }
    }

    foreach ($cmd in @("python3.12", "python")) {
        $found = Get-Command $cmd -ErrorAction SilentlyContinue
        if ($found) {
            $candidates += $found.Source
        }
    }

    $seen = @{}
    foreach ($candidate in $candidates) {
        $candidate = Normalize-CommandPath $candidate
        if (-not $candidate -or $seen.ContainsKey($candidate)) {
            continue
        }
        $seen[$candidate] = $true
        $python = Test-Python312Candidate $candidate
        if ($python) {
            return $python
        }
    }
    throw "Python 3.12 x64 was not found. Install Python 3.12, or run with -PythonExe C:\Path\To\Python312\python.exe."
}

function Assert-Python312($Exe) {
    $Exe = Normalize-CommandPath $Exe
    $versionJson = & $Exe -c "import json,sys; print(json.dumps({'major':sys.version_info.major,'minor':sys.version_info.minor,'exe':sys.executable}))"
    $info = $versionJson | ConvertFrom-Json
    if ($info.major -ne 3 -or $info.minor -ne 12) {
        throw "AMD ROCm Windows PyTorch wheels require Python 3.12. Current: Python $($info.major).$($info.minor) at $($info.exe). Pass -PythonExe C:\Path\To\Python312\python.exe."
    }
    Write-Ok "Python: $($info.exe)"
}

function Test-MSVCBuildTools {
    if (Get-Command "cl.exe" -ErrorAction SilentlyContinue) {
        return $true
    }
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path -LiteralPath $vswhere) {
        try {
            $installPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
            if ($LASTEXITCODE -eq 0 -and $installPath) {
                return $true
            }
        } catch {
        }
    }
    $fallback = @(
        (Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC"),
        (Join-Path $env:ProgramFiles "Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC")
    )
    foreach ($path in $fallback) {
        if ($path -and (Test-Path -LiteralPath $path)) {
            return $true
        }
    }
    return $false
}

function Ensure-MSVCBuildTools($InstallIfMissing) {
    if (Test-MSVCBuildTools) {
        Write-Ok "Microsoft C++ Build Tools detected."
        return
    }
    if ($InstallIfMissing) {
        $winget = Get-Command "winget" -ErrorAction SilentlyContinue
        if (-not $winget) {
            throw "Microsoft C++ Build Tools are missing and winget was not found. Install Visual Studio 2022 Build Tools with the C++ workload, then rerun this script."
        }
        Write-Warn "Installing Microsoft Visual Studio 2022 Build Tools. This can take several minutes."
        Invoke-External $winget.Source @(
            "install", "-e", "--id", "Microsoft.VisualStudio.2022.BuildTools",
            "--source", "winget",
            "--accept-package-agreements",
            "--accept-source-agreements",
            "--override", "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
        ) "Visual Studio Build Tools installation failed"
        if (Test-MSVCBuildTools) {
            Write-Ok "Microsoft C++ Build Tools installed."
            return
        }
    }
    throw @"
Microsoft C++ Build Tools are required to build chroma-hnswlib on Windows/Python 3.12.
Install them, then rerun this script:

winget install -e --id Microsoft.VisualStudio.2022.BuildTools --source winget --accept-package-agreements --accept-source-agreements --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"

Or rerun this script with -InstallBuildTools.
"@
}

function Test-VenvPython($PythonPath) {
    if (-not (Test-Path -LiteralPath $PythonPath)) {
        return $false
    }
    try {
        $probe = & $PythonPath -c "import sys; print(sys.executable)" 2>$null
        return ($LASTEXITCODE -eq 0 -and $probe)
    } catch {
        return $false
    }
}

function Remove-ProjectChildDirectory($Path) {
    $root = (Resolve-Path -LiteralPath ".").Path
    $target = (Resolve-Path -LiteralPath $Path -ErrorAction Stop).Path
    if (-not $target.StartsWith($root, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove outside project: $target"
    }
    if ((Split-Path -Parent $target) -ne $root) {
        throw "Refusing to remove nested or unexpected path: $target"
    }
    Remove-Item -LiteralPath $target -Recurse -Force
}

function Create-Or-RecreateVenv($PythonExe, $VenvPath, $ForceRecreate) {
    $venvPythonPath = Join-Path $VenvPath "Scripts\python.exe"
    if ((Test-Path -LiteralPath $VenvPath) -and ($ForceRecreate -or -not (Test-VenvPython $venvPythonPath))) {
        Write-Warn "Existing virtual environment is unusable or recreation was requested. Recreating: $VenvPath"
        Remove-ProjectChildDirectory $VenvPath
    }
    if (-not (Test-Path -LiteralPath $VenvPath)) {
        Invoke-External $PythonExe @("-m", "venv", $VenvPath) "Failed to create virtual environment"
    }
    $resolvedVenvPython = (Resolve-Path -LiteralPath $venvPythonPath -ErrorAction Stop).Path
    if (-not (Test-VenvPython $resolvedVenvPython)) {
        throw "The created virtual environment cannot run Python. Use official Python 3.12 from python.org, then rerun with -RecreateVenv."
    }
    return $resolvedVenvPython
}

function Get-PythonPackageVersion($Python, $PackageName) {
    try {
        $code = "import importlib.metadata as m; import sys; p=sys.argv[1];`ntry: print(m.version(p))`nexcept m.PackageNotFoundError: print('')"
        $value = & $Python -c $code $PackageName 2>$null
        if ($LASTEXITCODE -eq 0 -and $value) {
            return ($value | Select-Object -First 1).Trim()
        }
    } catch {
    }
    return ""
}

function Test-PackagesInstalled($Python, $Artifacts) {
    foreach ($artifact in $Artifacts) {
        $version = Get-PythonPackageVersion $Python $artifact.Package
        if (-not $version -or -not $version.StartsWith($artifact.VersionPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
            return $false
        }
    }
    return $true
}

function Test-RocmTorchReady($Python) {
    try {
        $code = "import torch; print(torch.__version__); print(torch.version.hip or ''); print(torch.cuda.is_available())"
        $lines = @(& $Python -c $code 2>$null)
        if ($LASTEXITCODE -ne 0 -or $lines.Count -lt 2) {
            return $false
        }
        $version = [string]$lines[0]
        $hip = [string]$lines[1]
        return ($version.StartsWith($TorchVersion, [System.StringComparison]::OrdinalIgnoreCase) -and $hip)
    } catch {
        return $false
    }
}

function Get-RemoteContentLength($Url) {
    try {
        $response = Invoke-WebRequest -Uri $Url -Method Head -UseBasicParsing -TimeoutSec 30
        $length = $response.Headers["Content-Length"]
        if ($length) {
            return [int64]($length | Select-Object -First 1)
        }
    } catch {
    }
    return 0
}

function Save-ArtifactIfNeeded($Artifact, $CacheDir) {
    New-Item -ItemType Directory -Force -Path $CacheDir | Out-Null
    $target = Join-Path $CacheDir $Artifact.File
    $remoteLength = Get-RemoteContentLength $Artifact.Url
    if (Test-Path -LiteralPath $target) {
        $localLength = (Get-Item -LiteralPath $target).Length
        if ($localLength -gt 0 -and ($remoteLength -le 0 -or $localLength -eq $remoteLength)) {
            Write-Ok "Using cached $($Artifact.File)"
            return $target
        }
        Write-Warn "Cached file is incomplete or size changed: $($Artifact.File). Redownloading."
        Remove-Item -LiteralPath $target -Force
    }

    Write-Host "Downloading $($Artifact.File)"
    $bits = Get-Command "Start-BitsTransfer" -ErrorAction SilentlyContinue
    if ($bits) {
        Start-BitsTransfer -Source $Artifact.Url -Destination $target
    } else {
        Invoke-WebRequest -Uri $Artifact.Url -OutFile $target -UseBasicParsing
    }
    if (-not (Test-Path -LiteralPath $target) -or (Get-Item -LiteralPath $target).Length -le 0) {
        throw "Download failed: $($Artifact.Url)"
    }
    if ($remoteLength -gt 0 -and (Get-Item -LiteralPath $target).Length -ne $remoteLength) {
        throw "Downloaded size mismatch for $($Artifact.File). Delete it from $CacheDir and rerun."
    }
    return $target
}

function Install-ArtifactGroup($Python, $Artifacts, $CacheDir, $Description, $Force, $ReadyTest) {
    if (-not $Force -and (& $ReadyTest)) {
        Write-Ok "$Description already installed; skipping."
        return
    }
    $files = @()
    foreach ($artifact in $Artifacts) {
        $files += (Save-ArtifactIfNeeded $artifact $CacheDir)
    }
    $args = @("-m", "pip", "install") + $files
    Invoke-External $Python $args "Failed to install $Description"
}

function Test-ProjectDepsReady($Python) {
    try {
        $code = "import importlib.util; mods=['uvicorn','fastapi','chromadb','qwen_asr','funasr','pyannote.audio']; missing=[m for m in mods if importlib.util.find_spec(m) is None]; print(','.join(missing)); raise SystemExit(1 if missing else 0)"
        $missing = & $Python -c $code 2>$null
        if ($LASTEXITCODE -eq 0) {
            return $true
        }
        if ($missing) {
            Write-Warn "Missing Python modules: $missing"
        }
    } catch {
    }
    return $false
}

function Install-ProjectDeps($Python, $Force) {
    if (-not $Force -and (Test-ProjectDepsReady $Python)) {
        Write-Ok "Project Python dependencies already installed; skipping."
        return
    }
    $tempRequirements = Join-Path $env:TEMP "matrix-live-diarizer-requirements-no-torch.txt"
    Get-Content -LiteralPath "requirements.txt" -Encoding UTF8 |
        Where-Object { $_ -notmatch "^\s*(torch|torchaudio|torchvision)==" } |
        Set-Content -LiteralPath $tempRequirements -Encoding UTF8
    Invoke-External $Python @("-m", "pip", "install", "-r", $tempRequirements) "Failed to install project dependencies"
    Invoke-External $Python @("-m", "pip", "check") "pip check failed"
}

function Set-EnvFileValue($Path, $Key, $Value) {
    $lines = @()
    if (Test-Path $Path) {
        $lines = Get-Content -LiteralPath $Path -Encoding UTF8
    }
    $seen = $false
    $out = foreach ($line in $lines) {
        if ($line -match "^\s*$([regex]::Escape($Key))\s*=") {
            $seen = $true
            "$Key=$Value"
        } else {
            $line
        }
    }
    if (-not $seen) {
        $out += "$Key=$Value"
    }
    Set-Content -LiteralPath $Path -Value $out -Encoding UTF8
}

function Ensure-EnvConfig {
    if (-not (Test-Path ".env")) {
        Copy-Item -LiteralPath ".env.example" -Destination ".env"
    }
    Set-EnvFileValue ".env" "ASR_ENGINE" "qwen3"
    Set-EnvFileValue ".env" "ASR_DEVICE" "cuda"
    Set-EnvFileValue ".env" "ASR_LOAD_TIMEOUT_SEC" "3600"
    Set-EnvFileValue ".env" "HF_HUB_DISABLE_XET" "1"
    Set-EnvFileValue ".env" "PYANNOTE_ROCM_DISABLE_LSTM_DROPOUT" "1"
}

function Ensure-FrontendBuild($Force, $Skip) {
    if ($Skip) {
        Write-Warn "Frontend build skipped."
        return
    }
    if (-not $Force -and (Test-Path -LiteralPath "web\dist\index.html")) {
        Write-Ok "Frontend build already exists; skipping."
        return
    }
    $npm = Get-Command "npm" -ErrorAction SilentlyContinue
    if (-not $npm) {
        throw "npm was not found. Install Node.js, or rerun with -SkipFrontendBuild if web\dist already exists."
    }
    Push-Location "web"
    try {
        Invoke-External $npm.Source @("install") "npm install failed"
        Invoke-External $npm.Source @("run", "build") "npm run build failed"
    } finally {
        Pop-Location
    }
}

function Verify-RocmPyTorch($Python) {
    $code = "import torch; print('torch=', torch.__version__); print('hip=', torch.version.hip); print('gpu_available=', torch.cuda.is_available()); print('device=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
    Invoke-External $Python @("-c", $code) "ROCm PyTorch verification failed"
}

try {
    Assert-Windows
    $ProjectRoot = Resolve-ProjectRoot
    Set-Location $ProjectRoot
    Initialize-InstallLog

    Write-Step "Checking AMD GPU"
    Assert-AMDGraphics $SkipGpuCheck

    Write-Step "Checking Python 3.12"
    $PythonExe = Find-Python312 $PythonExe
    Assert-Python312 $PythonExe

    Write-Step "Creating isolated ROCm virtual environment"
    $VenvPython = Create-Or-RecreateVenv $PythonExe $VenvPath $RecreateVenv
    Invoke-External $VenvPython @("-m", "pip", "install", "-U", "pip", "wheel", "setuptools") "Failed to update pip tooling"

    Write-Step "Installing AMD ROCm runtime"
    Install-ArtifactGroup $VenvPython $RocmRuntimeArtifacts $WheelCache "AMD ROCm runtime $RocmVersion" $ForceRocmRuntime {
        Test-PackagesInstalled $VenvPython $RocmRuntimeArtifacts
    }

    Write-Step "Installing AMD ROCm PyTorch"
    Install-ArtifactGroup $VenvPython $TorchArtifacts $WheelCache "AMD ROCm PyTorch $TorchVersion" $ForceTorch {
        Test-RocmTorchReady $VenvPython
    }

    Write-Step "Checking Microsoft C++ Build Tools"
    Ensure-MSVCBuildTools $InstallBuildTools

    Write-Step "Installing project dependencies"
    Install-ProjectDeps $VenvPython $ForceDeps

    Write-Step "Configuring .env for AMD GPU ASR"
    Ensure-EnvConfig

    Write-Step "Building frontend"
    Ensure-FrontendBuild $ForceFrontendBuild $SkipFrontendBuild

    Write-Step "Verifying ROCm PyTorch"
    Verify-RocmPyTorch $VenvPython

    Write-Host ""
    Write-Host "Installation complete." -ForegroundColor Green
    Write-Host "Use this Python: $VenvPython"
    Write-Host "Start command: $VenvPython main.py"

    if ($StartServer) {
        Write-Step "Starting server"
        Invoke-External $VenvPython @("main.py") "Server exited with an error"
    }
} finally {
    try {
        Stop-Transcript | Out-Null
    } catch {
    }
}
