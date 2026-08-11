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
    [ValidateSet("China", "Official", "Auto")]
    [string]$MirrorMode = "China",
    [string[]]$RocmBaseUrls = @(),
    [string[]]$PipIndexUrls = @(),
    [string[]]$NpmRegistries = @(),
    [string]$HfEndpoint = "",
    [switch]$StartServer
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "Continue"
$env:PIP_DISABLE_PIP_VERSION_CHECK = "1"

$RocmVersion = "7.2.1"
$TorchVersion = "2.9.1+rocm7.2.1"
$TorchVisionVersion = "0.24.1+rocm7.2.1"
$OfficialRocmBaseUrl = "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1"
$ChinaPipIndexUrls = @(
    "https://pypi.tuna.tsinghua.edu.cn/simple",
    "https://mirrors.aliyun.com/pypi/simple"
)
$OfficialPipIndexUrl = "https://pypi.org/simple"
$ChinaNpmRegistries = @("https://registry.npmmirror.com/")
$OfficialNpmRegistry = "https://registry.npmjs.org/"
$ChinaHfEndpoint = "https://hf-mirror.com"

$RocmRuntimeArtifacts = @(
    @{ Package = "rocm-sdk-core"; VersionPrefix = "7.2.1"; File = "rocm_sdk_core-7.2.1-py3-none-win_amd64.whl"; UrlFile = "rocm_sdk_core-7.2.1-py3-none-win_amd64.whl"; Url = "$OfficialRocmBaseUrl/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl" },
    @{ Package = "rocm-sdk-devel"; VersionPrefix = "7.2.1"; File = "rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl"; UrlFile = "rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl"; Url = "$OfficialRocmBaseUrl/rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl" },
    @{ Package = "rocm-sdk-libraries-custom"; VersionPrefix = "7.2.1"; File = "rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl"; UrlFile = "rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl"; Url = "$OfficialRocmBaseUrl/rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl" },
    @{ Package = "rocm"; VersionPrefix = "7.2.1"; File = "rocm-7.2.1.tar.gz"; UrlFile = "rocm-7.2.1.tar.gz"; Url = "$OfficialRocmBaseUrl/rocm-7.2.1.tar.gz" }
)

$TorchArtifacts = @(
    @{ Package = "torch"; VersionPrefix = $TorchVersion; File = "torch-2.9.1+rocm7.2.1-cp312-cp312-win_amd64.whl"; UrlFile = "torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl"; Url = "$OfficialRocmBaseUrl/torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl" },
    @{ Package = "torchaudio"; VersionPrefix = $TorchVersion; File = "torchaudio-2.9.1+rocm7.2.1-cp312-cp312-win_amd64.whl"; UrlFile = "torchaudio-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl"; Url = "$OfficialRocmBaseUrl/torchaudio-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl" },
    @{ Package = "torchvision"; VersionPrefix = $TorchVisionVersion; File = "torchvision-0.24.1+rocm7.2.1-cp312-cp312-win_amd64.whl"; UrlFile = "torchvision-0.24.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl"; Url = "$OfficialRocmBaseUrl/torchvision-0.24.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl" }
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

function Add-UniqueValue([System.Collections.ArrayList]$List, $Value) {
    $valueText = [string]$Value
    if (-not $valueText) {
        return
    }
    $valueText = $valueText.Trim()
    if (-not $valueText) {
        return
    }
    foreach ($existing in $List) {
        if ([string]::Equals([string]$existing, $valueText, [System.StringComparison]::OrdinalIgnoreCase)) {
            return
        }
    }
    [void]$List.Add($valueText)
}

function Add-ListValues([System.Collections.ArrayList]$List, $Values) {
    foreach ($value in @($Values)) {
        if ($null -eq $value) {
            continue
        }
        foreach ($part in ([string]$value -split "[,;]")) {
            Add-UniqueValue $List $part
        }
    }
}

function Join-Url($Base, $Leaf) {
    return "$(([string]$Base).TrimEnd('/'))/$Leaf"
}

function Resolve-InstallerMirrors {
    $script:ResolvedRocmBaseUrls = New-Object System.Collections.ArrayList
    $script:ResolvedPipIndexUrls = New-Object System.Collections.ArrayList
    $script:ResolvedNpmRegistries = New-Object System.Collections.ArrayList

    Add-ListValues $script:ResolvedRocmBaseUrls $RocmBaseUrls
    Add-ListValues $script:ResolvedRocmBaseUrls $env:ROCM_BASE_URLS
    if ($script:ResolvedRocmBaseUrls.Count -eq 0 -and $MirrorMode -ne "Official") {
        Write-Warn "No verified public China mirror is known for AMD ROCm Windows wheels. Use -RocmBaseUrls or ROCM_BASE_URLS if you have a private mirror. Falling back to AMD official URLs."
    }
    Add-UniqueValue $script:ResolvedRocmBaseUrls $OfficialRocmBaseUrl

    Add-ListValues $script:ResolvedPipIndexUrls $PipIndexUrls
    if ($script:ResolvedPipIndexUrls.Count -eq 0) {
        if ($env:PIP_INDEX_URL) {
            Add-ListValues $script:ResolvedPipIndexUrls $env:PIP_INDEX_URL
        } elseif ($MirrorMode -eq "Official") {
            Add-UniqueValue $script:ResolvedPipIndexUrls $OfficialPipIndexUrl
        } else {
            Add-ListValues $script:ResolvedPipIndexUrls $ChinaPipIndexUrls
            Add-UniqueValue $script:ResolvedPipIndexUrls $OfficialPipIndexUrl
        }
    } elseif ($MirrorMode -ne "Official") {
        Add-UniqueValue $script:ResolvedPipIndexUrls $OfficialPipIndexUrl
    }

    Add-ListValues $script:ResolvedNpmRegistries $NpmRegistries
    if ($script:ResolvedNpmRegistries.Count -eq 0) {
        if ($env:NPM_CONFIG_REGISTRY) {
            Add-ListValues $script:ResolvedNpmRegistries $env:NPM_CONFIG_REGISTRY
        } elseif ($MirrorMode -eq "Official") {
            Add-UniqueValue $script:ResolvedNpmRegistries $OfficialNpmRegistry
        } else {
            Add-ListValues $script:ResolvedNpmRegistries $ChinaNpmRegistries
            Add-UniqueValue $script:ResolvedNpmRegistries $OfficialNpmRegistry
        }
    } elseif ($MirrorMode -ne "Official") {
        Add-UniqueValue $script:ResolvedNpmRegistries $OfficialNpmRegistry
    }

    if ($HfEndpoint) {
        $script:ResolvedHfEndpoint = $HfEndpoint.Trim().TrimEnd("/")
    } elseif ($env:HF_ENDPOINT) {
        $script:ResolvedHfEndpoint = $env:HF_ENDPOINT.Trim().TrimEnd("/")
    } elseif ($MirrorMode -eq "Official") {
        $script:ResolvedHfEndpoint = ""
    } else {
        $script:ResolvedHfEndpoint = $ChinaHfEndpoint
    }

    Write-Host "Mirror mode: $MirrorMode"
    Write-Host "ROCm sources: $($script:ResolvedRocmBaseUrls -join ', ')"
    Write-Host "pip indexes: $($script:ResolvedPipIndexUrls -join ', ')"
    Write-Host "npm registries: $($script:ResolvedNpmRegistries -join ', ')"
    if ($script:ResolvedHfEndpoint) {
        Write-Host "Hugging Face endpoint: $script:ResolvedHfEndpoint"
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

function Invoke-RetryingDownload($Url, $Destination, [int64]$ExpectedLength) {
    $tempPath = "$Destination.part"
    $methods = @()
    $curl = Get-Command "curl.exe" -ErrorAction SilentlyContinue
    if ($curl) {
        $methods += @{ Name = "curl.exe"; Kind = "curl"; Path = $curl.Source }
    }
    if (Get-Command "Start-BitsTransfer" -ErrorAction SilentlyContinue) {
        $methods += @{ Name = "BITS"; Kind = "bits"; Path = "" }
    }
    $methods += @{ Name = "Invoke-WebRequest"; Kind = "iwr"; Path = "" }

    $lastError = ""
    for ($attempt = 1; $attempt -le 4; $attempt++) {
        foreach ($method in $methods) {
            if (Test-Path -LiteralPath $tempPath) {
                Remove-Item -LiteralPath $tempPath -Force
            }
            try {
                Write-Host ("Downloading with {0} (attempt {1}/4)" -f $method.Name, $attempt)
                if ($method.Kind -eq "curl") {
                    Invoke-External $method.Path @(
                        "--fail",
                        "--location",
                        "--retry", "5",
                        "--retry-delay", "2",
                        "--connect-timeout", "30",
                        "--output", $tempPath,
                        $Url
                    ) "curl.exe download failed"
                } elseif ($method.Kind -eq "bits") {
                    Start-BitsTransfer -Source $Url -Destination $tempPath -ErrorAction Stop
                } else {
                    $oldProgress = $ProgressPreference
                    try {
                        $script:ProgressPreference = "SilentlyContinue"
                        Invoke-WebRequest -Uri $Url -OutFile $tempPath -UseBasicParsing -TimeoutSec 3600
                    } finally {
                        $script:ProgressPreference = $oldProgress
                    }
                }

                if (-not (Test-Path -LiteralPath $tempPath)) {
                    throw "download produced no file"
                }
                $actualLength = (Get-Item -LiteralPath $tempPath).Length
                if ($actualLength -le 0) {
                    throw "downloaded file is empty"
                }
                if ($ExpectedLength -gt 0 -and $actualLength -ne $ExpectedLength) {
                    throw "size mismatch: expected $ExpectedLength bytes, got $actualLength bytes"
                }
                Move-Item -LiteralPath $tempPath -Destination $Destination -Force
                return
            } catch {
                $lastError = $_.Exception.Message
                Write-Warn ("{0} failed: {1}" -f $method.Name, $lastError)
            }
        }
        Start-Sleep -Seconds ([Math]::Min(20, 2 * $attempt))
    }

    if (Test-Path -LiteralPath $tempPath) {
        Remove-Item -LiteralPath $tempPath -Force
    }
    throw "Download failed after retries: $Url. Last error: $lastError"
}

function Test-ZipArchive($Path) {
    try {
        Add-Type -AssemblyName System.IO.Compression.FileSystem -ErrorAction SilentlyContinue
        $zip = [System.IO.Compression.ZipFile]::OpenRead($Path)
        $zip.Dispose()
        return $true
    } catch {
        return $false
    }
}

function Test-ArtifactFile($Path, [int64]$ExpectedLength) {
    if (-not (Test-Path -LiteralPath $Path)) {
        return $false
    }
    $item = Get-Item -LiteralPath $Path
    if ($item.Length -le 0) {
        return $false
    }
    if ($ExpectedLength -gt 0 -and $item.Length -ne $ExpectedLength) {
        return $false
    }
    if ($Path.EndsWith(".whl", [System.StringComparison]::OrdinalIgnoreCase)) {
        return (Test-ZipArchive $Path)
    }
    return $true
}

function Get-ArtifactUrls($Artifact) {
    $urls = New-Object System.Collections.ArrayList
    foreach ($baseUrl in $script:ResolvedRocmBaseUrls) {
        if ($Artifact.UrlFile) {
            Add-UniqueValue $urls (Join-Url $baseUrl $Artifact.UrlFile)
        }
    }
    Add-UniqueValue $urls $Artifact.Url
    return @($urls)
}

function Save-ArtifactIfNeeded($Artifact, $CacheDir) {
    New-Item -ItemType Directory -Force -Path $CacheDir | Out-Null
    $target = Join-Path $CacheDir $Artifact.File
    $sourceUrls = Get-ArtifactUrls $Artifact
    $remoteLength = 0
    foreach ($url in $sourceUrls) {
        $remoteLength = Get-RemoteContentLength $url
        if ($remoteLength -gt 0) {
            break
        }
    }
    if (Test-Path -LiteralPath $target) {
        if (Test-ArtifactFile $target $remoteLength) {
            Write-Ok "Using cached $($Artifact.File)"
            return $target
        }
        Write-Warn "Cached file is incomplete or size changed: $($Artifact.File). Redownloading."
        Remove-Item -LiteralPath $target -Force
    }

    Write-Host "Downloading $($Artifact.File)"
    $lastError = ""
    foreach ($url in $sourceUrls) {
        try {
            $length = Get-RemoteContentLength $url
            Invoke-RetryingDownload $url $target $length
            if (-not (Test-ArtifactFile $target $length)) {
                throw "downloaded artifact failed validation"
            }
            return $target
        } catch {
            $lastError = $_.Exception.Message
            Write-Warn "Download source failed: $url"
            Write-Warn $lastError
            if (Test-Path -LiteralPath $target) {
                Remove-Item -LiteralPath $target -Force
            }
        }
    }
    throw "Download failed for $($Artifact.File). Last error: $lastError"
}

function Invoke-PipInstallWithMirrors($Python, [string[]]$InstallArgs, $Description) {
    $lastError = ""
    foreach ($indexUrl in $script:ResolvedPipIndexUrls) {
        try {
            Write-Host "pip index: $indexUrl"
            $args = @(
                "-m", "pip"
            ) + $InstallArgs + @(
                "--index-url", $indexUrl,
                "--retries", "5",
                "--timeout", "120",
                "--prefer-binary"
            )
            Invoke-External $Python $args $Description
            return
        } catch {
            $lastError = $_.Exception.Message
            Write-Warn "$Description via $indexUrl failed: $lastError"
        }
    }
    throw "$Description failed with all pip indexes. Last error: $lastError"
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
    $args = @("install") + $files
    Invoke-PipInstallWithMirrors $Python $args "Failed to install $Description"
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
    Invoke-PipInstallWithMirrors $Python @("install", "-r", $tempRequirements) "Failed to install project dependencies"
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
    if ($script:ResolvedHfEndpoint) {
        Set-EnvFileValue ".env" "HF_ENDPOINT" $script:ResolvedHfEndpoint
    }
    Set-EnvFileValue ".env" "PYANNOTE_ROCM_DISABLE_LSTM_DROPOUT" "1"
}

function Invoke-NpmInstallWithMirrors($Npm) {
    $lastError = ""
    foreach ($registry in $script:ResolvedNpmRegistries) {
        try {
            Write-Host "npm registry: $registry"
            Invoke-External $Npm.Source @(
                "install",
                "--registry=$registry",
                "--fetch-retries=5",
                "--fetch-retry-mintimeout=20000",
                "--fetch-retry-maxtimeout=120000"
            ) "npm install failed"
            return
        } catch {
            $lastError = $_.Exception.Message
            Write-Warn "npm install via $registry failed: $lastError"
        }
    }
    throw "npm install failed with all registries. Last error: $lastError"
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
        Invoke-NpmInstallWithMirrors $npm
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
    Resolve-InstallerMirrors

    Write-Step "Checking AMD GPU"
    Assert-AMDGraphics $SkipGpuCheck

    Write-Step "Checking Python 3.12"
    $PythonExe = Find-Python312 $PythonExe
    Assert-Python312 $PythonExe

    Write-Step "Creating isolated ROCm virtual environment"
    $VenvPython = Create-Or-RecreateVenv $PythonExe $VenvPath $RecreateVenv
    Invoke-PipInstallWithMirrors $VenvPython @("install", "-U", "pip", "wheel", "setuptools") "Failed to update pip tooling"

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
