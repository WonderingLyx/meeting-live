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
    [string]$Proxy = "",
    [string]$Aria2cExe = "",
    [int]$Aria2Connections = 16,
    [string]$Aria2LowestSpeedLimit = "20K",
    [switch]$InstallAria2,
    [switch]$PrintRocmUrls,
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

function Resolve-Aria2cPath {
    $candidate = Normalize-CommandPath $Aria2cExe
    if ($candidate) {
        if (Test-Path -LiteralPath $candidate) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
        $cmd = Get-Command $candidate -ErrorAction SilentlyContinue
        if ($cmd) {
            return $cmd.Source
        }
        throw "aria2c not found: $candidate"
    }

    $existing = Get-Command "aria2c.exe" -ErrorAction SilentlyContinue
    if ($existing) {
        return $existing.Source
    }

    if ($InstallAria2) {
        $winget = Get-Command "winget" -ErrorAction SilentlyContinue
        if (-not $winget) {
            Write-Warn "winget not found; install aria2 manually or pass -Aria2cExe C:\Path\aria2c.exe."
            return ""
        }
        Write-Warn "Installing aria2 with winget for faster resumable ROCm downloads."
        Invoke-External $winget.Source @(
            "install", "-e", "--id", "aria2.aria2",
            "--source", "winget",
            "--accept-package-agreements",
            "--accept-source-agreements"
        ) "aria2 installation failed"
        $installed = Get-Command "aria2c.exe" -ErrorAction SilentlyContinue
        if ($installed) {
            return $installed.Source
        }
        Write-Warn "aria2 was installed but aria2c.exe is not on PATH yet; restart PowerShell or pass -Aria2cExe."
    }
    return ""
}

function Resolve-ProjectRoot {
    $scriptDir = $PSScriptRoot
    if (-not $scriptDir -and $PSCommandPath) {
        $scriptDir = Split-Path -Parent $PSCommandPath
    }
    if (-not $scriptDir -and $MyInvocation.MyCommand.Path) {
        $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    }
    if (-not $scriptDir) {
        throw "Unable to resolve installer script directory."
    }
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

    $script:ResolvedDownloadProxy = Normalize-CommandPath $Proxy
    if (-not $script:ResolvedDownloadProxy) {
        $script:ResolvedDownloadProxy = Normalize-CommandPath ($env:HTTPS_PROXY)
    }
    if (-not $script:ResolvedDownloadProxy) {
        $script:ResolvedDownloadProxy = Normalize-CommandPath ($env:HTTP_PROXY)
    }
    if ($script:ResolvedDownloadProxy) {
        Write-Host "download proxy: $script:ResolvedDownloadProxy"
    }

    $script:ResolvedAria2cPath = Resolve-Aria2cPath
    if ($script:ResolvedAria2cPath) {
        Write-Host "ROCm downloader: aria2c ($script:ResolvedAria2cPath), connections=$Aria2Connections"
    } elseif (Get-Command "curl.exe" -ErrorAction SilentlyContinue) {
        Write-Host "ROCm downloader: curl.exe"
    } else {
        Write-Host "ROCm downloader: PowerShell fallback"
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
        throw "Microsoft C++ Build Tools installation finished, but the C++ toolchain was not detected. Restart PowerShell and rerun the script, or install the C++ workload manually."
    }
    Write-Warn "Microsoft C++ Build Tools not detected; continuing. Default Windows install skips native ChromaDB/pyannote packages and uses the built-in memory vector store."
    Write-Warn "Install Build Tools only when you deliberately enable optional packages that compile native Windows wheels."
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
        $iwrArgs = @{
            Uri = $Url
            Method = "Head"
            UseBasicParsing = $true
            TimeoutSec = 30
        }
        if ($script:ResolvedDownloadProxy) {
            $iwrArgs.Proxy = $script:ResolvedDownloadProxy
        }
        $response = Invoke-WebRequest @iwrArgs
        $contentType = [string]($response.Headers["Content-Type"] | Select-Object -First 1)
        if ($contentType -match "text/html") {
            return -1
        }
        $length = $response.Headers["Content-Length"]
        if ($length) {
            return [int64]($length | Select-Object -First 1)
        }
    } catch {
    }
    return 0
}

function Test-RemoteArtifactUrl($Url) {
    $length = Get-RemoteContentLength $Url
    if ($length -lt 0) {
        return @{
            Ok = $false
            Length = 0
            Reason = "remote returned HTML, not an artifact"
        }
    }
    if ($length -gt 0 -and $length -lt 1MB) {
        return @{
            Ok = $false
            Length = $length
            Reason = "remote artifact is suspiciously small ($length bytes)"
        }
    }
    return @{
        Ok = $true
        Length = $length
        Reason = ""
    }
}

function Invoke-RetryingDownload($Url, $Destination, [int64]$ExpectedLength) {
    $tempPath = "$Destination.part"
    $methods = @()
    if ($script:ResolvedAria2cPath) {
        $methods += @{ Name = "aria2c"; Kind = "aria2"; Path = $script:ResolvedAria2cPath }
    }
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
            try {
                if (Test-ArtifactFile $Destination $ExpectedLength) {
                    Write-Ok "Using cached $(Split-Path -Leaf $Destination)"
                    return
                }
                if (Test-Path -LiteralPath $tempPath) {
                    $partialLength = (Get-Item -LiteralPath $tempPath).Length
                    $canResumePartial = $method.Kind -in @("aria2", "curl")
                    if ($ExpectedLength -gt 0 -and $partialLength -gt $ExpectedLength) {
                        Write-Warn "Partial file is larger than remote size. Restarting this artifact download."
                        Remove-Item -LiteralPath $tempPath -Force
                    } elseif (-not $canResumePartial) {
                        Write-Warn ("Keeping partial file for resumable downloader; skipping {0} because it cannot resume." -f $method.Name)
                        continue
                    } elseif ($partialLength -gt 0) {
                        Write-Host ("Resuming partial download: {0:N1} MB" -f ($partialLength / 1MB))
                    }
                }
                Write-Host ("Downloading with {0} (attempt {1}/4)" -f $method.Name, $attempt)
                if ($method.Kind -eq "aria2") {
                    $connections = [Math]::Max(1, [Math]::Min(32, $Aria2Connections))
                    $ariaArgs = @(
                        "--continue=true",
                        "--max-connection-per-server=$connections",
                        "--split=$connections",
                        "--min-split-size=1M",
                        "--retry-wait=2",
                        "--max-tries=10",
                        "--timeout=60",
                        "--connect-timeout=30",
                        "--summary-interval=5",
                        "--lowest-speed-limit=$Aria2LowestSpeedLimit",
                        "--file-allocation=none",
                        "--allow-overwrite=true",
                        "--auto-file-renaming=false",
                        "--dir", (Split-Path -Parent $tempPath),
                        "--out", (Split-Path -Leaf $tempPath)
                    )
                    if ($script:ResolvedDownloadProxy) {
                        $ariaArgs += "--all-proxy=$script:ResolvedDownloadProxy"
                    }
                    $ariaArgs += $Url
                    Invoke-External $method.Path $ariaArgs "aria2c download failed"
                } elseif ($method.Kind -eq "curl") {
                    $curlArgs = @(
                        "--fail",
                        "--location",
                        "--retry", "5",
                        "--retry-delay", "2",
                        "--connect-timeout", "30",
                        "--continue-at", "-",
                        "--output", $tempPath,
                        $Url
                    )
                    if ($script:ResolvedDownloadProxy) {
                        $curlArgs = @("--proxy", $script:ResolvedDownloadProxy) + $curlArgs
                    }
                    Invoke-External $method.Path $curlArgs "curl.exe download failed"
                } elseif ($method.Kind -eq "bits") {
                    if ($script:ResolvedDownloadProxy) {
                        Write-Warn "BITS does not use the script proxy parameter reliably; skipping BITS while proxy is set."
                        continue
                    }
                    Start-BitsTransfer -Source $Url -Destination $tempPath -ErrorAction Stop
                } else {
                    $oldProgress = $ProgressPreference
                    try {
                        $script:ProgressPreference = "SilentlyContinue"
                        $iwrArgs = @{
                            Uri = $Url
                            OutFile = $tempPath
                            UseBasicParsing = $true
                            TimeoutSec = 3600
                        }
                        if ($script:ResolvedDownloadProxy) {
                            $iwrArgs.Proxy = $script:ResolvedDownloadProxy
                        }
                        Invoke-WebRequest @iwrArgs
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
        Write-Warn "Partial download kept for resume: $tempPath"
    }
    throw "Download failed after retries: $Url. Rerun the installer to resume from the .part file. Last error: $lastError"
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
    # Let pip validate wheel internals. PowerShell/.NET ZIP checks can falsely fail on
    # very large ROCm wheel files and waste long downloads.
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
    $partial = "$target.part"
    $sourceUrls = Get-ArtifactUrls $Artifact
    $remoteLength = 0
    foreach ($url in $sourceUrls) {
        $probe = Test-RemoteArtifactUrl $url
        if ($probe.Ok -and $probe.Length -gt 0) {
            $remoteLength = [int64]$probe.Length
            break
        }
    }
    if (Test-Path -LiteralPath $target) {
        if (Test-ArtifactFile $target $remoteLength) {
            Write-Ok "Using cached $($Artifact.File)"
            return $target
        }
        $localLength = (Get-Item -LiteralPath $target).Length
        if ($remoteLength -gt 0 -and $localLength -gt 0 -and $localLength -lt $remoteLength) {
            Write-Warn "Cached target is incomplete; moving it to .part for resume: $($Artifact.File)"
            if ((Test-Path -LiteralPath $partial) -and (Get-Item -LiteralPath $partial).Length -ge $localLength) {
                Remove-Item -LiteralPath $target -Force
            } else {
                Move-Item -LiteralPath $target -Destination $partial -Force
            }
        } else {
            Write-Warn "Cached file is invalid or size changed: $($Artifact.File). Restarting this artifact download."
            Remove-Item -LiteralPath $target -Force
        }
    }
    if ($remoteLength -gt 0 -and (Test-ArtifactFile $partial $remoteLength)) {
        Write-Ok "Promoting completed partial download: $($Artifact.File)"
        Move-Item -LiteralPath $partial -Destination $target -Force
        return $target
    } elseif ((Test-Path -LiteralPath $partial) -and $remoteLength -le 0) {
        Write-Warn "Remote size is unknown; keeping partial file for resumable downloader instead of treating it as complete."
    }

    Write-Host "Downloading $($Artifact.File)"
    $lastError = ""
    foreach ($url in $sourceUrls) {
        try {
            $probe = Test-RemoteArtifactUrl $url
            if (-not $probe.Ok) {
                throw $probe.Reason
            }
            $length = [int64]$probe.Length
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
                if (Test-ArtifactFile $target $length) {
                    return $target
                }
                $targetLength = (Get-Item -LiteralPath $target).Length
                if ($length -gt 0 -and $targetLength -gt 0 -and $targetLength -lt $length) {
                    Move-Item -LiteralPath $target -Destination $partial -Force
                }
            }
        }
    }
    throw "Download failed for $($Artifact.File). Last error: $lastError"
}

function Show-RocmDownloadPlan {
    Write-Host ""
    Write-Host "ROCm artifact cache:" -ForegroundColor Cyan
    Write-Host (Join-Path (Resolve-Path ".").Path $WheelCache)
    Write-Host ""
    Write-Host "Download these files into that directory, then rerun the installer:" -ForegroundColor Cyan
    foreach ($artifact in @($RocmRuntimeArtifacts + $TorchArtifacts)) {
        $urls = Get-ArtifactUrls $artifact
        Write-Host ""
        Write-Host $artifact.File
        foreach ($url in $urls) {
            Write-Host "  $url"
        }
    }
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
        $code = "import importlib.util; mods=['uvicorn','fastapi','qwen_asr','funasr','modelscope','librosa','soundfile']; missing=[m for m in mods if importlib.util.find_spec(m) is None]; print(','.join(missing)); raise SystemExit(1 if missing else 0)"
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
        Where-Object { $_ -notmatch "^\s*(torch|torchaudio|torchvision|chromadb|chroma-hnswlib|pyannote\.audio)([=<>!~ ;]|$)" } |
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
    Set-EnvFileValue ".env" "SPEAKER_ENGINE" "campplus"
    Set-EnvFileValue ".env" "SPEAKER_DEVICE" "cuda"
    Set-EnvFileValue ".env" "SPEAKER_VECTOR_STORE" "memory"
    Set-EnvFileValue ".env" "DIARIZATION_ENGINE" "funasr_campplus"
    Set-EnvFileValue ".env" "DIARIZATION_PROVIDER" "modelscope"
    Set-EnvFileValue ".env" "DIARIZATION_ENDPOINT" "https://modelscope.cn"
    Set-EnvFileValue ".env" "DIARIZATION_MODEL" "paraformer-zh + fsmn-vad + ct-punc + cam++"
    Set-EnvFileValue ".env" "PYANNOTE_DEVICE" "cuda"
    Set-EnvFileValue ".env" "HF_HUB_DISABLE_XET" "1"
    if ($script:ResolvedHfEndpoint) {
        Set-EnvFileValue ".env" "HF_ENDPOINT" $script:ResolvedHfEndpoint
    }
    Set-EnvFileValue ".env" "PYANNOTE_ROCM_DISABLE_LSTM_DROPOUT" "1"
}

function Ensure-ModelSettings {
    New-Item -ItemType Directory -Force -Path "config" | Out-Null
    $path = "config\model-settings.json"
    if (Test-Path -LiteralPath $path) {
        try {
            $settings = Get-Content -LiteralPath $path -Raw -Encoding UTF8 | ConvertFrom-Json
            Set-JsonProperty (Ensure-JsonObjectProperty $settings "asr") "device" "cuda"
            Set-JsonProperty (Ensure-JsonObjectProperty $settings "speaker") "device" "cuda"
            Set-JsonProperty (Ensure-JsonObjectProperty $settings "diarization") "device" "cuda"
            $settings | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $path -Encoding UTF8
            Write-Ok "Updated $path devices for AMD GPU"
        } catch {
            Write-Warn "Existing model settings could not be updated for AMD GPU: $($_.Exception.Message)"
        }
        return
    }
    $settings = [ordered]@{
        version = 1
        asr = [ordered]@{
            provider = "modelscope"
            endpoint = "https://modelscope.cn"
            api_key = ""
            model = "qwen3"
            device = "cuda"
            word_timestamps = $false
            load_timeout_sec = 3600
        }
        speaker = [ordered]@{
            provider = "modelscope"
            endpoint = "https://modelscope.cn"
            api_key = ""
            model = "campplus"
            device = "cuda"
        }
        diarization = [ordered]@{
            engine = "funasr_campplus"
            provider = "modelscope"
            endpoint = "https://modelscope.cn"
            api_key = ""
            model = "paraformer-zh + fsmn-vad + ct-punc + cam++"
            command = ""
            device = "cuda"
        }
        llm = [ordered]@{
            provider = "ollama"
            endpoint = "http://127.0.0.1:11434/v1"
            api_key = ""
            model = "qwen2.5:1.5b"
            enabled = $false
            allow_public = $false
            timeout_sec = 180
            max_input_tokens = 8000
            mock = $false
        }
    }
    $settings | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $path -Encoding UTF8
    Write-Ok "Created $path"
}

function Ensure-JsonObjectProperty($Object, [string]$Name) {
    if (-not $Object.PSObject.Properties[$Name]) {
        $Object | Add-Member -NotePropertyName $Name -NotePropertyValue ([pscustomobject]@{})
    }
    return $Object.$Name
}

function Set-JsonProperty($Object, [string]$Name, $Value) {
    if ($Object.PSObject.Properties[$Name]) {
        $Object.$Name = $Value
    } else {
        $Object | Add-Member -NotePropertyName $Name -NotePropertyValue $Value
    }
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

function Write-StartScripts {
    $defaultVenv = $VenvPath
    $launcher = @"
param(
    [string]`$VenvPath = "$defaultVenv",
    [string]`$Url = "http://127.0.0.1:8000",
    [switch]`$NoBrowser
)

`$ErrorActionPreference = "Stop"

`$Root = `$PSScriptRoot
if (-not `$Root -and `$PSCommandPath) {
    `$Root = Split-Path -Parent `$PSCommandPath
}
if (-not `$Root -and `$MyInvocation.MyCommand.Path) {
    `$Root = Split-Path -Parent `$MyInvocation.MyCommand.Path
}
if (-not `$Root) {
    throw "Unable to resolve project directory."
}

`$candidateVenvs = @()
if (`$VenvPath) {
    `$candidateVenvs += `$VenvPath
}
`$candidateVenvs += @(".venv-rocm-win", ".venv-win")

`$seen = @{}
`$Python = `$null
foreach (`$candidate in `$candidateVenvs) {
    if (-not `$candidate -or `$seen.ContainsKey(`$candidate)) {
        continue
    }
    `$seen[`$candidate] = `$true
    `$candidatePython = Join-Path `$Root (Join-Path `$candidate "Scripts\python.exe")
    if (Test-Path -LiteralPath `$candidatePython) {
        `$Python = `$candidatePython
        break
    }
}

if (-not `$Python) {
    throw "Python environment not found. Run .\install-windows.cmd first, or pass -VenvPath .venv-rocm-win."
}

Set-Location `$Root
`$env:HF_HUB_DISABLE_XET = "1"

if (-not `$NoBrowser) {
    Start-Job -ScriptBlock {
        param(`$TargetUrl)
        for (`$i = 0; `$i -lt 60; `$i++) {
            try {
                `$response = Invoke-WebRequest -Uri `$TargetUrl -UseBasicParsing -TimeoutSec 2
                if (`$response.StatusCode -ge 200 -and `$response.StatusCode -lt 500) {
                    Start-Process `$TargetUrl
                    return
                }
            } catch {
            }
            Start-Sleep -Seconds 1
        }
    } -ArgumentList `$Url | Out-Null
}

& `$Python "main.py"
"@
    Set-Content -LiteralPath "start-windows.ps1" -Value $launcher -Encoding UTF8
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
    if ($PrintRocmUrls) {
        Show-RocmDownloadPlan
        return
    }

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

    Write-Step "Checking optional Microsoft C++ Build Tools"
    Ensure-MSVCBuildTools $InstallBuildTools

    Write-Step "Installing project dependencies"
    Install-ProjectDeps $VenvPython $ForceDeps

    Write-Step "Configuring .env for AMD GPU ASR"
    Ensure-EnvConfig
    Ensure-ModelSettings
    Write-StartScripts

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
