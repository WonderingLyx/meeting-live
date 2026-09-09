param(
    [ValidateSet("Cpu", "AmdRocm", "NvidiaCuda")]
    [string]$Profile = "Cpu",
    [string]$PythonExe = "",
    [string]$VenvPath = "",
    [string]$PythonRuntimeDir = ".runtime\python-3.12",
    [string]$PythonInstaller = "",
    [string[]]$PythonRuntimeUrls = @(),
    [string]$OfflineAssetsDir = "offline",
    [string]$RocmWheelCache = ".download-cache\rocm-win-7.2.1",
    [string]$NvidiaWheelCache = ".download-cache\nvidia-wheels",
    [ValidateSet("China", "Official", "Auto")]
    [string]$MirrorMode = "China",
    [string[]]$PipIndexUrls = @(),
    [string[]]$NpmRegistries = @(),
    [string]$HfEndpoint = "",
    [string[]]$RocmBaseUrls = @(),
    [string]$Proxy = "",
    [string]$Aria2cExe = "",
    [int]$Aria2Connections = 16,
    [string]$Aria2LowestSpeedLimit = "20K",
    [ValidateSet("cu126", "cu128", "cu130")]
    [string]$CudaWheel = "cu128",
    [ValidateSet("auto", "latest", "stable")]
    [string]$TorchBuild = "stable",
    [string[]]$TorchIndexUrls = @(),
    [string[]]$TorchFindLinks = @(),
    [switch]$InstallAria2,
    [switch]$PrintRocmUrls,
    [switch]$OfflineOnly,
    [switch]$SkipGpuCheck,
    [switch]$ForceTorch,
    [switch]$RecreateVenv,
    [switch]$ForceDeps,
    [switch]$ForceFrontendBuild,
    [switch]$SkipFrontendBuild,
    [switch]$InstallChroma,
    [switch]$InstallPyannote,
    [switch]$InstallFfmpeg,
    [switch]$StartServer
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "Continue"
$env:PIP_DISABLE_PIP_VERSION_CHECK = "1"
$StableFunasrVersion = "1.4.1"
$PythonRuntimePackageVersion = "3.12.10"
$PythonRuntimePackageFile = "python.$PythonRuntimePackageVersion.nupkg"
$StableQwenAsrVersion = "0.0.6"
$StableInsightFaceVersion = "1.0.1"
$StableOnnxRuntimeCpuVersion = "1.29.0"
$StableOnnxRuntimeGpuVersion = "1.29.0"
$StableOnnxRuntimeDirectmlVersion = "1.24.4"
$StableOnnxVersion = "1.22.0"
$StableOpenCvVersion = "4.14.0.94"
$StablePillowVersion = "12.3.0"
$StableScikitImageVersion = "0.26.0"

$script:NumericRuntimeThreadEnv = [ordered]@{
    OPENBLAS_NUM_THREADS = "1"
    OMP_NUM_THREADS = "1"
    MKL_NUM_THREADS = "1"
    NUMEXPR_NUM_THREADS = "1"
    VECLIB_MAXIMUM_THREADS = "1"
    BLIS_NUM_THREADS = "1"
    GOTO_NUM_THREADS = "1"
    OPENBLAS_MAIN_FREE = "1"
}

$ChinaPipIndexUrls = @(
    "https://pypi.tuna.tsinghua.edu.cn/simple",
    "https://mirrors.aliyun.com/pypi/simple"
)
$OfficialPipIndexUrl = "https://pypi.org/simple"
$ChinaNpmRegistries = @("https://registry.npmmirror.com/")
$OfficialNpmRegistry = "https://registry.npmjs.org/"
$ChinaHfEndpoint = "https://hf-mirror.com"
$ChinaTorchFindLinkBaseUrls = @(
    "https://mirror.nju.edu.cn/pytorch/whl",
    "https://mirrors.aliyun.com/pytorch-wheels",
    "https://mirror.sjtu.edu.cn/pytorch-wheels"
)
$OfficialTorchIndexBaseUrl = "https://download.pytorch.org/whl"

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

function Write-Utf8NoBomText([string]$Path, [string]$Value) {
    $fullPath = [System.IO.Path]::GetFullPath($Path)
    $parent = [System.IO.Path]::GetDirectoryName($fullPath)
    if ($parent) {
        [System.IO.Directory]::CreateDirectory($parent) | Out-Null
    }
    $encoding = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($fullPath, $Value, $encoding)
}

function Write-JsonNoBom([string]$Path, $Value) {
    Write-Utf8NoBomText $Path (($Value | ConvertTo-Json -Depth 8) + [Environment]::NewLine)
}

function Set-NumericRuntimeThreadEnv {
    foreach ($item in $script:NumericRuntimeThreadEnv.GetEnumerator()) {
        [System.Environment]::SetEnvironmentVariable($item.Key, $item.Value, "Process")
        Set-Item -Path "Env:$($item.Key)" -Value $item.Value
    }
}

function Invoke-External($Exe, [string[]]$ArgumentList, $FailureMessage) {
    $oldErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & $Exe @ArgumentList 2>&1 | ForEach-Object { Write-Host $_ }
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $oldErrorActionPreference
    }
    if ($exitCode -ne 0) {
        throw "$FailureMessage (exit code $exitCode)"
    }
}

function Invoke-NativeQuietExitCode($Exe, [string[]]$ArgumentList) {
    $oldErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & $Exe @ArgumentList *> $null
        return $LASTEXITCODE
    } catch {
        return 1
    } finally {
        $ErrorActionPreference = $oldErrorActionPreference
    }
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

function Initialize-InstallLog($Name) {
    $logDir = Join-Path (Resolve-Path ".").Path "logs"
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    $logPath = Join-Path $logDir ("{0}-{1}.log" -f $Name, (Get-Date -Format "yyyyMMdd-HHmmss"))
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

function Resolve-InstallerMirrors {
    $script:ResolvedPipIndexUrls = New-Object System.Collections.ArrayList
    $script:ResolvedNpmRegistries = New-Object System.Collections.ArrayList

    Add-ListValues $script:ResolvedPipIndexUrls $PipIndexUrls
    if ($script:ResolvedPipIndexUrls.Count -eq 0) {
        if ($MirrorMode -eq "Auto" -and $env:PIP_INDEX_URL) {
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
        if ($MirrorMode -eq "Auto" -and $env:NPM_CONFIG_REGISTRY) {
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
    } elseif ($MirrorMode -eq "Auto" -and $env:HF_ENDPOINT) {
        $script:ResolvedHfEndpoint = $env:HF_ENDPOINT.Trim().TrimEnd("/")
    } elseif ($MirrorMode -eq "Official") {
        $script:ResolvedHfEndpoint = ""
    } else {
        $script:ResolvedHfEndpoint = $ChinaHfEndpoint
    }

    Write-Host "Profile: $Profile"
    Write-Host "Mirror mode: $MirrorMode"
    Write-Host "pip indexes: $($script:ResolvedPipIndexUrls -join ', ')"
    Write-Host "npm registries: $($script:ResolvedNpmRegistries -join ', ')"
    if ($Profile -eq "NvidiaCuda") {
        $findLinks = Get-TorchFindLinks
        if ($findLinks.Count -gt 0) {
            Write-Host "PyTorch CUDA find-links: $($findLinks -join ', ')"
        }
        Write-Host "PyTorch CUDA indexes: $((Get-TorchIndexUrls) -join ', ')"
    }
    if ($script:ResolvedHfEndpoint) {
        Write-Host "Hugging Face endpoint: $script:ResolvedHfEndpoint"
    }
}

function Assert-Windows {
    if (-not $IsWindows -and $env:OS -ne "Windows_NT") {
        throw "This script is for Windows only."
    }
}

function Normalize-CommandPath($Value) {
    if (-not $Value) {
        return ""
    }
    return $Value.Trim().Trim("'").Trim('"')
}

function Resolve-ProjectPath([string]$Path, [switch]$CreateDirectory) {
    if ([System.IO.Path]::IsPathRooted($Path)) {
        $resolved = $Path
    } else {
        $resolved = Join-Path (Resolve-Path ".").Path $Path
    }
    if ($CreateDirectory) {
        New-Item -ItemType Directory -Force -Path $resolved | Out-Null
    }
    return $resolved
}

function Test-PythonRuntimeArchive($Path) {
    if (-not (Test-Path -LiteralPath $Path)) {
        return $false
    }
    $item = Get-Item -LiteralPath $Path
    if ($item.Length -lt 5MB) {
        return $false
    }
    try {
        Add-Type -AssemblyName System.IO.Compression.FileSystem
        $zip = [System.IO.Compression.ZipFile]::OpenRead($Path)
        try {
            $names = @{}
            foreach ($entry in $zip.Entries) {
                $names[$entry.FullName.Replace("/", "\").ToLowerInvariant()] = $true
            }
            return (
                $names.ContainsKey("tools\python.exe") -and
                $names.ContainsKey("tools\lib\venv\__init__.py") -and
                $names.ContainsKey("tools\lib\ensurepip\__init__.py")
            )
        } finally {
            $zip.Dispose()
        }
    } catch {
        return $false
    }
}

function Get-PythonRuntimeDownloadUrls {
    $urls = New-Object System.Collections.ArrayList
    Add-ListValues $urls $PythonRuntimeUrls
    Add-ListValues $urls $env:PYTHON_RUNTIME_URLS
    Add-UniqueValue $urls "https://globalcdn.nuget.org/packages/$PythonRuntimePackageFile"
    return @($urls)
}

function Get-PythonRuntimeContentLength($Url) {
    try {
        $iwrArgs = @{
            Uri = $Url
            Method = "Head"
            UseBasicParsing = $true
            TimeoutSec = 30
        }
        $proxyValue = Normalize-CommandPath $Proxy
        if (-not $proxyValue) {
            $proxyValue = Normalize-CommandPath $env:HTTPS_PROXY
        }
        if (-not $proxyValue) {
            $proxyValue = Normalize-CommandPath $env:HTTP_PROXY
        }
        if ($proxyValue) {
            $iwrArgs.Proxy = $proxyValue
        }
        $response = Invoke-WebRequest @iwrArgs
        $length = $response.Headers["Content-Length"]
        if ($length) {
            return [int64]($length | Select-Object -First 1)
        }
    } catch {
    }
    return 0
}

function Invoke-PythonRuntimeDownload($Url, $Destination, [int64]$ExpectedLength) {
    $tempPath = "$Destination.part"
    $lastError = ""
    $curl = Get-Command "curl.exe" -ErrorAction SilentlyContinue
    $proxyValue = Normalize-CommandPath $Proxy
    if (-not $proxyValue) {
        $proxyValue = Normalize-CommandPath $env:HTTPS_PROXY
    }
    if (-not $proxyValue) {
        $proxyValue = Normalize-CommandPath $env:HTTP_PROXY
    }
    for ($attempt = 1; $attempt -le 4; $attempt++) {
        try {
            if (Test-PythonRuntimeArchive $Destination) {
                Write-Ok "Using cached $(Split-Path -Leaf $Destination)"
                return
            }
            if ($curl) {
                if (Test-Path -LiteralPath $tempPath) {
                    $partialLength = (Get-Item -LiteralPath $tempPath).Length
                    if ($ExpectedLength -gt 0 -and $partialLength -gt $ExpectedLength) {
                        Remove-Item -LiteralPath $tempPath -Force
                    } elseif ($partialLength -gt 0) {
                        Write-Host ("Resuming Python runtime download: {0:N1} MB" -f ($partialLength / 1MB))
                    }
                }
                Write-Host ("Downloading Python runtime with curl.exe (attempt {0}/4)" -f $attempt)
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
                if ($proxyValue) {
                    $curlArgs = @("--proxy", $proxyValue) + $curlArgs
                }
                Invoke-External $curl.Source $curlArgs "curl.exe Python runtime download failed"
            } else {
                if (Test-Path -LiteralPath $tempPath) {
                    Write-Warn "Invoke-WebRequest cannot resume the existing Python runtime partial file; restarting this small download."
                    Remove-Item -LiteralPath $tempPath -Force
                }
                Write-Host ("Downloading Python runtime with Invoke-WebRequest (attempt {0}/4)" -f $attempt)
                $iwrArgs = @{
                    Uri = $Url
                    OutFile = $tempPath
                    UseBasicParsing = $true
                    TimeoutSec = 1800
                }
                if ($proxyValue) {
                    $iwrArgs.Proxy = $proxyValue
                }
                Invoke-WebRequest @iwrArgs
            }
            if (-not (Test-Path -LiteralPath $tempPath)) {
                throw "download produced no file"
            }
            $actualLength = (Get-Item -LiteralPath $tempPath).Length
            if ($ExpectedLength -gt 0 -and $actualLength -ne $ExpectedLength) {
                throw "size mismatch: expected $ExpectedLength bytes, got $actualLength bytes"
            }
            Move-Item -LiteralPath $tempPath -Destination $Destination -Force
            if (-not (Test-PythonRuntimeArchive $Destination)) {
                throw "downloaded Python runtime archive failed validation"
            }
            return
        } catch {
            $lastError = $_.Exception.Message
            Write-Warn "Python runtime download failed: $lastError"
            Start-Sleep -Seconds ([Math]::Min(10, 2 * $attempt))
        }
    }
    if (Test-Path -LiteralPath $tempPath) {
        Write-Warn "Python runtime partial download kept for resume: $tempPath"
    }
    throw "Python runtime download failed after retries. Last error: $lastError"
}

function Save-PythonRuntimeAssetIfNeeded {
    $cacheDir = Resolve-ProjectPath ".download-cache\python" -CreateDirectory
    $target = Join-Path $cacheDir $PythonRuntimePackageFile
    if (Test-PythonRuntimeArchive $target) {
        Write-Ok "Using cached Python runtime asset: $target"
        return $target
    }
    if ($OfflineOnly) {
        return ""
    }
    $lastError = ""
    foreach ($url in (Get-PythonRuntimeDownloadUrls)) {
        try {
            Write-Host "Downloading project-local Python runtime asset: $url"
            $length = Get-PythonRuntimeContentLength $url
            Invoke-PythonRuntimeDownload $url $target $length
            return $target
        } catch {
            $lastError = $_.Exception.Message
            Write-Warn "Python runtime source failed: $url"
            Write-Warn $lastError
        }
    }
    throw "Unable to download Python runtime asset. Put python.3.12.x.nupkg under offline\python, or set PYTHON_RUNTIME_URLS to a reachable mirror. Last error: $lastError"
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

function Get-LocalPythonCandidatePaths {
    $paths = @()
    if ($PythonRuntimeDir) {
        $paths += (Join-Path (Resolve-ProjectPath $PythonRuntimeDir) "python.exe")
    }
    $paths += @(
        (Resolve-ProjectPath ".runtime\python\python.exe"),
        (Resolve-ProjectPath "tools\python312\python.exe"),
        (Resolve-ProjectPath "offline\python\python.exe")
    )
    return $paths
}

function Resolve-PythonRuntimeAssetPath {
    $candidate = Normalize-CommandPath $PythonInstaller
    if ($candidate) {
        if (-not (Test-PathUnderProjectRoot $candidate)) {
            throw "-PythonInstaller must point to a Python runtime asset inside this project directory. Host/global Python installers are not used."
        }
        return (Resolve-Path -LiteralPath $candidate -ErrorAction Stop).Path
    }

    $searchDirs = @(
        (Resolve-ProjectPath (Join-Path $OfflineAssetsDir "python")),
        (Resolve-ProjectPath ".download-cache\python")
    )
    Write-Host "Python runtime asset search dirs:"
    foreach ($dir in $searchDirs) {
        Write-Host "  $dir"
    }
    foreach ($dir in $searchDirs) {
        if (-not (Test-Path -LiteralPath $dir)) {
            Write-Warn "Python installer directory not found: $dir"
            continue
        }
        $assets = @(Get-ChildItem -LiteralPath $dir -File -Recurse -ErrorAction SilentlyContinue |
            Where-Object {
                $_.Name -match "(?i)^python\.3\.12.*\.nupkg$" -or
                $_.Name -match "(?i)^python-3\.12.*(amd64|x64).*\.(zip|nupkg)$"
            })
        if ($assets.Count -gt 0) {
            Write-Host "Python runtime asset candidates:"
            $assets | ForEach-Object { Write-Host "  $($_.FullName)" }
        }
        $asset = $assets |
            ForEach-Object {
                $rank = if ($_.Name -match "(?i)^python\.3\.12.*\.nupkg$") {
                    0
                } else {
                    1
                }
                [pscustomobject]@{ Item = $_; Rank = $rank }
            } |
            Sort-Object Rank, @{ Expression = { $_.Item.LastWriteTime }; Descending = $true } |
            Select-Object -First 1 -ExpandProperty Item
        if ($asset) {
            Write-Host "Using Python runtime asset: $($asset.FullName)"
            return $asset.FullName
        }
    }
    return Save-PythonRuntimeAssetIfNeeded
}

function Remove-ProjectSubtree($Path) {
    if (-not (Test-Path -LiteralPath $Path)) {
        return
    }
    $root = (Resolve-Path -LiteralPath ".").Path
    $target = (Resolve-Path -LiteralPath $Path -ErrorAction Stop).Path
    if ($target -eq $root -or -not $target.StartsWith($root, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove outside project: $target"
    }
    Remove-Item -LiteralPath $target -Recurse -Force
}

function Expand-PortablePythonRuntime($ArchivePath, $TargetDir) {
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $parent = Split-Path -Parent $TargetDir
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
    $extractDir = Join-Path $parent ("python-extract-{0}" -f ([System.Guid]::NewGuid().ToString("N")))
    Remove-ProjectSubtree $TargetDir
    New-Item -ItemType Directory -Force -Path $extractDir | Out-Null
    New-Item -ItemType Directory -Force -Path $TargetDir | Out-Null
    try {
        [System.IO.Compression.ZipFile]::ExtractToDirectory($ArchivePath, $extractDir)
        $sourceDir = Join-Path $extractDir "tools"
        if (-not (Test-Path -LiteralPath (Join-Path $sourceDir "python.exe"))) {
            $sourceDir = $extractDir
        }
        if (-not (Test-Path -LiteralPath (Join-Path $sourceDir "python.exe"))) {
            throw "Portable Python archive does not contain python.exe at root or tools\python.exe: $ArchivePath"
        }
        Get-ChildItem -LiteralPath $sourceDir -Force | Copy-Item -Destination $TargetDir -Recurse -Force
    } finally {
        Remove-ProjectSubtree $extractDir
    }
}

function Install-LocalPythonRuntimeIfAvailable {
    foreach ($candidate in (Get-LocalPythonCandidatePaths)) {
        $python = Test-Python312Candidate $candidate
        if ($python) {
            return $python
        }
    }

    $asset = Resolve-PythonRuntimeAssetPath
    if (-not $asset) {
        return ""
    }

    $targetDir = Resolve-ProjectPath $PythonRuntimeDir
    $extension = [System.IO.Path]::GetExtension($asset).ToLowerInvariant()
    Write-Host "Preparing project-local Python 3.12 runtime:"
    Write-Host "  asset:  $asset"
    Write-Host "  target:    $targetDir"
    if ($extension -eq ".nupkg" -or $extension -eq ".zip") {
        Expand-PortablePythonRuntime $asset $targetDir
    } else {
        Remove-ProjectSubtree $targetDir
        New-Item -ItemType Directory -Force -Path $targetDir | Out-Null
        Invoke-External $asset @(
            "/quiet",
            "InstallAllUsers=0",
            "TargetDir=$targetDir",
            "PrependPath=0",
            "Include_pip=1",
            "Include_launcher=0",
            "Include_tcltk=0",
            "Include_test=0",
            "Shortcuts=0",
            "AssociateFiles=0"
        ) "Project-local Python installation failed"
    }

    $runtimePython = Join-Path $targetDir "python.exe"
    $python = Test-Python312Candidate $runtimePython
    if ($python) {
        return $python
    }
    throw "Project-local Python runtime asset was applied but Python 3.12 was not usable at $runtimePython. Prefer offline\python\python.3.12.x.nupkg over the Windows .exe installer."
}

function Test-PathUnderProjectRoot($Path) {
    if (-not $Path) {
        return $false
    }
    try {
        $root = (Resolve-Path -LiteralPath ".").Path
        $resolved = (Resolve-Path -LiteralPath $Path -ErrorAction Stop).Path
        return $resolved.StartsWith($root, [System.StringComparison]::OrdinalIgnoreCase)
    } catch {
        return $false
    }
}

function Find-Python312($Preferred) {
    $candidates = @()
    if ($Preferred) {
        if (-not (Test-PathUnderProjectRoot $Preferred)) {
            throw "-PythonExe is restricted to this project directory. Host Python is disabled. Put python.3.12.x.nupkg under offline\python, or keep Python under .runtime\python-3.12."
        }
        $candidates += $Preferred
    }
    $localPython = Install-LocalPythonRuntimeIfAvailable
    if ($localPython) {
        $candidates += $localPython
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
    throw "Project-local Python 3.12 runtime was not found. Put python.3.12.x.nupkg under offline\python, or keep an extracted runtime under .runtime\python-3.12. Host Python is intentionally not used."
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

function Create-Or-RecreateVenv($PythonExe, $Path, $ForceRecreate) {
    $venvPythonPath = Join-Path $Path "Scripts\python.exe"
    if ((Test-Path -LiteralPath $Path) -and ($ForceRecreate -or -not (Test-Path -LiteralPath $venvPythonPath))) {
        Write-Warn "Existing virtual environment is unusable or recreation was requested. Recreating: $Path"
        Remove-ProjectChildDirectory $Path
    }
    if (-not (Test-Path -LiteralPath $Path)) {
        Invoke-External $PythonExe @("-m", "venv", $Path) "Failed to create virtual environment"
    }
    return (Resolve-Path -LiteralPath $venvPythonPath -ErrorAction Stop).Path
}

function Ensure-Ffmpeg {
    if (Get-Command "ffmpeg" -ErrorAction SilentlyContinue) {
        Write-Ok "FFmpeg detected."
        return
    }
    if ($InstallFfmpeg) {
        $winget = Get-Command "winget" -ErrorAction SilentlyContinue
        if (-not $winget) {
            throw "FFmpeg was not found and winget is unavailable. Install FFmpeg manually and make sure ffmpeg.exe is in PATH."
        }
        Write-Warn "Installing FFmpeg with winget."
        Invoke-External $winget.Source @(
            "install", "-e", "--id", "Gyan.FFmpeg",
            "--source", "winget",
            "--accept-package-agreements",
            "--accept-source-agreements"
        ) "FFmpeg installation failed"
        return
    }
    Write-Warn "FFmpeg was not found. WAV/FLAC may still work, but MP3/M4A decoding can fail. Install FFmpeg or rerun with -InstallFfmpeg."
}

function Assert-NvidiaGraphics($Skip) {
    if ($Skip) {
        Write-Warn "NVIDIA GPU check skipped."
        return
    }
    $nvidiaSmi = Get-Command "nvidia-smi" -ErrorAction SilentlyContinue
    if (-not $nvidiaSmi) {
        throw "nvidia-smi was not found. Install an NVIDIA driver first, or rerun with -SkipGpuCheck if this is an offline package preparation machine."
    }
    try {
        $names = @(& $nvidiaSmi.Source --query-gpu=name --format=csv,noheader 2>$null)
        if ($LASTEXITCODE -ne 0 -or -not $names) {
            throw "nvidia-smi did not report any GPU."
        }
        Write-Ok "NVIDIA GPU: $($names -join ', ')"
    } catch {
        throw "NVIDIA GPU check failed: $($_.Exception.Message)"
    }
}

function Resolve-NvidiaAria2cPath {
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
        Write-Warn "Installing aria2 with winget for faster resumable NVIDIA wheel downloads."
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

function Initialize-NvidiaWheelCache {
    $script:ResolvedNvidiaWheelCache = Resolve-ProjectPath (Join-Path $NvidiaWheelCache $CudaWheel) -CreateDirectory
    $script:ResolvedNvidiaAria2cPath = Resolve-NvidiaAria2cPath
    if ($script:ResolvedNvidiaAria2cPath) {
        Write-Host "NVIDIA wheel downloader: aria2c ($script:ResolvedNvidiaAria2cPath), connections=$Aria2Connections"
    } elseif (Get-Command "curl.exe" -ErrorAction SilentlyContinue) {
        Write-Host "NVIDIA wheel downloader: curl.exe"
    } else {
        Write-Host "NVIDIA wheel downloader: Invoke-WebRequest"
    }
    Write-Host "NVIDIA wheel cache: $script:ResolvedNvidiaWheelCache"
}

function Invoke-PipInstallWithMirrors($Python, [string[]]$InstallArgs, $Description) {
    $lastError = ""
    $offlineWheelArgs = @()
    $offlineWheels = if ([System.IO.Path]::IsPathRooted((Join-Path $OfflineAssetsDir "wheels"))) {
        Join-Path $OfflineAssetsDir "wheels"
    } else {
        Join-Path (Resolve-Path ".").Path (Join-Path $OfflineAssetsDir "wheels")
    }
    if (Test-Path -LiteralPath $offlineWheels) {
        $offlineWheelArgs = @("--find-links", $offlineWheels)
        Write-Host "offline wheels: $offlineWheels"
    }
    foreach ($indexUrl in $script:ResolvedPipIndexUrls) {
        try {
            Write-Host "pip index: $indexUrl"
            $args = @("-m", "pip") + $InstallArgs + $offlineWheelArgs + @(
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

function Get-TorchIndexUrls {
    $urls = New-Object System.Collections.ArrayList
    Add-ListValues $urls $TorchIndexUrls
    if ($urls.Count -eq 0) {
        if ($MirrorMode -eq "Official") {
            Add-UniqueValue $urls "$OfficialTorchIndexBaseUrl/$CudaWheel"
        } elseif ($MirrorMode -eq "Auto" -and $env:TORCH_INDEX_URLS) {
            Add-ListValues $urls $env:TORCH_INDEX_URLS
            Add-UniqueValue $urls "$OfficialTorchIndexBaseUrl/$CudaWheel"
        } else {
            Add-UniqueValue $urls "$OfficialTorchIndexBaseUrl/$CudaWheel"
        }
    } elseif ($MirrorMode -ne "Official") {
        Add-UniqueValue $urls "$OfficialTorchIndexBaseUrl/$CudaWheel"
    }
    return @($urls)
}

function Get-TorchFindLinks {
    $urls = New-Object System.Collections.ArrayList
    Add-ListValues $urls $TorchFindLinks
    if ($urls.Count -eq 0) {
        if ($MirrorMode -eq "Auto" -and $env:TORCH_FIND_LINKS) {
            Add-ListValues $urls $env:TORCH_FIND_LINKS
        } elseif ($MirrorMode -ne "Official") {
            foreach ($baseUrl in $ChinaTorchFindLinkBaseUrls) {
                Add-UniqueValue $urls "$($baseUrl.TrimEnd('/'))/$CudaWheel"
            }
        }
    }
    return @($urls)
}

function Get-PrimaryPipIndexUrl {
    if ($script:ResolvedPipIndexUrls -and $script:ResolvedPipIndexUrls.Count -gt 0) {
        return [string]$script:ResolvedPipIndexUrls[0]
    }
    return $OfficialPipIndexUrl
}

function Get-ExtraPipIndexArgs {
    $args = @()
    if (-not $script:ResolvedPipIndexUrls) {
        return $args
    }
    for ($i = 1; $i -lt $script:ResolvedPipIndexUrls.Count; $i++) {
        $args += @("--extra-index-url", [string]$script:ResolvedPipIndexUrls[$i])
    }
    return $args
}

function Get-CudaTorchBuildDefinition($Name) {
    if ($Name -eq "latest") {
        return [ordered]@{ Name = "latest"; Torch = "2.11.0"; TorchAudio = "2.11.0"; TorchVision = "0.26.0" }
    }
    if ($CudaWheel -eq "cu130") {
        return [ordered]@{ Name = "stable"; Torch = "2.9.0"; TorchAudio = "2.9.0"; TorchVision = "0.24.0" }
    }
    return [ordered]@{ Name = "stable"; Torch = "2.8.0"; TorchAudio = "2.8.0"; TorchVision = "0.23.0" }
}

function Get-CudaTorchArtifacts($Build) {
    return @(
        [ordered]@{
            Package = "torch"
            Version = $Build.Torch
            File = "torch-$($Build.Torch)+$CudaWheel-cp312-cp312-win_amd64.whl"
        },
        [ordered]@{
            Package = "torchaudio"
            Version = $Build.TorchAudio
            File = "torchaudio-$($Build.TorchAudio)+$CudaWheel-cp312-cp312-win_amd64.whl"
        },
        [ordered]@{
            Package = "torchvision"
            Version = $Build.TorchVision
            File = "torchvision-$($Build.TorchVision)+$CudaWheel-cp312-cp312-win_amd64.whl"
        }
    )
}

function Get-CudaTorchBuildCandidates {
    if ($TorchBuild -eq "auto") {
        return @((Get-CudaTorchBuildDefinition "latest"), (Get-CudaTorchBuildDefinition "stable"))
    }
    return @((Get-CudaTorchBuildDefinition $TorchBuild))
}

function Get-CudaTorchDownloadBaseUrls {
    $urls = New-Object System.Collections.ArrayList
    Add-ListValues $urls (Get-TorchFindLinks)
    Add-ListValues $urls (Get-TorchIndexUrls)
    return @($urls)
}

function Write-CudaTorchSources {
    Write-Host "PyTorch CUDA wheel sources:"
    foreach ($baseUrl in (Get-CudaTorchDownloadBaseUrls)) {
        Write-Host "  $(([string]$baseUrl).TrimEnd('/'))"
    }
}

function Get-RemoteWheelProbe($Url) {
    try {
        $response = Invoke-WebRequest -Uri $Url -Method Head -UseBasicParsing -TimeoutSec 30
        $contentType = [string]($response.Headers["Content-Type"] | Select-Object -First 1)
        if ($contentType -match "text/html") {
            return @{ Ok = $false; Length = 0; Reason = "remote returned HTML, not a wheel" }
        }
        $lengthHeader = $response.Headers["Content-Length"]
        $length = if ($lengthHeader) { [int64]($lengthHeader | Select-Object -First 1) } else { 0L }
        if ($length -gt 0 -and $length -lt 100KB) {
            return @{ Ok = $false; Length = $length; Reason = "remote wheel is suspiciously small ($length bytes)" }
        }
        return @{ Ok = $true; Length = $length; Reason = "" }
    } catch {
        return @{ Ok = $true; Length = 0L; Reason = "" }
    }
}

function Test-CachedCudaWheel($Path, [int64]$ExpectedLength) {
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
    return $true
}

function Invoke-CudaWheelDownload($Url, $Destination, [int64]$ExpectedLength) {
    $tempPath = "$Destination.part"
    $methods = @()
    if ($script:ResolvedNvidiaAria2cPath) {
        $methods += @{ Name = "aria2c"; Kind = "aria2"; Path = $script:ResolvedNvidiaAria2cPath }
    }
    $curl = Get-Command "curl.exe" -ErrorAction SilentlyContinue
    if ($curl) {
        $methods += @{ Name = "curl.exe"; Kind = "curl"; Path = $curl.Source }
    }
    $methods += @{ Name = "Invoke-WebRequest"; Kind = "iwr"; Path = "" }

    $lastError = ""
    for ($attempt = 1; $attempt -le 4; $attempt++) {
        foreach ($method in $methods) {
            try {
                if (Test-CachedCudaWheel $Destination $ExpectedLength) {
                    Write-Ok "Using cached $(Split-Path -Leaf $Destination)"
                    return
                }
                if (Test-Path -LiteralPath $tempPath) {
                    $partialLength = (Get-Item -LiteralPath $tempPath).Length
                    if ($ExpectedLength -gt 0 -and $partialLength -gt $ExpectedLength) {
                        Write-Warn "Partial CUDA wheel is larger than remote size. Restarting this artifact download."
                        Remove-Item -LiteralPath $tempPath -Force
                    } elseif ($method.Kind -eq "iwr") {
                        Write-Warn "Keeping partial CUDA wheel for a resumable downloader; skipping Invoke-WebRequest."
                        continue
                    } elseif ($partialLength -gt 0) {
                        Write-Host ("Resuming partial CUDA wheel: {0:N1} MB" -f ($partialLength / 1MB))
                    }
                }

                Write-Host ("Downloading CUDA wheel with {0} (attempt {1}/4)" -f $method.Name, $attempt)
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
                        "--out", (Split-Path -Leaf $tempPath),
                        $Url
                    )
                    Invoke-External $method.Path $ariaArgs "aria2c CUDA wheel download failed"
                } elseif ($method.Kind -eq "curl") {
                    Invoke-External $method.Path @(
                        "--fail",
                        "--location",
                        "--retry", "8",
                        "--retry-delay", "2",
                        "--connect-timeout", "30",
                        "--continue-at", "-",
                        "--output", $tempPath,
                        $Url
                    ) "curl.exe CUDA wheel download failed"
                } else {
                    Invoke-WebRequest -Uri $Url -OutFile $tempPath -UseBasicParsing -TimeoutSec 7200
                }

                if (-not (Test-Path -LiteralPath $tempPath)) {
                    throw "download produced no file"
                }
                $actualLength = (Get-Item -LiteralPath $tempPath).Length
                if ($actualLength -le 0) {
                    throw "downloaded wheel is empty"
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
        Write-Warn "Partial CUDA wheel kept for resume: $tempPath"
    }
    throw "Download failed after retries: $Url. Rerun the installer to resume from the .part file. Last error: $lastError"
}

function Save-CudaTorchWheelIfNeeded($Artifact, $CacheDir) {
    New-Item -ItemType Directory -Force -Path $CacheDir | Out-Null
    $target = Join-Path $CacheDir $Artifact.File
    $partial = "$target.part"
    $encodedFile = $Artifact.File.Replace("+", "%2B")
    $sourceUrls = @()
    foreach ($baseUrl in (Get-CudaTorchDownloadBaseUrls)) {
        $sourceUrls += "$(([string]$baseUrl).TrimEnd('/'))/$encodedFile"
    }
    Write-Host "CUDA wheel candidates for $($Artifact.File):"
    foreach ($url in $sourceUrls) {
        Write-Host "  $url"
    }

    $remoteLength = 0L
    foreach ($url in $sourceUrls) {
        $probe = Get-RemoteWheelProbe $url
        if ($probe.Ok -and $probe.Length -gt 0) {
            $remoteLength = [int64]$probe.Length
            break
        }
    }

    if (Test-CachedCudaWheel $target $remoteLength) {
        Write-Ok "Using cached $($Artifact.File)"
        return $target
    }
    if (Test-Path -LiteralPath $target) {
        $localLength = (Get-Item -LiteralPath $target).Length
        if ($remoteLength -gt 0 -and $localLength -gt 0 -and $localLength -lt $remoteLength) {
            Write-Warn "Cached CUDA wheel is incomplete; moving it to .part for resume: $($Artifact.File)"
            if ((Test-Path -LiteralPath $partial) -and (Get-Item -LiteralPath $partial).Length -ge $localLength) {
                Remove-Item -LiteralPath $target -Force
            } else {
                Move-Item -LiteralPath $target -Destination $partial -Force
            }
        } else {
            Write-Warn "Cached CUDA wheel is invalid or size changed: $($Artifact.File). Restarting this artifact download."
            Remove-Item -LiteralPath $target -Force
        }
    }
    if ($remoteLength -gt 0 -and (Test-CachedCudaWheel $partial $remoteLength)) {
        Write-Ok "Promoting completed partial CUDA wheel: $($Artifact.File)"
        Move-Item -LiteralPath $partial -Destination $target -Force
        return $target
    }

    if ($OfflineOnly) {
        throw "Offline CUDA wheel missing or incomplete: $target"
    }

    $lastError = ""
    foreach ($url in $sourceUrls) {
        try {
            $probe = Get-RemoteWheelProbe $url
            if (-not $probe.Ok) {
                throw $probe.Reason
            }
            $length = [int64]$probe.Length
            Write-Host "Downloading $($Artifact.File)"
            Write-Host "Source: $url"
            Invoke-CudaWheelDownload $url $target $length
            if (-not (Test-CachedCudaWheel $target $length)) {
                throw "downloaded CUDA wheel failed validation"
            }
            return $target
        } catch {
            $lastError = $_.Exception.Message
            Write-Warn "CUDA wheel source failed: $url"
            Write-Warn $lastError
        }
    }
    throw "Download failed for $($Artifact.File). Last error: $lastError"
}

function Save-CudaTorchWheels($Build) {
    if (-not $script:ResolvedNvidiaWheelCache) {
        Initialize-NvidiaWheelCache
    }
    $paths = @()
    foreach ($artifact in (Get-CudaTorchArtifacts $Build)) {
        $paths += (Save-CudaTorchWheelIfNeeded $artifact $script:ResolvedNvidiaWheelCache)
    }
    return @($paths)
}

function Test-CudaTorchVersionText($Version, $Build) {
    if ($Version -notmatch "\+$CudaWheel") {
        return $false
    }
    if ($Build) {
        return $Version.StartsWith("$($Build.Torch)+", [System.StringComparison]::OrdinalIgnoreCase)
    }
    return $true
}

function Test-CudaTorchReady($Python, $Build = $null) {
    try {
        $code = @"
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS', 'GOTO_NUM_THREADS', 'OPENBLAS_MAIN_FREE'):
    os.environ.setdefault(key, '1')
import torch
print(torch.__version__)
print(torch.version.cuda or '')
print(torch.cuda.is_available())
"@
        $lines = @(& $Python -c $code 2>$null)
        if ($LASTEXITCODE -ne 0 -or $lines.Count -lt 3) {
            return $false
        }
        $version = [string]$lines[0]
        $cuda = [string]$lines[1]
        $available = [string]$lines[2]
        return ((Test-CudaTorchVersionText $version $Build) -and $cuda -and $available.Trim().ToLowerInvariant() -eq "true")
    } catch {
        return $false
    }
}

function Test-CudaTorchInstalled($Python, $Build = $null) {
    try {
        $code = @"
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS', 'GOTO_NUM_THREADS', 'OPENBLAS_MAIN_FREE'):
    os.environ.setdefault(key, '1')
import torch
print(torch.__version__)
print(torch.version.cuda or '')
"@
        $lines = @(& $Python -c $code 2>$null)
        if ($LASTEXITCODE -ne 0 -or $lines.Count -lt 2) {
            return $false
        }
        $version = [string]$lines[0]
        $cuda = [string]$lines[1]
        return ((Test-CudaTorchVersionText $version $Build) -and $cuda)
    } catch {
        return $false
    }
}

function Write-CudaTorchProbeDetails($Python) {
    try {
        $code = @"
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS', 'GOTO_NUM_THREADS', 'OPENBLAS_MAIN_FREE'):
    os.environ.setdefault(key, '1')
import torch
print('torch=', torch.__version__)
print('cuda_runtime=', torch.version.cuda)
print('gpu_available=', torch.cuda.is_available())
print('device=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')
"@
        $lines = @(& $Python -c $code 2>&1)
        foreach ($line in $lines) {
            Write-Warn "[TORCH] $line"
        }
    } catch {
        Write-Warn "[TORCH] $($_.Exception.Message)"
    }
}

function Install-CudaTorchRuntimeDeps($Python) {
    Invoke-PipInstallWithMirrors $Python @(
        "install",
        "filelock>=3.13.0",
        "typing-extensions>=4.10.0",
        "numpy>=1.24.0,<3.0.0",
        "sympy>=1.13.3,<2.0.0",
        "networkx>=3.0,<4.0",
        "jinja2>=3.1.0,<4.0.0",
        "fsspec>=2024.0.0",
        "setuptools",
        "--prefer-binary"
    ) "Failed to install PyTorch runtime dependencies"
}

function Install-CudaTorch($Python) {
    $requestedBuild = if ($TorchBuild -eq "auto") { $null } else { Get-CudaTorchBuildDefinition $TorchBuild }
    if (-not $ForceTorch) {
        if ($SkipGpuCheck -and (Test-CudaTorchInstalled $Python $requestedBuild)) {
            Write-Ok "NVIDIA CUDA PyTorch already installed; skipping."
            return
        }
        if ((-not $SkipGpuCheck) -and (Test-CudaTorchReady $Python $requestedBuild)) {
            Write-Ok "NVIDIA CUDA PyTorch already installed and GPU is available; skipping."
            return
        }
    }
    $lastError = ""
    foreach ($build in (Get-CudaTorchBuildCandidates)) {
        try {
            $pipIndexUrl = Get-PrimaryPipIndexUrl
            Write-Host "PyTorch CUDA build: $($build.Name) (torch=$($build.Torch), torchaudio=$($build.TorchAudio), torchvision=$($build.TorchVision))"
            Write-CudaTorchSources
            $wheelPaths = Save-CudaTorchWheels $build
            Write-Host "PyTorch CUDA local wheels:"
            foreach ($wheelPath in $wheelPaths) {
                Write-Host "  $wheelPath"
            }
            Write-Host "Wheel platform note: win_amd64 means Windows x64 CPU/Python ABI, not AMD GPU."
            Write-Host "PyTorch dependency index: $pipIndexUrl"
            Install-CudaTorchRuntimeDeps $Python
            $installArgs = @(
                "-m",
                "pip",
                "install",
                "--no-index",
                "--no-deps"
            ) + $wheelPaths
            if ($ForceTorch) {
                $installArgs += "--force-reinstall"
            }
            Invoke-External $Python $installArgs "Failed to install NVIDIA CUDA PyTorch"
            $ready = if ($SkipGpuCheck) { Test-CudaTorchInstalled $Python $build } else { Test-CudaTorchReady $Python $build }
            if ($ready) {
                Write-Ok "NVIDIA CUDA PyTorch build '$($build.Name)' is usable."
                return
            }
            Write-CudaTorchProbeDetails $Python
            throw "PyTorch CUDA build '$($build.Name)' installed but torch import/GPU verification failed. This usually means an NVIDIA driver, Windows DLL, or PyTorch runtime compatibility issue. Try updating the NVIDIA driver or rerun with -CudaWheel cu126."
        } catch {
            $lastError = $_.Exception.Message
            Write-Warn "PyTorch CUDA build $($build.Name) failed: $lastError"
        }
    }
    throw "Failed to install NVIDIA CUDA PyTorch with all candidate builds. Last error: $lastError"
}

function Verify-CudaPyTorch($Python, [bool]$RequireGpu) {
    Set-NumericRuntimeThreadEnv
    $require = if ($RequireGpu) { "1" } else { "0" }
    $code = @"
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS', 'GOTO_NUM_THREADS', 'OPENBLAS_MAIN_FREE'):
    os.environ.setdefault(key, '1')
import torch
print('torch=', torch.__version__)
print('cuda_runtime=', torch.version.cuda)
print('gpu_available=', torch.cuda.is_available())
print('device=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')
require_gpu = '$require' == '1'
ok = bool(torch.version.cuda) and (torch.cuda.is_available() or not require_gpu)
raise SystemExit(0 if ok else 1)
"@
    Invoke-External $Python @("-c", $code) "NVIDIA CUDA PyTorch verification failed"
}

function Test-ProjectDepsReady($Python) {
    try {
        $code = @"
import importlib.metadata as md
import importlib.util

missing = [
    m
    for m in ('uvicorn', 'fastapi', 'qwen_asr', 'funasr', 'modelscope', 'librosa', 'imageio_ffmpeg', 'soundfile')
    if importlib.util.find_spec(m) is None
]

def version(dist):
    try:
        return md.version(dist).split('+', 1)[0]
    except Exception:
        return None

for dist, expected in {
    'funasr': '$StableFunasrVersion',
    'qwen-asr': '$StableQwenAsrVersion',
}.items():
    current = version(dist)
    if current != expected:
        missing.append(f'{dist}=={expected}(current={current})')

print(','.join(missing))
raise SystemExit(1 if missing else 0)
"@
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

function Install-ProjectDeps($Python, [bool]$UseMemoryVectorStore, [bool]$SkipTorchPackages, [bool]$SkipFacePackages) {
    if (-not $ForceDeps -and (Test-ProjectDepsReady $Python)) {
        Write-Ok "Project Python dependencies already installed; skipping."
        return
    }
    $tempRequirements = Join-Path $env:TEMP "matrix-live-diarizer-windows-requirements.txt"
    $skipNames = @()
    if ($UseMemoryVectorStore) {
        $skipNames += @("chromadb", "chroma-hnswlib")
    }
    if (-not $InstallPyannote) {
        $skipNames += @("pyannote.audio")
    }
    if ($SkipTorchPackages) {
        $skipNames += @("torch", "torchaudio", "torchvision")
    }
    if ($SkipFacePackages) {
        $skipNames += @(
            "insightface",
            "onnxruntime",
            "onnxruntime-gpu",
            "onnxruntime-directml",
            "onnx",
            "opencv-python",
            "opencv-python-headless",
            "Pillow",
            "scikit-image"
        )
    }
    $skipPattern = if ($skipNames.Count) {
        "^\s*($([string]::Join('|', ($skipNames | ForEach-Object { [regex]::Escape($_) }))))([=<>!~ ;]|$)"
    } else {
        "a^"
    }
    Get-Content -LiteralPath "requirements.txt" -Encoding UTF8 |
        Where-Object { $_ -notmatch $skipPattern } |
        Set-Content -LiteralPath $tempRequirements -Encoding UTF8

    Invoke-PipInstallWithMirrors $Python @("install", "-U", "pip", "wheel", "setuptools") "Failed to update pip tooling"
    Invoke-PipInstallWithMirrors $Python @("install", "-r", $tempRequirements) "Failed to install project dependencies"
    & $Python -m pip check
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "pip check reported dependency conflicts; continuing because optional model packages may carry loose constraints."
    }
}

function Test-FaceRuntimeReady($Python, [string]$OrtFlavor) {
    try {
        $code = @"
import importlib.metadata as md
import importlib.util
missing=[m for m in ('insightface','onnxruntime','cv2') if importlib.util.find_spec(m) is None]
if missing:
    print('missing=' + ','.join(missing))
    raise SystemExit(1)
import cv2
import insightface
from insightface.app import FaceAnalysis
import onnxruntime as ort
providers=ort.get_available_providers()
print('onnxruntime_providers=' + ','.join(providers))
print('opencv=' + getattr(cv2, '__version__', 'unknown'))
print('insightface=' + getattr(insightface, '__version__', 'unknown'))
def version(dist):
    try:
        return md.version(dist).split('+', 1)[0]
    except Exception:
        return None
expected = {
    'insightface': '$StableInsightFaceVersion',
    'onnx': '$StableOnnxVersion',
    'Pillow': '$StablePillowVersion',
    'scikit-image': '$StableScikitImageVersion',
}
ort_dist = {
    'cpu': 'onnxruntime',
    'cuda': 'onnxruntime-gpu',
    'directml': 'onnxruntime-directml',
}.get('$OrtFlavor', 'onnxruntime')
ort_expected = {
    'cpu': '$StableOnnxRuntimeCpuVersion',
    'cuda': '$StableOnnxRuntimeGpuVersion',
    'directml': '$StableOnnxRuntimeDirectmlVersion',
}.get('$OrtFlavor', '$StableOnnxRuntimeCpuVersion')
expected[ort_dist] = ort_expected
mismatched = []
for dist, wanted in expected.items():
    current = version(dist)
    if current != wanted:
        mismatched.append(f'{dist}=={wanted}(current={current})')
opencv_current = version('opencv-python-headless') or version('opencv-python')
if opencv_current != '$StableOpenCvVersion':
    mismatched.append(f'opencv-python-headless/opencv-python==$StableOpenCvVersion(current={opencv_current})')
if mismatched:
    print('version_mismatch=' + ','.join(mismatched))
    raise SystemExit(3)
required = {
    'cpu': 'CPUExecutionProvider',
    'cuda': 'CUDAExecutionProvider',
    'directml': 'DmlExecutionProvider',
}.get('$OrtFlavor', 'CPUExecutionProvider')
if required not in providers:
    raise SystemExit(2)
"@
        & $Python -c $code
        return ($LASTEXITCODE -eq 0)
    } catch {
    return $false
    }
}

function Test-Cv2Ready($Python) {
    $oldErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $lines = @(& $Python -c "import cv2; print('opencv=' + getattr(cv2, '__version__', 'unknown'))" 2>$null)
        $exitCode = $LASTEXITCODE
        if ($exitCode -eq 0) {
            foreach ($line in $lines) {
                Write-Host $line
            }
            return $true
        }
        return $false
    } catch {
        return $false
    } finally {
        $ErrorActionPreference = $oldErrorActionPreference
    }
}

function Install-OpenCvForFace($Python) {
    if (Test-Cv2Ready $Python) {
        Write-Ok "OpenCV cv2 import is already available."
        return
    }
    Write-Host "Cleaning existing OpenCV packages before installing cv2 runtime."
    [void](Invoke-NativeQuietExitCode $Python @("-m", "pip", "uninstall", "-y", "opencv-python", "opencv-python-headless"))
    $lastError = ""
    foreach ($package in @("opencv-python-headless==$StableOpenCvVersion", "opencv-python==$StableOpenCvVersion")) {
        try {
            Invoke-PipInstallWithMirrors $Python @(
                "install",
                $package,
                "--prefer-binary"
            ) "Failed to install $package"
            if (Test-Cv2Ready $Python) {
                return
            }
            $lastError = "$package installed but import cv2 still failed"
            Write-Warn $lastError
        } catch {
            $lastError = $_.Exception.Message
            Write-Warn "OpenCV package failed: $package"
            Write-Warn $lastError
        }
    }
    throw "Failed to install an OpenCV package that provides cv2. Last error: $lastError"
}

function Install-FaceDeps($Python, [ValidateSet("cpu", "cuda", "directml")] [string]$OrtFlavor) {
    if (-not $ForceDeps -and (Test-FaceRuntimeReady $Python $OrtFlavor)) {
        Write-Ok "Face recognition runtime already installed; skipping."
        return
    }
    $ortPackage = if ($OrtFlavor -eq "cuda") {
        "onnxruntime-gpu==$StableOnnxRuntimeGpuVersion"
    } elseif ($OrtFlavor -eq "directml") {
        "onnxruntime-directml==$StableOnnxRuntimeDirectmlVersion"
    } else {
        "onnxruntime==$StableOnnxRuntimeCpuVersion"
    }
    Invoke-PipInstallWithMirrors $Python @(
        "install",
        $ortPackage,
        "numpy>=1.24.0,<3.0.0",
        "scipy>=1.10.0,<2.0.0",
        "onnx==$StableOnnxVersion",
        "Pillow==$StablePillowVersion",
        "scikit-image==$StableScikitImageVersion",
        "tqdm>=4.66.0,<5.0.0",
        "requests>=2.31.0,<3.0.0",
        "--prefer-binary"
    ) "Failed to install ONNXRuntime face dependencies"
    Install-OpenCvForFace $Python
    Invoke-PipInstallWithMirrors $Python @(
        "install",
        "insightface==$StableInsightFaceVersion",
        "--no-deps",
        "--prefer-binary"
    ) "Failed to install InsightFace"
    if (-not (Test-FaceRuntimeReady $Python $OrtFlavor)) {
        throw "Face recognition runtime verification failed. Check that insightface, onnxruntime, and cv2 are importable in this virtual environment."
    }
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

function Test-EnvFileKey($Path, $Key) {
    if (-not (Test-Path -LiteralPath $Path)) {
        return $false
    }
    foreach ($line in (Get-Content -LiteralPath $Path -Encoding UTF8)) {
        if ($line -match "^\s*$([regex]::Escape($Key))\s*=") {
            return $true
        }
    }
    return $false
}

function Set-EnvFileValueIfMissing($Path, $Key, $Value) {
    if (-not (Test-EnvFileKey $Path $Key)) {
        Set-EnvFileValue $Path $Key $Value
    }
}

function Repair-EnvSecretPlaceholders($Path) {
    if (-not (Test-Path -LiteralPath $Path)) {
        return
    }
    $secretKeys = @(
        "LLM_API_KEY",
        "ASR_API_KEY",
        "SPEAKER_API_KEY",
        "DIARIZATION_API_KEY",
        "FACE_API_KEY",
        "HF_TOKEN",
        "HUGGINGFACE_TOKEN",
        "HUGGINGFACE_HUB_TOKEN",
        "MODELSCOPE_API_TOKEN",
        "MODELSCOPE_TOKEN"
    )
    $lines = Get-Content -LiteralPath $Path -Encoding UTF8
    $changed = $false
    $out = New-Object System.Collections.ArrayList
    foreach ($line in $lines) {
        $replaced = $false
        foreach ($key in $secretKeys) {
            $pattern = "^\s*" + [regex]::Escape($key) + "\s*=(.*)$"
            if ($line -match $pattern) {
                $value = [string]$Matches[1]
                $trimmed = $value.Trim()
                $firstCharCode = if ($trimmed.Length -gt 0) { [int][char]$trimmed[0] } else { -1 }
                if ($trimmed.StartsWith("#") -or $firstCharCode -eq 0xFF03) {
                    $cleanedSecretLine = $key + "="
                    [void]$out.Add($cleanedSecretLine)
                    $changed = $true
                } else {
                    [void]$out.Add($line)
                }
                $replaced = $true
                break
            }
        }
        if (-not $replaced) {
            [void]$out.Add($line)
        }
    }
    if ($changed) {
        Set-Content -LiteralPath $Path -Value $out -Encoding UTF8
        Write-Ok "Cleaned placeholder API key values in $Path"
    }
}

function Ensure-EnvConfig([string]$Device = "cpu", [string]$AsrEngine = "sensevoice_zh", [string]$FaceDevice = "") {
    if (-not $FaceDevice) {
        $FaceDevice = $Device
    }
    if (-not (Test-Path ".env")) {
        Copy-Item -LiteralPath ".env.example" -Destination ".env"
    }
    Repair-EnvSecretPlaceholders ".env"
    Set-EnvFileValueIfMissing ".env" "PORT" "8321"
    Set-EnvFileValue ".env" "MODELS_DIR" "./models"
    Set-EnvFileValue ".env" "HF_HOME" "./models/huggingface"
    Set-EnvFileValue ".env" "HF_HUB_CACHE" "./models/huggingface/hub"
    Set-EnvFileValue ".env" "MODELSCOPE_CACHE" "./models/modelscope"
    Set-EnvFileValue ".env" "TORCH_HOME" "./models/torch"
    Set-EnvFileValue ".env" "OPENBLAS_NUM_THREADS" "1"
    Set-EnvFileValue ".env" "OMP_NUM_THREADS" "1"
    Set-EnvFileValue ".env" "MKL_NUM_THREADS" "1"
    Set-EnvFileValue ".env" "NUMEXPR_NUM_THREADS" "1"
    Set-EnvFileValue ".env" "VECLIB_MAXIMUM_THREADS" "1"
    Set-EnvFileValue ".env" "BLIS_NUM_THREADS" "1"
    Set-EnvFileValue ".env" "GOTO_NUM_THREADS" "1"
    Set-EnvFileValue ".env" "OPENBLAS_MAIN_FREE" "1"
    Set-EnvFileValue ".env" "ASR_ENGINE" $AsrEngine
    Set-EnvFileValue ".env" "ASR_STARTUP_ALLOW_DOWNLOAD" "false"
    Set-EnvFileValue ".env" "ASR_STARTUP_FALLBACKS" "sensevoice_zh,paraformer_full,paraformer"
    Set-EnvFileValue ".env" "ASR_DEVICE" $Device
    Set-EnvFileValue ".env" "ASR_FUNASR_ALLOW_ROCM_GPU" "false"
    Set-EnvFileValue ".env" "ASR_FUNASR_ALLOW_ROCM_PARAFORMER" "false"
    Set-EnvFileValue ".env" "ASR_FUNASR_ROCM_SAFE_VERSIONS" $StableFunasrVersion
    Set-EnvFileValue ".env" "ASR_LOAD_TIMEOUT_SEC" "3600"
    Set-EnvFileValue ".env" "SPEAKER_ENGINE" "campplus"
    Set-EnvFileValue ".env" "SPEAKER_DEVICE" $Device
    Set-EnvFileValue ".env" "SPEAKER_VECTOR_STORE" "memory"
    Set-EnvFileValue ".env" "DIARIZATION_ENGINE" "funasr_campplus"
    Set-EnvFileValue ".env" "DIARIZATION_PROVIDER" "modelscope"
    Set-EnvFileValue ".env" "DIARIZATION_ENDPOINT" "https://modelscope.cn"
    Set-EnvFileValue ".env" "DIARIZATION_MODEL" "paraformer-zh + fsmn-vad + ct-punc + cam++"
    Set-EnvFileValue ".env" "PYANNOTE_DEVICE" $Device
    Set-EnvFileValue ".env" "HF_HUB_DISABLE_XET" "1"
    if ($script:ResolvedHfEndpoint) {
        Set-EnvFileValue ".env" "HF_ENDPOINT" $script:ResolvedHfEndpoint
    }
    Set-EnvFileValue ".env" "LLM_TIMEOUT_SEC" "200"
    Set-EnvFileValue ".env" "FACE_RECOGNITION_ENABLED" "true"
    Set-EnvFileValue ".env" "FACE_PROVIDER" "insightface"
    Set-EnvFileValue ".env" "FACE_ENDPOINT" "file://./models/face/insightface"
    Set-EnvFileValue ".env" "FACE_MODEL" "buffalo_l"
    Set-EnvFileValue ".env" "FACE_DEVICE" $FaceDevice
    Set-EnvFileValue ".env" "FACE_MATCH_THRESHOLD" "0.55"
    Set-EnvFileValue ".env" "FACE_MATCH_MARGIN" "0.08"
    Set-EnvFileValue ".env" "FACE_FRAME_INTERVAL_SEC" "5"
}

function Ensure-ModelSettings([string]$Device = "cpu", [string]$AsrModel = "sensevoice_zh", [string]$FaceDevice = "") {
    if (-not $FaceDevice) {
        $FaceDevice = $Device
    }
    New-Item -ItemType Directory -Force -Path "config" | Out-Null
    $path = "config\model-settings.json"
    if (Test-Path -LiteralPath $path) {
        try {
            $settings = Get-Content -LiteralPath $path -Raw -Encoding UTF8 | ConvertFrom-Json
            $asrSettings = Ensure-JsonObjectProperty $settings "asr"
            $currentAsrModel = ""
            if ($asrSettings.PSObject.Properties["model"]) {
                $currentAsrModel = [string]$asrSettings.model
            }
            if (-not $currentAsrModel -or $currentAsrModel -eq "qwen3") {
                Set-JsonProperty $asrSettings "model" $AsrModel
                Set-JsonProperty $asrSettings "provider" "modelscope"
                Set-JsonProperty $asrSettings "endpoint" "https://modelscope.cn"
            }
            Set-JsonProperty $asrSettings "device" $Device
            Set-JsonProperty (Ensure-JsonObjectProperty $settings "speaker") "device" $Device
            Set-JsonProperty (Ensure-JsonObjectProperty $settings "diarization") "device" $Device
            Set-JsonProperty (Ensure-JsonObjectProperty $settings "face") "device" $FaceDevice
            Set-JsonProperty (Ensure-JsonObjectProperty $settings "face") "model" "buffalo_l"
            Set-JsonProperty (Ensure-JsonObjectProperty $settings "face") "provider" "insightface"
            Set-JsonProperty (Ensure-JsonObjectProperty $settings "face") "endpoint" "file://./models/face/insightface"
            Set-JsonProperty (Ensure-JsonObjectProperty $settings "face") "enabled" $true
            Set-JsonProperty (Ensure-JsonObjectProperty $settings "face") "match_threshold" 0.55
            Set-JsonProperty (Ensure-JsonObjectProperty $settings "face") "match_margin" 0.08
            Set-JsonProperty (Ensure-JsonObjectProperty $settings "face") "frame_interval_sec" 5
            Set-JsonProperty (Ensure-JsonObjectProperty $settings "llm") "timeout_sec" 200
            Write-JsonNoBom $path $settings
            Write-Ok "Updated $path devices for $Profile"
        } catch {
            Write-Warn "Existing model settings could not be updated: $($_.Exception.Message)"
        }
        return
    }
    $settings = [ordered]@{
        version = 1
        asr = [ordered]@{
            provider = "modelscope"
            endpoint = "https://modelscope.cn"
            api_key = ""
            model = $AsrModel
            device = $Device
            word_timestamps = $false
            load_timeout_sec = 3600
        }
        speaker = [ordered]@{
            provider = "modelscope"
            endpoint = "https://modelscope.cn"
            api_key = ""
            model = "campplus"
            device = $Device
        }
        diarization = [ordered]@{
            engine = "funasr_campplus"
            provider = "modelscope"
            endpoint = "https://modelscope.cn"
            api_key = ""
            model = "paraformer-zh + fsmn-vad + ct-punc + cam++"
            command = ""
            device = $Device
        }
        face = [ordered]@{
            provider = "insightface"
            endpoint = "file://./models/face/insightface"
            api_key = ""
            model = "buffalo_l"
            device = $FaceDevice
            enabled = $true
            match_threshold = 0.55
            match_margin = 0.08
            frame_interval_sec = 5
        }
        llm = [ordered]@{
            provider = "ollama"
            endpoint = "http://127.0.0.1:11434/v1"
            api_key = ""
            model = "qwen2.5:1.5b"
            enabled = $false
            allow_public = $false
            timeout_sec = 200
            max_input_tokens = 8000
            mock = $false
        }
    }
    Write-JsonNoBom $path $settings
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

function Ensure-FrontendBuild {
    if ($SkipFrontendBuild) {
        Write-Warn "Frontend build skipped."
        return
    }
    if (-not $ForceFrontendBuild -and (Test-Path -LiteralPath "web\dist\index.html")) {
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

function Write-StartScripts($Python) {
    $defaultVenv = $VenvPath
    $activeDir = Resolve-ProjectPath ".runtime" -CreateDirectory
    Set-Content -LiteralPath (Join-Path $activeDir "active-venv.txt") -Value $defaultVenv -Encoding ASCII
    $launcher = @"
param(
    [string]`$VenvPath = "$defaultVenv",
    [string]`$Url = "",
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

`$script:TranscriptStarted = `$false
`$script:StartLogPath = `$null
`$logDir = Join-Path `$Root "logs"
try {
    New-Item -ItemType Directory -Force -Path `$logDir | Out-Null
    `$script:StartLogPath = Join-Path `$logDir ("start-windows-{0}.log" -f (Get-Date -Format "yyyyMMdd-HHmmss"))
    Start-Transcript -LiteralPath `$script:StartLogPath -Append | Out-Null
    `$script:TranscriptStarted = `$true
    Write-Host "Startup log: `$script:StartLogPath"
} catch {
    Write-Warning "Startup logging is unavailable: `$(`$_.Exception.Message)"
}

function Stop-StartupTranscript {
    if (`$script:TranscriptStarted) {
        try {
            Stop-Transcript | Out-Null
        } catch {
        }
        `$script:TranscriptStarted = `$false
    }
}

trap {
    Write-Host ""
    Write-Host "[ERROR] `$(`$_.Exception.Message)" -ForegroundColor Red
    if (`$script:StartLogPath) {
        Write-Host "Latest startup log: `$script:StartLogPath"
    }
    Stop-StartupTranscript
    exit 1
}

function Get-EnvFileValue([string]`$Key, [string]`$Default = "") {
    `$processValue = [System.Environment]::GetEnvironmentVariable(`$Key, "Process")
    if (`$processValue) {
        return `$processValue
    }
    `$envPath = Join-Path `$Root ".env"
    if (-not (Test-Path -LiteralPath `$envPath)) {
        return `$Default
    }
    foreach (`$line in (Get-Content -LiteralPath `$envPath -Encoding UTF8)) {
        `$pattern = '^\s*' + [regex]::Escape(`$Key) + '\s*=\s*(.*)'
        if (`$line -match `$pattern) {
            `$value = `$Matches[1].Trim()
            `$value = `$value -replace "\s+#.*$", ""
            `$value = `$value.Trim().Trim("'").Trim('"')
            if (`$value) {
                return `$value
            }
            return `$Default
        }
    }
    return `$Default
}

function Get-StartUrl {
    if (`$Url) {
        return `$Url
    }
    `$hostValue = (Get-EnvFileValue "HOST" "127.0.0.1").Trim()
    `$portValue = (Get-EnvFileValue "PORT" "8321").Trim()
    `$httpsValue = (Get-EnvFileValue "ENABLE_HTTPS" "false").Trim().ToLowerInvariant()
    `$portNumber = 8321
    if (-not [int]::TryParse(`$portValue, [ref]`$portNumber) -or `$portNumber -le 0 -or `$portNumber -gt 65535) {
        Write-Warning "Invalid PORT=`$portValue in .env; falling back to 8321."
        `$portNumber = 8321
    }
    if (-not `$hostValue -or `$hostValue -eq "0.0.0.0" -or `$hostValue -eq "::" -or `$hostValue -eq "*") {
        `$hostValue = "127.0.0.1"
    } elseif (`$hostValue.Contains(":") -and -not `$hostValue.StartsWith("[")) {
        `$hostValue = "[`$hostValue]"
    }
    `$scheme = if (`$httpsValue -in @("1", "true", "yes", "on")) { "https" } else { "http" }
    return "`${scheme}://`${hostValue}:`${portNumber}"
}

function Test-TcpPortOpen([string]`$TargetHost, [int]`$PortNumber) {
    try {
        `$client = New-Object System.Net.Sockets.TcpClient
        `$async = `$client.BeginConnect(`$TargetHost, `$PortNumber, `$null, `$null)
        `$connected = `$async.AsyncWaitHandle.WaitOne(300)
        if (`$connected) {
            `$client.EndConnect(`$async)
        }
        `$client.Close()
        return [bool]`$connected
    } catch {
        return `$false
    }
}

function Test-LocalPortInUse([int]`$PortNumber) {
    try {
        `$listener = Get-NetTCPConnection -State Listen -LocalPort `$PortNumber -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if (`$listener) {
            return `$true
        }
    } catch {
    }
    return Test-TcpPortOpen "127.0.0.1" `$PortNumber
}

function Get-PortFromUrl([string]`$TargetUrl) {
    try {
        `$uri = [Uri]`$TargetUrl
        return `$uri.Port
    } catch {
        return 8321
    }
}

function Test-ProjectFfmpegAvailable {
    if (Get-Command "ffmpeg" -ErrorAction SilentlyContinue) {
        Write-Host "FFmpeg detected in PATH."
        return `$true
    }
    `$ffmpegExe = if (`$IsWindows -or `$env:OS -eq "Windows_NT") { "ffmpeg.exe" } else { "ffmpeg" }
    foreach (`$candidate in @(
        (Join-Path `$Root ".runtime\ffmpeg\bin\`$ffmpegExe"),
        (Join-Path `$Root "offline\ffmpeg\bin\`$ffmpegExe"),
        (Join-Path `$Root "tools\ffmpeg\bin\`$ffmpegExe"),
        (Join-Path `$Root "ffmpeg\bin\`$ffmpegExe")
    )) {
        if (Test-Path -LiteralPath `$candidate) {
            Write-Host "Project FFmpeg detected: `$candidate"
            return `$true
        }
    }
    return `$false
}

function Test-ImageioFfmpegAvailable([string]`$PythonExe) {
    `$code = @"
import os
try:
    import imageio_ffmpeg
    path = imageio_ffmpeg.get_ffmpeg_exe()
    print(path or "")
    raise SystemExit(0 if path and os.path.isfile(path) else 1)
except Exception as exc:
    print(str(exc))
    raise SystemExit(1)
"@
    `$output = @(& `$PythonExe -c `$code 2>`$null)
    if (`$LASTEXITCODE -eq 0) {
        if (`$output) {
            Write-Host "imageio-ffmpeg detected: `$(`$output[-1])"
        }
        return `$true
    }
    return `$false
}

function Install-ImageioFfmpeg([string]`$PythonExe) {
    `$cacheDir = Join-Path `$Root ".download-cache\pip"
    New-Item -ItemType Directory -Force -Path `$cacheDir | Out-Null
    `$env:PIP_CACHE_DIR = `$cacheDir
    `$findLinks = @()
    foreach (`$dir in @(
        (Join-Path `$Root "offline\wheels"),
        (Join-Path `$Root ".download-cache\wheels")
    )) {
        if (Test-Path -LiteralPath `$dir) {
            `$findLinks += @("--find-links", `$dir)
        }
    }
    foreach (`$index in @(
        "https://pypi.tuna.tsinghua.edu.cn/simple",
        "https://mirrors.aliyun.com/pypi/simple",
        "https://pypi.org/simple"
    )) {
        Write-Host "Installing imageio-ffmpeg for MP4/WebM upload decoding: `$index"
        & `$PythonExe -m pip install --disable-pip-version-check --prefer-binary @findLinks -i `$index "imageio-ffmpeg>=0.5.1,<1.0.0"
        if (`$LASTEXITCODE -eq 0) {
            return `$true
        }
        Write-Warning "imageio-ffmpeg install failed from `$index"
    }
    return `$false
}

function Ensure-AudioTranscodeRuntime([string]`$PythonExe) {
    if (Test-ProjectFfmpegAvailable) {
        return
    }
    if (Test-ImageioFfmpegAvailable `$PythonExe) {
        return
    }
    Write-Warning "FFmpeg runtime is missing. Trying to install project-local imageio-ffmpeg."
    if ((Install-ImageioFfmpeg `$PythonExe) -and (Test-ImageioFfmpegAvailable `$PythonExe)) {
        return
    }
    Write-Warning "MP4/WebM/M4A uploads may fail until FFmpeg or imageio-ffmpeg is available in this virtual environment."
}

`$candidateVenvs = @()
if (`$VenvPath) {
    `$candidateVenvs += `$VenvPath
} else {
    `$activeVenvPath = Join-Path `$Root ".runtime\active-venv.txt"
    if (Test-Path -LiteralPath `$activeVenvPath) {
        `$activeVenv = (Get-Content -LiteralPath `$activeVenvPath -TotalCount 1).Trim()
        if (`$activeVenv) {
            `$candidateVenvs += `$activeVenv
        }
    }
}
`$candidateVenvs += @(".venv-win", ".venv-nvidia-win", ".venv-rocm-win")

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
    throw "Python environment not found. Run exactly one installer first: .\install-windows.cmd, .\install-windows-amd-gpu.cmd, or .\install-windows-nvidia-gpu.cmd. You can also pass -VenvPath explicitly."
}
Write-Host "Using Python: `$Python"

Set-Location `$Root
`$env:OPENBLAS_NUM_THREADS = "1"
`$env:OMP_NUM_THREADS = "1"
`$env:MKL_NUM_THREADS = "1"
`$env:NUMEXPR_NUM_THREADS = "1"
`$env:VECLIB_MAXIMUM_THREADS = "1"
`$env:BLIS_NUM_THREADS = "1"
`$env:GOTO_NUM_THREADS = "1"
`$env:OPENBLAS_MAIN_FREE = "1"
`$env:HF_HUB_DISABLE_XET = "1"

Ensure-AudioTranscodeRuntime `$Python

`$ResolvedUrl = Get-StartUrl
Write-Host "Open URL: `$ResolvedUrl"
`$ResolvedPort = Get-PortFromUrl `$ResolvedUrl
if (Test-LocalPortInUse `$ResolvedPort) {
    try {
        `$healthUrl = "`$ResolvedUrl/health"
        `$response = Invoke-WebRequest -Uri `$healthUrl -UseBasicParsing -TimeoutSec 2
        if (`$response.StatusCode -ge 200 -and `$response.StatusCode -lt 500) {
            Write-Host "Service already running at `$ResolvedUrl"
            if (-not `$NoBrowser) {
                Start-Process `$ResolvedUrl
            }
            Stop-StartupTranscript
            exit 0
        }
    } catch {
    }
    throw "Port `$ResolvedPort is already in use. Change PORT in .env, for example PORT=8322, then rerun start-windows.ps1."
}

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
    } -ArgumentList `$ResolvedUrl | Out-Null
}

`$oldErrorActionPreference = `$ErrorActionPreference
try {
    `$ErrorActionPreference = "Continue"
    & `$Python "main.py" 2>&1 | ForEach-Object { Write-Host `$_ }
    `$serverExitCode = `$LASTEXITCODE
} finally {
    `$ErrorActionPreference = `$oldErrorActionPreference
}
if (`$serverExitCode -ne 0) {
    throw "Server exited with code `$serverExitCode"
}
Stop-StartupTranscript
exit 0
"@
    Set-Content -LiteralPath "start-windows.ps1" -Value $launcher -Encoding UTF8
}

try {
    Assert-Windows
    $ProjectRoot = Resolve-ProjectRoot
    Set-Location $ProjectRoot
    Set-NumericRuntimeThreadEnv

    if ($Profile -eq "AmdRocm") {
        $args = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "scripts\install-windows-rocm.ps1")
        if ($PythonExe) { $args += @("-PythonExe", $PythonExe) }
        if ($PythonRuntimeDir) { $args += @("-PythonRuntimeDir", $PythonRuntimeDir) }
        if ($PythonInstaller) { $args += @("-PythonInstaller", $PythonInstaller) }
        foreach ($url in $PythonRuntimeUrls) { $args += @("-PythonRuntimeUrls", $url) }
        if ($OfflineAssetsDir) { $args += @("-OfflineAssetsDir", $OfflineAssetsDir) }
        if ($RocmWheelCache) { $args += @("-WheelCache", $RocmWheelCache) }
        if ($RecreateVenv) { $args += "-RecreateVenv" }
        if ($ForceTorch) { $args += "-ForceTorch" }
        if ($ForceDeps) { $args += "-ForceDeps" }
        if ($ForceFrontendBuild) { $args += "-ForceFrontendBuild" }
        if ($SkipFrontendBuild) { $args += "-SkipFrontendBuild" }
        if ($SkipGpuCheck) { $args += "-SkipGpuCheck" }
        if ($MirrorMode) { $args += @("-MirrorMode", $MirrorMode) }
        if ($HfEndpoint) { $args += @("-HfEndpoint", $HfEndpoint) }
        foreach ($url in $RocmBaseUrls) { $args += @("-RocmBaseUrls", $url) }
        if ($Proxy) { $args += @("-Proxy", $Proxy) }
        if ($Aria2cExe) { $args += @("-Aria2cExe", $Aria2cExe) }
        if ($Aria2Connections) { $args += @("-Aria2Connections", "$Aria2Connections") }
        if ($Aria2LowestSpeedLimit) { $args += @("-Aria2LowestSpeedLimit", $Aria2LowestSpeedLimit) }
        if ($InstallAria2) { $args += "-InstallAria2" }
        if ($PrintRocmUrls) { $args += "-PrintRocmUrls" }
        if ($OfflineOnly) { $args += "-OfflineOnly" }
        if ($StartServer) { $args += "-StartServer" }
        Invoke-External "powershell" $args "AMD ROCm installer failed"
        return
    }

    if ($Profile -eq "NvidiaCuda") {
        if (-not $VenvPath) {
            $VenvPath = ".venv-nvidia-win"
        }
        $UseMemoryVectorStore = -not $InstallChroma

        Initialize-InstallLog "install-windows-nvidia"
        Resolve-InstallerMirrors

        Write-Step "Checking NVIDIA GPU"
        Assert-NvidiaGraphics $SkipGpuCheck

        Write-Step "Preparing project-local Python 3.12 runtime"
        $PythonExe = Find-Python312 $PythonExe
        Write-Ok "Python: $PythonExe"

        Write-Step "Checking FFmpeg"
        Ensure-Ffmpeg

        Write-Step "Creating NVIDIA virtual environment"
        if (-not $RecreateVenv) {
            Write-Host "NVIDIA installer recreates the project virtual environment by default: $VenvPath"
        }
        $VenvPython = Create-Or-RecreateVenv $PythonExe $VenvPath $true
        Write-Ok "Virtual environment: $VenvPython"

        Write-Step "Installing NVIDIA CUDA PyTorch"
        Invoke-PipInstallWithMirrors $VenvPython @("install", "-U", "pip", "wheel", "setuptools") "Failed to update pip tooling"
        Install-CudaTorch $VenvPython

        Write-Step "Installing project dependencies"
        Install-ProjectDeps $VenvPython $UseMemoryVectorStore $true $true

        Write-Step "Installing face recognition runtime"
        Install-FaceDeps $VenvPython "cuda"

        Write-Step "Configuring local defaults for NVIDIA GPU"
        Ensure-EnvConfig "cuda" "sensevoice_zh" "cuda"
        Ensure-ModelSettings "cuda" "sensevoice_zh" "cuda"
        Write-StartScripts $VenvPython

        Write-Step "Building frontend"
        Ensure-FrontendBuild

        Write-Step "Verifying NVIDIA CUDA PyTorch"
        Verify-CudaPyTorch $VenvPython (-not $SkipGpuCheck)

        Write-Host ""
        Write-Host "Installation complete." -ForegroundColor Green
        Write-Host "Start command: powershell -NoProfile -ExecutionPolicy Bypass -File .\start-windows.ps1"
        Write-Host "Open: http://127.0.0.1:8321"

        if ($StartServer) {
            Write-Step "Starting server"
            Invoke-External $VenvPython @("main.py") "Server exited with an error"
        }
        return
    }

    if (-not $VenvPath) {
        $VenvPath = ".venv-win"
    }
    $UseMemoryVectorStore = -not $InstallChroma

    Initialize-InstallLog "install-windows"
    Resolve-InstallerMirrors

    Write-Step "Preparing project-local Python 3.12 runtime"
    $PythonExe = Find-Python312 $PythonExe
    Write-Ok "Python: $PythonExe"

    Write-Step "Checking FFmpeg"
    Ensure-Ffmpeg

    Write-Step "Creating CPU virtual environment"
    if (-not $RecreateVenv) {
        Write-Host "CPU installer recreates the project virtual environment by default: $VenvPath"
    }
    $VenvPython = Create-Or-RecreateVenv $PythonExe $VenvPath $true
    Write-Ok "Virtual environment: $VenvPython"

    Write-Step "Installing Python dependencies"
    Install-ProjectDeps $VenvPython $UseMemoryVectorStore $false $true

    Write-Step "Installing face recognition runtime"
    Install-FaceDeps $VenvPython "cpu"

    Write-Step "Configuring local defaults"
    Ensure-EnvConfig "cpu" "sensevoice_zh" "cpu"
    Ensure-ModelSettings "cpu" "sensevoice_zh" "cpu"
    Write-StartScripts $VenvPython

    Write-Step "Building frontend"
    Ensure-FrontendBuild

    Write-Host ""
    Write-Host "Installation complete." -ForegroundColor Green
    Write-Host "Start command: powershell -NoProfile -ExecutionPolicy Bypass -File .\start-windows.ps1"
    Write-Host "Open: http://127.0.0.1:8321"

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
