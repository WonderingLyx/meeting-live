param(
    [string]$VenvPath = "",
    [string]$Url = "",
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"

$Root = $PSScriptRoot
if (-not $Root -and $PSCommandPath) {
    $Root = Split-Path -Parent $PSCommandPath
}
if (-not $Root -and $MyInvocation.MyCommand.Path) {
    $Root = Split-Path -Parent $MyInvocation.MyCommand.Path
}
if (-not $Root) {
    throw "Unable to resolve project directory."
}

$script:TranscriptStarted = $false
$script:StartLogPath = $null
$logDir = Join-Path $Root "logs"
try {
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    $script:StartLogPath = Join-Path $logDir ("start-windows-{0}.log" -f (Get-Date -Format "yyyyMMdd-HHmmss"))
    Start-Transcript -LiteralPath $script:StartLogPath -Append | Out-Null
    $script:TranscriptStarted = $true
    Write-Host "Startup log: $script:StartLogPath"
} catch {
    Write-Warning "Startup logging is unavailable: $($_.Exception.Message)"
}

function Stop-StartupTranscript {
    if ($script:TranscriptStarted) {
        try {
            Stop-Transcript | Out-Null
        } catch {
        }
        $script:TranscriptStarted = $false
    }
}

trap {
    Write-Host ""
    Write-Host "[ERROR] $($_.Exception.Message)" -ForegroundColor Red
    if ($script:StartLogPath) {
        Write-Host "Latest startup log: $script:StartLogPath"
    }
    Stop-StartupTranscript
    exit 1
}

function Get-EnvFileValue([string]$Key, [string]$Default = "") {
    $processValue = [System.Environment]::GetEnvironmentVariable($Key, "Process")
    if ($processValue) {
        return $processValue
    }
    $envPath = Join-Path $Root ".env"
    if (-not (Test-Path -LiteralPath $envPath)) {
        return $Default
    }
    foreach ($line in (Get-Content -LiteralPath $envPath -Encoding UTF8)) {
        if ($line -match "^\s*$([regex]::Escape($Key))\s*=\s*(.*?)\s*$") {
            $value = $Matches[1].Trim()
            $value = $value -replace "\s+#.*$", ""
            $value = $value.Trim().Trim("'").Trim('"')
            if ($value) {
                return $value
            }
            return $Default
        }
    }
    return $Default
}

function Get-StartUrl {
    if ($Url) {
        return $Url
    }
    $hostValue = (Get-EnvFileValue "HOST" "127.0.0.1").Trim()
    $portValue = (Get-EnvFileValue "PORT" "8321").Trim()
    $httpsValue = (Get-EnvFileValue "ENABLE_HTTPS" "false").Trim().ToLowerInvariant()
    $portNumber = 8321
    if (-not [int]::TryParse($portValue, [ref]$portNumber) -or $portNumber -le 0 -or $portNumber -gt 65535) {
        Write-Warning "Invalid PORT=$portValue in .env; falling back to 8321."
        $portNumber = 8321
    }
    if (-not $hostValue -or $hostValue -eq "0.0.0.0" -or $hostValue -eq "::" -or $hostValue -eq "*") {
        $hostValue = "127.0.0.1"
    } elseif ($hostValue.Contains(":") -and -not $hostValue.StartsWith("[")) {
        $hostValue = "[$hostValue]"
    }
    $scheme = if ($httpsValue -in @("1", "true", "yes", "on")) { "https" } else { "http" }
    return "${scheme}://${hostValue}:${portNumber}"
}

function Test-TcpPortOpen([string]$TargetHost, [int]$PortNumber) {
    try {
        $client = New-Object System.Net.Sockets.TcpClient
        $async = $client.BeginConnect($TargetHost, $PortNumber, $null, $null)
        $connected = $async.AsyncWaitHandle.WaitOne(300)
        if ($connected) {
            $client.EndConnect($async)
        }
        $client.Close()
        return [bool]$connected
    } catch {
        return $false
    }
}

function Test-LocalPortInUse([int]$PortNumber) {
    try {
        $listener = Get-NetTCPConnection -State Listen -LocalPort $PortNumber -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($listener) {
            return $true
        }
    } catch {
    }
    return Test-TcpPortOpen "127.0.0.1" $PortNumber
}

function Get-PortFromUrl([string]$TargetUrl) {
    try {
        $uri = [Uri]$TargetUrl
        return $uri.Port
    } catch {
        return 8321
    }
}

function Set-ProjectFfmpegEnvironment([string]$FfmpegPath) {
    if (-not $FfmpegPath -or -not (Test-Path -LiteralPath $FfmpegPath)) {
        return
    }
    $ffmpegDir = Split-Path -Parent $FfmpegPath
    $env:FFMPEG_BINARY = $FfmpegPath
    $env:FFMPEG_PATH = $FfmpegPath
    if ($ffmpegDir -and -not ($env:PATH -split ";" | Where-Object { $_ -eq $ffmpegDir })) {
        $env:PATH = "$ffmpegDir;$env:PATH"
    }
}

function Test-ProjectFfmpegAvailable {
    $cmd = Get-Command "ffmpeg" -ErrorAction SilentlyContinue
    if ($cmd) {
        Set-ProjectFfmpegEnvironment $cmd.Source
        Write-Host "FFmpeg detected in PATH: $($cmd.Source)"
        return $true
    }
    $ffmpegExe = if ($IsWindows -or $env:OS -eq "Windows_NT") { "ffmpeg.exe" } else { "ffmpeg" }
    foreach ($candidate in @(
        (Join-Path $Root ".runtime\ffmpeg\bin\$ffmpegExe"),
        (Join-Path $Root "offline\ffmpeg\bin\$ffmpegExe"),
        (Join-Path $Root "tools\ffmpeg\bin\$ffmpegExe"),
        (Join-Path $Root "ffmpeg\bin\$ffmpegExe")
    )) {
        if (Test-Path -LiteralPath $candidate) {
            Set-ProjectFfmpegEnvironment $candidate
            Write-Host "Project FFmpeg detected: $candidate"
            return $true
        }
    }
    return $false
}

function Test-ImageioFfmpegAvailable([string]$PythonExe) {
    $code = "import importlib.util,os,sys; spec=importlib.util.find_spec('imageio_ffmpeg'); sys.exit(1) if spec is None else None; import imageio_ffmpeg; path=imageio_ffmpeg.get_ffmpeg_exe(); print(path or ''); sys.exit(0 if path and os.path.isfile(path) else 1)"
    $oldErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $output = @(& $PythonExe -c $code 2>$null)
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $oldErrorActionPreference
    }
    if ($exitCode -eq 0) {
        if ($output) {
            Set-ProjectFfmpegEnvironment $output[-1]
            Write-Host "imageio-ffmpeg detected: $($output[-1])"
        }
        return $true
    }
    return $false
}

function Install-ImageioFfmpeg([string]$PythonExe) {
    $cacheDir = Join-Path $Root ".download-cache\pip"
    New-Item -ItemType Directory -Force -Path $cacheDir | Out-Null
    $env:PIP_CACHE_DIR = $cacheDir
    $findLinks = @()
    foreach ($dir in @(
        (Join-Path $Root "offline\wheels"),
        (Join-Path $Root ".download-cache\wheels")
    )) {
        if (Test-Path -LiteralPath $dir) {
            $findLinks += @("--find-links", $dir)
        }
    }
    foreach ($index in @(
        "https://pypi.tuna.tsinghua.edu.cn/simple",
        "https://mirrors.aliyun.com/pypi/simple",
        "https://pypi.org/simple"
    )) {
        Write-Host "Installing imageio-ffmpeg for MP4/WebM upload decoding: $index"
        $oldErrorActionPreference = $ErrorActionPreference
        try {
            $ErrorActionPreference = "Continue"
            & $PythonExe -m pip install --disable-pip-version-check --prefer-binary @findLinks -i $index "imageio-ffmpeg>=0.5.1,<1.0.0"
            $exitCode = $LASTEXITCODE
        } finally {
            $ErrorActionPreference = $oldErrorActionPreference
        }
        if ($exitCode -eq 0) {
            return $true
        }
        Write-Warning "imageio-ffmpeg install failed from $index"
    }
    return $false
}

function Ensure-AudioTranscodeRuntime([string]$PythonExe) {
    if (Test-ProjectFfmpegAvailable) {
        return
    }
    if (Test-ImageioFfmpegAvailable $PythonExe) {
        return
    }
    Write-Warning "FFmpeg runtime is missing. Trying to install project-local imageio-ffmpeg."
    if ((Install-ImageioFfmpeg $PythonExe) -and (Test-ImageioFfmpegAvailable $PythonExe)) {
        return
    }
    Write-Warning "MP4/WebM/M4A uploads may fail until FFmpeg or imageio-ffmpeg is available in this virtual environment."
}

$candidateVenvs = @()
if ($VenvPath) {
    $candidateVenvs += $VenvPath
} else {
    $activeVenvPath = Join-Path $Root ".runtime\active-venv.txt"
    if (Test-Path -LiteralPath $activeVenvPath) {
        $activeVenv = (Get-Content -LiteralPath $activeVenvPath -TotalCount 1).Trim()
        if ($activeVenv) {
            $candidateVenvs += $activeVenv
        }
    }
}
$candidateVenvs += @(".venv-win", ".venv-rocm-win", ".venv-nvidia-win")

$seen = @{}
$Python = $null
foreach ($candidate in $candidateVenvs) {
    if (-not $candidate -or $seen.ContainsKey($candidate)) {
        continue
    }
    $seen[$candidate] = $true
    $candidatePython = Join-Path $Root (Join-Path $candidate "Scripts\python.exe")
    if (Test-Path -LiteralPath $candidatePython) {
        $Python = $candidatePython
        break
    }
}

if (-not $Python) {
    throw "Python environment not found. Run exactly one installer first: .\install-windows.cmd, .\install-windows-amd-gpu.cmd, or .\install-windows-nvidia-gpu.cmd. You can also pass -VenvPath explicitly."
}
Write-Host "Using Python: $Python"

Set-Location $Root
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
$env:VECLIB_MAXIMUM_THREADS = "1"
$env:BLIS_NUM_THREADS = "1"
$env:GOTO_NUM_THREADS = "1"
$env:OPENBLAS_MAIN_FREE = "1"
$env:HF_HUB_DISABLE_XET = "1"

Ensure-AudioTranscodeRuntime $Python

$ResolvedUrl = Get-StartUrl
Write-Host "Open URL: $ResolvedUrl"
$ResolvedPort = Get-PortFromUrl $ResolvedUrl
if (Test-LocalPortInUse $ResolvedPort) {
    try {
        $healthUrl = "$ResolvedUrl/health"
        $response = Invoke-WebRequest -Uri $healthUrl -UseBasicParsing -TimeoutSec 2
        if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) {
            Write-Host "Service already running at $ResolvedUrl"
            if (-not $NoBrowser) {
                Start-Process $ResolvedUrl
            }
            Stop-StartupTranscript
            exit 0
        }
    } catch {
    }
    throw "Port $ResolvedPort is already in use. Change PORT in .env, for example PORT=8322, then rerun start-windows.ps1."
}

if (-not $NoBrowser) {
    Start-Job -ScriptBlock {
        param($TargetUrl)
        for ($i = 0; $i -lt 60; $i++) {
            try {
                $response = Invoke-WebRequest -Uri $TargetUrl -UseBasicParsing -TimeoutSec 2
                if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) {
                    Start-Process $TargetUrl
                    return
                }
            } catch {
            }
            Start-Sleep -Seconds 1
        }
    } -ArgumentList $ResolvedUrl | Out-Null
}

$oldErrorActionPreference = $ErrorActionPreference
try {
    $ErrorActionPreference = "Continue"
    & $Python "main.py" 2>&1 | ForEach-Object { Write-Host $_ }
    $serverExitCode = $LASTEXITCODE
} finally {
    $ErrorActionPreference = $oldErrorActionPreference
}
if ($serverExitCode -ne 0) {
    throw "Server exited with code $serverExitCode"
}
Stop-StartupTranscript
exit 0
