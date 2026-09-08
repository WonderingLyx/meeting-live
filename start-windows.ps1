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
    throw "Port $ResolvedPort is already in use. Change PORT in .env, for example PORT=8001, then rerun start-windows.ps1."
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
