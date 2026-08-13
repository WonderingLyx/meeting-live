param(
    [string]$VenvPath = "",
    [string]$Url = "http://127.0.0.1:8000",
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

$candidateVenvs = @()
if ($VenvPath) {
    $candidateVenvs += $VenvPath
}
$candidateVenvs += @(".venv-nvidia-win", ".venv-win", ".venv-rocm-win")

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
    throw "Python environment not found. Run .\install-windows.cmd first, or pass -VenvPath .venv-nvidia-win."
}

Set-Location $Root
$env:HF_HUB_DISABLE_XET = "1"

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
    } -ArgumentList $Url | Out-Null
}

& $Python "main.py"
