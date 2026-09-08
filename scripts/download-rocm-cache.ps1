param(
    [string]$CacheDir = ".download-cache\rocm-win-7.2.1",
    [int]$Parallel = 4,
    [int]$Retry = 20
)

$ErrorActionPreference = "Stop"

function Resolve-ProjectRoot {
    $scriptDir = $PSScriptRoot
    if (-not $scriptDir -and $PSCommandPath) {
        $scriptDir = Split-Path -Parent $PSCommandPath
    }
    if (-not $scriptDir -and $MyInvocation.MyCommand.Path) {
        $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    }
    if (-not $scriptDir) {
        throw "Unable to resolve script directory."
    }
    return (Resolve-Path (Join-Path $scriptDir "..")).Path
}

function Resolve-CachePath($Path) {
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return $Path
    }
    return (Join-Path (Resolve-ProjectRoot) $Path)
}

function Format-Bytes([int64]$Bytes) {
    if ($Bytes -ge 1GB) {
        return ("{0:n2} GB" -f ($Bytes / 1GB))
    }
    if ($Bytes -ge 1MB) {
        return ("{0:n1} MB" -f ($Bytes / 1MB))
    }
    if ($Bytes -ge 1KB) {
        return ("{0:n1} KB" -f ($Bytes / 1KB))
    }
    return "$Bytes B"
}

$artifacts = @(
    [pscustomobject]@{ Name = "rocm_sdk_core-7.2.1-py3-none-win_amd64.whl"; Url = "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl"; Expected = 644793492L },
    [pscustomobject]@{ Name = "rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl"; Url = "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl"; Expected = 232840013L },
    [pscustomobject]@{ Name = "rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl"; Url = "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl"; Expected = 489964648L },
    [pscustomobject]@{ Name = "rocm-7.2.1.tar.gz"; Url = "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm-7.2.1.tar.gz"; Expected = 15940L },
    [pscustomobject]@{ Name = "torch-2.9.1+rocm7.2.1-cp312-cp312-win_amd64.whl"; Url = "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl"; Expected = 821065907L },
    [pscustomobject]@{ Name = "torchaudio-2.9.1+rocm7.2.1-cp312-cp312-win_amd64.whl"; Url = "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torchaudio-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl"; Expected = 514708L },
    [pscustomobject]@{ Name = "torchvision-0.24.1+rocm7.2.1-cp312-cp312-win_amd64.whl"; Url = "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torchvision-0.24.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl"; Expected = 1863312L }
)

$cache = Resolve-CachePath $CacheDir
New-Item -ItemType Directory -Force -Path $cache | Out-Null
Write-Host "ROCm cache: $cache"

$curl = Get-Command "curl.exe" -ErrorAction SilentlyContinue
if (-not $curl) {
    throw "curl.exe was not found."
}

function Test-Complete($Artifact) {
    $target = Join-Path $cache $Artifact.Name
    if (-not (Test-Path -LiteralPath $target)) {
        return $false
    }
    return ((Get-Item -LiteralPath $target).Length -eq [int64]$Artifact.Expected)
}

function Start-ArtifactDownload($Artifact) {
    return Start-Job -Name $Artifact.Name -ArgumentList $Artifact.Name, $Artifact.Url, ([int64]$Artifact.Expected), $cache, $Retry -ScriptBlock {
        param($Name, $Url, [int64]$Expected, $Cache, [int]$Retry)
        $target = Join-Path $Cache $Name
        $part = "$target.part"
        if ((Test-Path -LiteralPath $target) -and ((Get-Item -LiteralPath $target).Length -eq $Expected)) {
            return "SKIP $Name"
        }
        if (Test-Path -LiteralPath $target) {
            $bad = "$target.bad-$([DateTime]::Now.ToString('yyyyMMddHHmmss'))"
            Move-Item -LiteralPath $target -Destination $bad -Force
        }
        if (Test-Path -LiteralPath $part) {
            $partLen = (Get-Item -LiteralPath $part).Length
            if ($partLen -eq $Expected) {
                Move-Item -LiteralPath $part -Destination $target -Force
                return "DONE $Name from existing part"
            }
            if ($partLen -gt $Expected) {
                Remove-Item -LiteralPath $part -Force
            }
        }
        & curl.exe --silent --show-error -L --fail --retry $Retry --retry-delay 10 --retry-all-errors --connect-timeout 30 --speed-time 120 --speed-limit 1024 --continue-at - --output $part $Url
        if ($LASTEXITCODE -ne 0) {
            throw "curl failed for $Name with exit code $LASTEXITCODE"
        }
        $actual = (Get-Item -LiteralPath $part).Length
        if ($actual -ne $Expected) {
            throw "size mismatch for $Name expected=$Expected actual=$actual"
        }
        Move-Item -LiteralPath $part -Destination $target -Force
        return "DONE $Name"
    }
}

function Show-Progress {
    $doneBytes = 0L
    $totalBytes = 0L
    $lines = @()
    foreach ($artifact in $artifacts) {
        $target = Join-Path $cache $artifact.Name
        $part = "$target.part"
        $size = 0L
        $state = "pending"
        if (Test-Path -LiteralPath $target) {
            $size = (Get-Item -LiteralPath $target).Length
            $state = "done"
        } elseif (Test-Path -LiteralPath $part) {
            $size = (Get-Item -LiteralPath $part).Length
            $state = "part"
        }
        $expected = [int64]$artifact.Expected
        $totalBytes += $expected
        $doneBytes += [Math]::Min([int64]$size, $expected)
        $pct = if ($expected -gt 0) { [Math]::Round(([Math]::Min([int64]$size, $expected) * 100.0 / $expected), 1) } else { 0 }
        $lines += ("  {0,-72} {1,6}% {2,9} / {3,9} {4}" -f $artifact.Name, $pct, (Format-Bytes $size), (Format-Bytes $expected), $state)
    }
    $overall = if ($totalBytes -gt 0) { [Math]::Round(($doneBytes * 100.0 / $totalBytes), 1) } else { 0 }
    Write-Host ""
    Write-Host ("[{0}] overall {1}% ({2} / {3})" -f (Get-Date -Format "HH:mm:ss"), $overall, (Format-Bytes $doneBytes), (Format-Bytes $totalBytes))
    $lines | ForEach-Object { Write-Host $_ }
}

$pending = New-Object System.Collections.Queue
foreach ($artifact in $artifacts) {
    if (Test-Complete $artifact) {
        Write-Host "SKIP complete $($artifact.Name)"
    } else {
        $pending.Enqueue($artifact)
    }
}

$jobs = @()
while ($pending.Count -gt 0 -or ($jobs | Where-Object { $_.State -in @("Running", "NotStarted") }).Count -gt 0) {
    while ($pending.Count -gt 0 -and ($jobs | Where-Object { $_.State -in @("Running", "NotStarted") }).Count -lt $Parallel) {
        $next = $pending.Dequeue()
        Write-Host "START $($next.Name)"
        $jobs += Start-ArtifactDownload $next
    }
    Show-Progress
    Start-Sleep -Seconds 30
}

$failed = $false
foreach ($job in $jobs) {
    Receive-Job $job -Keep
    if ($job.State -ne "Completed") {
        $failed = $true
        Write-Host "FAILED JOB $($job.Name): $($job.State)" -ForegroundColor Red
    }
}
if ($jobs.Count -gt 0) {
    Remove-Job $jobs -Force -ErrorAction SilentlyContinue
}

foreach ($artifact in $artifacts) {
    $target = Join-Path $cache $artifact.Name
    if (-not (Test-Path -LiteralPath $target)) {
        throw "missing after download: $($artifact.Name)"
    }
    $actual = (Get-Item -LiteralPath $target).Length
    if ($actual -ne [int64]$artifact.Expected) {
        throw "final size mismatch: $($artifact.Name) expected=$($artifact.Expected) actual=$actual"
    }
}
if ($failed) {
    throw "one or more ROCm download jobs failed"
}

Write-Host ""
Write-Host "ROCm offline cache complete." -ForegroundColor Green
Get-ChildItem -LiteralPath $cache | Select-Object Name,Length,LastWriteTime
