param(
    [string]$VenvPath = "",
    [ValidateSet("auto", "cpu", "cuda", "directml")]
    [string]$OrtFlavor = "auto",
    [ValidateSet("China", "Official", "Auto")]
    [string]$MirrorMode = "China",
    [string[]]$PipIndexUrls = @()
)

$ErrorActionPreference = "Stop"
$env:PIP_DISABLE_PIP_VERSION_CHECK = "1"

$ChinaPipIndexUrls = @(
    "https://pypi.tuna.tsinghua.edu.cn/simple",
    "https://mirrors.aliyun.com/pypi/simple"
)
$OfficialPipIndexUrl = "https://pypi.org/simple"
$StableInsightFaceVersion = "1.0.1"
$StableOnnxRuntimeCpuVersion = "1.29.0"
$StableOnnxRuntimeGpuVersion = "1.29.0"
$StableOnnxRuntimeDirectmlVersion = "1.24.4"
$StableOnnxVersion = "1.22.0"
$StableOpenCvVersion = "4.14.0.94"
$StablePillowVersion = "12.3.0"
$StableScikitImageVersion = "0.26.0"

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
    & $Exe @ArgumentList | ForEach-Object { Write-Host $_ }
    if ($LASTEXITCODE -ne 0) {
        throw "$FailureMessage (exit code $LASTEXITCODE)"
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

function Add-UniqueValue([System.Collections.ArrayList]$List, $Value) {
    $text = ([string]$Value).Trim()
    if (-not $text) {
        return
    }
    foreach ($existing in $List) {
        if ([string]::Equals([string]$existing, $text, [System.StringComparison]::OrdinalIgnoreCase)) {
            return
        }
    }
    [void]$List.Add($text)
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
        throw "Unable to resolve repair script directory."
    }
    return (Resolve-Path (Join-Path $scriptDir "..")).Path
}

function Resolve-PipIndexes {
    $indexes = New-Object System.Collections.ArrayList
    foreach ($item in @($PipIndexUrls)) {
        foreach ($part in ([string]$item -split "[,;]")) {
            Add-UniqueValue $indexes $part
        }
    }
    if ($indexes.Count -eq 0) {
        if ($MirrorMode -eq "Auto" -and $env:PIP_INDEX_URL) {
            foreach ($part in ([string]$env:PIP_INDEX_URL -split "[,;]")) {
                Add-UniqueValue $indexes $part
            }
        } elseif ($MirrorMode -eq "Official") {
            Add-UniqueValue $indexes $OfficialPipIndexUrl
        } else {
            foreach ($url in $ChinaPipIndexUrls) {
                Add-UniqueValue $indexes $url
            }
            Add-UniqueValue $indexes $OfficialPipIndexUrl
        }
    }
    return $indexes
}

function Resolve-VenvPython {
    $candidates = New-Object System.Collections.ArrayList
    if ($VenvPath) {
        Add-UniqueValue $candidates $VenvPath
    }
    $activeFile = Join-Path (Resolve-Path ".").Path ".runtime\active-venv.txt"
    if (-not $VenvPath -and (Test-Path -LiteralPath $activeFile)) {
        $active = (Get-Content -LiteralPath $activeFile -TotalCount 1).Trim()
        Add-UniqueValue $candidates $active
    }
    foreach ($candidate in @(".venv-win", ".venv-nvidia-win", ".venv-rocm-win")) {
        Add-UniqueValue $candidates $candidate
    }
    foreach ($candidate in $candidates) {
        $venvRoot = if ([System.IO.Path]::IsPathRooted([string]$candidate)) {
            [string]$candidate
        } else {
            Join-Path (Resolve-Path ".").Path ([string]$candidate)
        }
        $python = Join-Path $venvRoot "Scripts\python.exe"
        if (Test-Path -LiteralPath $python) {
            return @{
                VenvPath = [string]$candidate
                Python = $python
            }
        }
    }
    throw "No project virtual environment was found. Run install-windows.cmd first, or pass -VenvPath .venv-win."
}

function Resolve-OrtFlavor([string]$DetectedVenv) {
    if ($OrtFlavor -ne "auto") {
        return $OrtFlavor
    }
    if ($DetectedVenv -match "(?i)nvidia") {
        return "cuda"
    }
    if ($DetectedVenv -match "(?i)(rocm|amd)") {
        return "directml"
    }
    return "cpu"
}

function Invoke-PipInstallWithMirrors($Python, [string[]]$InstallArgs, $Description) {
    $lastError = ""
    $offlineWheelArgs = @()
    $offlineWheels = Join-Path (Resolve-Path ".").Path "offline\wheels"
    if (Test-Path -LiteralPath $offlineWheels) {
        $offlineWheelArgs = @("--find-links", $offlineWheels)
        Write-Host "offline wheels: $offlineWheels"
    }
    foreach ($indexUrl in (Resolve-PipIndexes)) {
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

function Install-OpenCv($Python) {
    if (Test-Cv2Ready $Python) {
        Write-Ok "OpenCV cv2 import is available."
        return
    }
    Write-Host "Cleaning existing OpenCV packages before installing cv2 runtime."
    [void](Invoke-NativeQuietExitCode $Python @("-m", "pip", "uninstall", "-y", "opencv-python", "opencv-python-headless"))
    $lastError = ""
    foreach ($package in @("opencv-python-headless==$StableOpenCvVersion", "opencv-python==$StableOpenCvVersion")) {
        try {
            Invoke-PipInstallWithMirrors $Python @("install", $package, "--prefer-binary") "Failed to install $package"
            if (Test-Cv2Ready $Python) {
                return
            }
            $lastError = "$package installed but import cv2 still failed"
            Write-Warn $lastError
        } catch {
            $lastError = $_.Exception.Message
            Write-Warn $lastError
        }
    }
    throw "Failed to install OpenCV cv2 runtime. Last error: $lastError"
}

function Test-FaceRuntimeReady($Python, [string]$Flavor) {
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
print('python=' + __import__('sys').executable)
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
}.get('$Flavor', 'onnxruntime')
ort_expected = {
    'cpu': '$StableOnnxRuntimeCpuVersion',
    'cuda': '$StableOnnxRuntimeGpuVersion',
    'directml': '$StableOnnxRuntimeDirectmlVersion',
}.get('$Flavor', '$StableOnnxRuntimeCpuVersion')
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
}.get('$Flavor', 'CPUExecutionProvider')
if required not in providers:
    print('missing_provider=' + required)
    raise SystemExit(2)
"@
    & $Python -c $code
    return ($LASTEXITCODE -eq 0)
}

try {
    $root = Resolve-ProjectRoot
    Set-Location $root
    $resolved = Resolve-VenvPython
    $python = $resolved.Python
    $flavor = Resolve-OrtFlavor $resolved.VenvPath
    $ortPackage = if ($flavor -eq "cuda") {
        "onnxruntime-gpu==$StableOnnxRuntimeGpuVersion"
    } elseif ($flavor -eq "directml") {
        "onnxruntime-directml==$StableOnnxRuntimeDirectmlVersion"
    } else {
        "onnxruntime==$StableOnnxRuntimeCpuVersion"
    }

    Write-Step "Repairing face recognition runtime"
    Write-Host "venv: $($resolved.VenvPath)"
    Write-Host "python: $python"
    Write-Host "ONNXRuntime flavor: $flavor"

    Invoke-PipInstallWithMirrors $python @("install", "-U", "pip", "wheel", "setuptools") "Failed to update pip tooling"
    Invoke-PipInstallWithMirrors $python @(
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
    ) "Failed to install face runtime dependencies"
    Install-OpenCv $python
    Invoke-PipInstallWithMirrors $python @(
        "install",
        "insightface==$StableInsightFaceVersion",
        "--no-deps",
        "--prefer-binary"
    ) "Failed to install InsightFace"

    if (-not (Test-FaceRuntimeReady $python $flavor)) {
        throw "Face runtime verification failed."
    }
    Write-Ok "Face runtime is ready. Restart the server with .\start-windows.cmd."
} catch {
    Write-Host ""
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}
