param(
    [ValidateSet("Cpu", "AmdRocm", "NvidiaCuda")]
    [string]$Profile = "Cpu",
    [string]$PythonExe = "",
    [string]$VenvPath = "",
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
    [string]$TorchBuild = "auto",
    [string[]]$TorchIndexUrls = @(),
    [switch]$InstallAria2,
    [switch]$PrintRocmUrls,
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

$ChinaPipIndexUrls = @(
    "https://pypi.tuna.tsinghua.edu.cn/simple",
    "https://mirrors.aliyun.com/pypi/simple"
)
$OfficialPipIndexUrl = "https://pypi.org/simple"
$ChinaNpmRegistries = @("https://registry.npmmirror.com/")
$OfficialNpmRegistry = "https://registry.npmjs.org/"
$ChinaHfEndpoint = "https://hf-mirror.com"
$ChinaTorchIndexBaseUrls = @("https://mirror.sjtu.edu.cn/pytorch-wheels")
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

function Invoke-External($Exe, [string[]]$ArgumentList, $FailureMessage) {
    & $Exe @ArgumentList
    if ($LASTEXITCODE -ne 0) {
        throw "$FailureMessage (exit code $LASTEXITCODE)"
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

function Invoke-PipInstallWithMirrors($Python, [string[]]$InstallArgs, $Description) {
    $lastError = ""
    foreach ($indexUrl in $script:ResolvedPipIndexUrls) {
        try {
            Write-Host "pip index: $indexUrl"
            $args = @("-m", "pip") + $InstallArgs + @(
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
            foreach ($baseUrl in $ChinaTorchIndexBaseUrls) {
                Add-UniqueValue $urls "$($baseUrl.TrimEnd('/'))/$CudaWheel"
            }
            Add-UniqueValue $urls "$OfficialTorchIndexBaseUrl/$CudaWheel"
        }
    } elseif ($MirrorMode -ne "Official") {
        Add-UniqueValue $urls "$OfficialTorchIndexBaseUrl/$CudaWheel"
    }
    return @($urls)
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

function Get-CudaTorchBuildCandidates {
    if ($TorchBuild -eq "auto") {
        return @((Get-CudaTorchBuildDefinition "latest"), (Get-CudaTorchBuildDefinition "stable"))
    }
    return @((Get-CudaTorchBuildDefinition $TorchBuild))
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
        $code = "import torch; print(torch.__version__); print(torch.version.cuda or ''); print(torch.cuda.is_available())"
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
        $code = "import torch; print(torch.__version__); print(torch.version.cuda or '')"
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
        $code = "import torch; print('torch=', torch.__version__); print('cuda_runtime=', torch.version.cuda); print('gpu_available=', torch.cuda.is_available()); print('device=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
        $lines = @(& $Python -c $code 2>&1)
        foreach ($line in $lines) {
            Write-Warn "[TORCH] $line"
        }
    } catch {
        Write-Warn "[TORCH] $($_.Exception.Message)"
    }
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
        foreach ($indexUrl in (Get-TorchIndexUrls)) {
            try {
                Write-Host "PyTorch CUDA build: $($build.Name) (torch=$($build.Torch), torchaudio=$($build.TorchAudio), torchvision=$($build.TorchVision))"
                Write-Host "PyTorch CUDA index: $indexUrl"
                Invoke-External $Python @(
                    "-m", "pip", "install",
                    "torch==$($build.Torch)",
                    "torchaudio==$($build.TorchAudio)",
                    "torchvision==$($build.TorchVision)",
                    "--index-url", $indexUrl,
                    "--retries", "5",
                    "--timeout", "120",
                    "--prefer-binary"
                ) "Failed to install NVIDIA CUDA PyTorch"
                $ready = if ($SkipGpuCheck) { Test-CudaTorchInstalled $Python $build } else { Test-CudaTorchReady $Python $build }
                if ($ready) {
                    Write-Ok "NVIDIA CUDA PyTorch build '$($build.Name)' is usable."
                    return
                }
                Write-CudaTorchProbeDetails $Python
                throw "PyTorch CUDA build '$($build.Name)' installed but torch import/GPU verification failed. This usually means a Windows DLL, driver, or runtime compatibility issue."
            } catch {
                $lastError = $_.Exception.Message
                Write-Warn "PyTorch CUDA build $($build.Name) via $indexUrl failed: $lastError"
            }
        }
    }
    throw "Failed to install NVIDIA CUDA PyTorch with all indexes. Last error: $lastError"
}

function Verify-CudaPyTorch($Python, [bool]$RequireGpu) {
    $require = if ($RequireGpu) { "1" } else { "0" }
    $code = @"
import torch
print('torch=', torch.__version__)
print('cuda_runtime=', torch.version.cuda)
print('gpu_available=', torch.cuda.is_available())
print('device=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')
require_gpu = "$require" == "1"
ok = bool(torch.version.cuda) and (torch.cuda.is_available() or not require_gpu)
raise SystemExit(0 if ok else 1)
"@
    Invoke-External $Python @("-c", $code) "NVIDIA CUDA PyTorch verification failed"
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

function Install-ProjectDeps($Python, [bool]$UseMemoryVectorStore, [bool]$SkipTorchPackages) {
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

function Ensure-EnvConfig([string]$Device = "cpu", [string]$AsrEngine = "sensevoice_zh") {
    if (-not (Test-Path ".env")) {
        Copy-Item -LiteralPath ".env.example" -Destination ".env"
    }
    Set-EnvFileValue ".env" "ASR_ENGINE" $AsrEngine
    Set-EnvFileValue ".env" "ASR_DEVICE" $Device
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
}

function Ensure-ModelSettings([string]$Device = "cpu", [string]$AsrModel = "sensevoice_zh") {
    New-Item -ItemType Directory -Force -Path "config" | Out-Null
    $path = "config\model-settings.json"
    if (Test-Path -LiteralPath $path) {
        try {
            $settings = Get-Content -LiteralPath $path -Raw -Encoding UTF8 | ConvertFrom-Json
            Set-JsonProperty (Ensure-JsonObjectProperty $settings "asr") "device" $Device
            Set-JsonProperty (Ensure-JsonObjectProperty $settings "speaker") "device" $Device
            Set-JsonProperty (Ensure-JsonObjectProperty $settings "diarization") "device" $Device
            $settings | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $path -Encoding UTF8
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
`$candidateVenvs += @(".venv-nvidia-win", ".venv-win", ".venv-rocm-win")

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
    throw "Python environment not found. Run .\install-windows.cmd first, or pass -VenvPath .venv-nvidia-win."
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

try {
    Assert-Windows
    $ProjectRoot = Resolve-ProjectRoot
    Set-Location $ProjectRoot

    if ($Profile -eq "AmdRocm") {
        $args = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "scripts\install-windows-rocm.ps1")
        if ($PythonExe) { $args += @("-PythonExe", $PythonExe) }
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

        Write-Step "Checking Python 3.12"
        $PythonExe = Find-Python312 $PythonExe
        Write-Ok "Python: $PythonExe"

        Write-Step "Checking FFmpeg"
        Ensure-Ffmpeg

        Write-Step "Creating NVIDIA virtual environment"
        $VenvPython = Create-Or-RecreateVenv $PythonExe $VenvPath $RecreateVenv
        Write-Ok "Virtual environment: $VenvPython"

        Write-Step "Installing NVIDIA CUDA PyTorch"
        Invoke-PipInstallWithMirrors $VenvPython @("install", "-U", "pip", "wheel", "setuptools") "Failed to update pip tooling"
        Install-CudaTorch $VenvPython

        Write-Step "Installing project dependencies"
        Install-ProjectDeps $VenvPython $UseMemoryVectorStore $true

        Write-Step "Configuring local defaults for NVIDIA GPU"
        Ensure-EnvConfig "cuda" "qwen3"
        Ensure-ModelSettings "cuda" "qwen3"
        Write-StartScripts $VenvPython

        Write-Step "Building frontend"
        Ensure-FrontendBuild

        Write-Step "Verifying NVIDIA CUDA PyTorch"
        Verify-CudaPyTorch $VenvPython (-not $SkipGpuCheck)

        Write-Host ""
        Write-Host "Installation complete." -ForegroundColor Green
        Write-Host "Start command: powershell -NoProfile -ExecutionPolicy Bypass -File .\start-windows.ps1"
        Write-Host "Open: http://127.0.0.1:8000"

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

    Write-Step "Checking Python 3.12"
    $PythonExe = Find-Python312 $PythonExe
    Write-Ok "Python: $PythonExe"

    Write-Step "Checking FFmpeg"
    Ensure-Ffmpeg

    Write-Step "Creating virtual environment"
    $VenvPython = Create-Or-RecreateVenv $PythonExe $VenvPath $RecreateVenv
    Write-Ok "Virtual environment: $VenvPython"

    Write-Step "Installing Python dependencies"
    Install-ProjectDeps $VenvPython $UseMemoryVectorStore $false

    Write-Step "Configuring local defaults"
    Ensure-EnvConfig "cpu" "sensevoice_zh"
    Ensure-ModelSettings "cpu" "sensevoice_zh"
    Write-StartScripts $VenvPython

    Write-Step "Building frontend"
    Ensure-FrontendBuild

    Write-Host ""
    Write-Host "Installation complete." -ForegroundColor Green
    Write-Host "Start command: powershell -NoProfile -ExecutionPolicy Bypass -File .\start-windows.ps1"
    Write-Host "Open: http://127.0.0.1:8000"

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
