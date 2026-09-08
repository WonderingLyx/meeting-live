# AMD ROCm Deployment

This project can run ASR on an AMD GPU when PyTorch is installed with ROCm support. In PyTorch, ROCm still uses the `cuda` device API, so the project setting is:

```env
ASR_DEVICE=cuda
```

Do not set `ASR_DEVICE=rocm`. The UI will show the backend as ROCm when `torch.version.hip` is present.

## Recommended Target

For Ryzen AI Max+ 395 / Radeon 8060S machines, use Ubuntu 24.04.4 when possible. AMD's current compatibility matrix lists Radeon 8060S / Ryzen AI Max+ 395 as `gfx1151`.

Native Windows ROCm is listed by AMD for this hardware, but the Linux path is still the most predictable for this project because the Python, FFmpeg, FunASR, pyannote, and PyTorch wheels are easier to keep consistent.

## Windows 11 Preview Path

For AMD Windows systems, use the project helper script instead of the Linux commands below:

```powershell
.\scripts\install-windows-rocm.cmd
```

The script uses only the project-local Python runtime. It first reuses `.runtime\python-3.12\python.exe`; if that is missing, it extracts `offline\python\python.3.12.x.nupkg` into `.runtime\python-3.12`. In a Git clone where the offline asset is not present, it downloads `python.3.12.10.nupkg` into `.download-cache\python` and extracts it there. It does not scan system Python, Anaconda, `py -3.12`, or `PATH`. The `.venv-rocm-win` virtual environment is recreated by default on each AMD install to avoid stale failed installs.

The script is idempotent:

- ROCm and PyTorch wheel files are cached under `.download-cache\rocm-win-7.2.1`.
- The project-local Python runtime asset is cached under `.download-cache\python`.
- Python wheels that need VPN access can be bundled under `offline\wheels`; pip checks that directory before falling back to configured indexes.
- If ROCm PyTorch is already installed and verifies correctly, the large wheel install is skipped.
- If project Python dependencies are already importable, dependency installation is skipped.
- If `web\dist\index.html` already exists, the frontend build is skipped unless forced.
- Logs are written to `logs\install-windows-rocm-*.log`.

Useful options:

```powershell
.\scripts\install-windows-rocm.cmd -InstallBuildTools
.\scripts\install-windows-rocm.cmd
.\scripts\install-windows-rocm.cmd -ForceTorch
.\scripts\install-windows-rocm.cmd -ForceDeps
.\scripts\install-windows-rocm.cmd -StartServer
```

China mirror mode is enabled by default for Python/npm/model downloads:

```powershell
.\scripts\install-windows-rocm.cmd -MirrorMode China
```

This uses:

- PyPI: `https://pypi.tuna.tsinghua.edu.cn/simple`, then `https://mirrors.aliyun.com/pypi/simple`, then the official PyPI index as fallback.
- npm: `https://registry.npmmirror.com/`, then the official npm registry as fallback.
- Hugging Face: `HF_ENDPOINT=https://hf-mirror.com` written into `.env`.
- ModelScope models keep using ModelScope.

AMD's Windows ROCm 7.2.1 wheel artifacts do not currently have a verified public China mirror. The script supports private mirrors and always keeps AMD's official repository as fallback:

```powershell
.\scripts\install-windows-rocm.cmd `
  -RocmBaseUrls "https://your-mirror.example.com/rocm/windows/rocm-rel-7.2.1"
```

The mirror directory must expose the same filenames as AMD's repository, for example `rocm_sdk_core-7.2.1-py3-none-win_amd64.whl` and `torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl`.

If a network is unstable, rerun the same command. Valid downloads are reused from `.download-cache\rocm-win-7.2.1`, and incomplete `.part` files are discarded.

To build a Windows AMD package that refuses to ship without the ROCm cache and offline Python installer:

```powershell
.\package-windows.cmd -PackageName matrix-live-diarizer-windows-amd-api -Profile AmdRocm
```

Requirements:

- Windows 11.
- `offline\python\python.3.12.x.nupkg` inside the project when building an offline zip. Git clone installs can download this asset automatically unless `-OfflineOnly` is used.
- A supported AMD GPU / APU and matching AMD Software driver.
- Run from this project root.

After the script finishes:

```powershell
.\.venv-rocm-win\Scripts\python.exe main.py
```

Then open `http://127.0.0.1:8000/settings` and check that the ASR device panel shows ROCm and `ASR_DEVICE=cuda`.

### pyannote MIOpenDropoutHIP on Windows ROCm

If uploaded meeting diarization fails with:

```text
MIOpenDropoutHIP.cpp
fatal error: 'rocrand/rocrand_xorwow.h' file not found
miopenStatusUnknownError
```

This is a pyannote/ROCm inference path issue, not an ASR download issue. The project enables this workaround by default:

```env
PYANNOTE_ROCM_DISABLE_LSTM_DROPOUT=1
```

It disables recurrent-layer dropout inside pyannote on ROCm. Dropout is a training-time regularizer, so this does not change inference output. If you want the conservative fallback instead, set:

```env
PYANNOTE_DEVICE=cpu
```

ASR can still use the AMD GPU with `ASR_DEVICE=cuda` while pyannote diarization runs on CPU.

## 1. System Packages

```bash
sudo apt update
sudo apt install -y git ffmpeg python3.12 python3.12-venv python3-pip nodejs npm
```

Install the AMD GPU driver / ROCm stack following AMD's ROCm install guide for your exact OS and GPU. After installation, verify the GPU is visible:

```bash
rocminfo | grep -E "Name:|gfx"
```

For Radeon 8060S, expect `gfx1151`.

## 2. Create Python Environment

```bash
cd /opt
git clone <your-repo-url> matrix-live-diarizer
cd matrix-live-diarizer

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel setuptools
```

## 3. Install ROCm PyTorch For gfx1151

Follow AMD's latest PyTorch install page if the version changes. For the current AMD `gfx1151` wheel route:

```bash
python -m pip uninstall -y torch torchvision torchaudio pytorch-triton-rocm
python -m pip install "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.0.0/pytorch_triton_rocm-3.4.0%2Brocm7.0.0.gitf52c2b70-cp312-cp312-linux_x86_64.whl"
python -m pip install torch torchvision torchaudio --index-url "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.0.0/gfx1151/"
```

Verify PyTorch sees ROCm:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("hip:", torch.version.hip)
print("gpu available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY
```

The expected result is `hip` not empty and `gpu available: True`.

## 4. Install Project Dependencies

Install the remaining Python dependencies after ROCm PyTorch:

```bash
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip check
```

If `pip` tries to replace the ROCm PyTorch wheel, reinstall the ROCm PyTorch commands from step 3.

## 5. Build Frontend

```bash
cd web
npm install --registry=https://registry.npmmirror.com/
npm run build
cd ..
```

## 6. Configure `.env`

```bash
cp .env.example .env
```

Recommended AMD GPU ASR settings:

```env
HOST=127.0.0.1
PORT=8000
WORKERS=1
ASR_ENGINE=sensevoice_zh
ASR_DEVICE=cuda
ASR_FUNASR_ALLOW_ROCM_GPU=false
ASR_FUNASR_ALLOW_ROCM_PARAFORMER=false
ASR_FUNASR_ROCM_SAFE_VERSIONS=1.4.1
ASR_LOAD_TIMEOUT_SEC=3600
ASR_STARTUP_ALLOW_DOWNLOAD=false
ASR_STARTUP_FALLBACKS=sensevoice_zh,paraformer_full,paraformer
HF_HUB_DISABLE_XET=1
SPEAKER_ENGINE=campplus
```

`ASR_FUNASR_ALLOW_ROCM_GPU=false` means FunASR ASR/diarization models use CPU on Windows ROCm by default. `ASR_FUNASR_ALLOW_ROCM_PARAFORMER=false` only blocks Paraformer-family FunASR models when the installed FunASR version is outside `ASR_FUNASR_ROCM_SAFE_VERSIONS`. The 2026-08-13 AMD test log loaded the Paraformer/CAM++ stack successfully with `funasr==1.4.1`; `funasr==1.4.13` crashed after `ckpt:` with process exit `-1073741819` / `0xC0000005`. Rerun the installer to downgrade FunASR if the settings page reports this version guard.

If port `8000` is occupied, change `PORT` to another free port such as `8001`.
The Windows launcher reads `HOST`, `PORT`, and `ENABLE_HTTPS` from `.env` when opening the browser.

For pyannote diarization, also set `HF_TOKEN` after accepting the model terms on Hugging Face.

## 7. Start

```bash
source .venv/bin/activate
python main.py
```

Open:

```text
http://127.0.0.1:8000/settings
```

On the settings page, the ASR device area should show ROCm, the Radeon GPU name, and `ASR_DEVICE=cuda`. Use the CPU/GPU buttons there to switch devices without manually editing `.env`.
