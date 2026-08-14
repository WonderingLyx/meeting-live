# Windows GPU 部署指南

本文只覆盖 Windows 本地部署，分为 NVIDIA CUDA 和 AMD ROCm 两条路线。两条路线都会把项目默认设备写为 `cuda`；这是 PyTorch 的统一设备名，NVIDIA 底层走 CUDA，AMD ROCm 底层走 HIP/ROCm。

## 选择路线

| 机器 | 推荐分支 | 一键入口 | 虚拟环境 | PyTorch |
| --- | --- | --- | --- | --- |
| NVIDIA GPU | `nvidia-api` | `install-windows-nvidia-gpu.cmd` | `.venv-nvidia-win` | `torch==2.11.0` CUDA wheel |
| AMD Radeon / Ryzen AI | `amd-api` 或包含 AMD 脚本的 `nvidia-api` | `install-windows-amd-gpu.cmd` | `.venv-rocm-win` | `torch==2.9.1+rocm7.2.1` |

如果只是普通 CPU 机器，运行 `install-windows.cmd`，不要使用 GPU 入口。

## 通用前置条件

必须安装：

1. Windows 10/11 x64。
2. Python 3.12 x64。安装时勾选 `Add Python to PATH`，否则用 `-PythonExe` 指定路径。
3. Node.js LTS，用于构建前端。
4. FFmpeg。新开 PowerShell 后执行 `ffmpeg -version` 能正常输出。
5. 显卡驱动。NVIDIA 用 `nvidia-smi` 验证；AMD 按下文安装指定 Adrenalin/PRO 驱动。

建议配置：

| 项目 | 建议 |
| --- | --- |
| 内存 | 16 GB 起，长会议建议 32 GB+ |
| 显存 | 8 GB 起可测试，Qwen ASR / 大型分离模型建议 12 GB+ |
| 磁盘 | 预留 20 GB+，首次下载模型和 wheel 会占用空间 |
| 网络 | 首次安装需要访问 PyPI、npm、ModelScope、PyTorch 或 AMD 官方源 |

## 获取代码或 zip

从仓库拉分支：

```powershell
git clone git@github.com:WonderingLyx/meeting-live.git
cd meeting-live
git switch nvidia-api
```

已有仓库：

```powershell
git fetch origin
git switch nvidia-api
git pull
```

如果使用打包产物，直接解压对应 zip：

```text
dist-packages\matrix-live-diarizer-windows-nvidia-api.zip
dist-packages\matrix-live-diarizer-windows-amd-api.zip
```

解压后进入项目根目录执行后续安装命令。

## NVIDIA CUDA 部署

### 兼容机器

满足下面条件即可走 NVIDIA 路线：

- Windows x64。
- NVIDIA CUDA-capable GPU。
- 安装 NVIDIA 驱动后，PowerShell 执行 `nvidia-smi` 能看到显卡。
- Python 3.12 x64。

推荐显卡范围：

- GeForce RTX 20/30/40/50 系列。
- RTX A 系列、Quadro RTX、T 系列、Tesla / Data Center GPU。
- 其他 CUDA-capable GPU 可以尝试，但旧显卡可能因为驱动或 PyTorch wheel 架构支持不足而失败。

### CUDA wheel 和驱动

脚本支持三个 PyTorch CUDA wheel：

| 参数 | PyTorch 源 | 建议驱动 |
| --- | --- | --- |
| `cu126` | `https://download.pytorch.org/whl/cu126` | Windows 驱动建议 528.33+，更稳妥是安装最新 Studio/Game Ready/Data Center 驱动 |
| `cu128` | `https://download.pytorch.org/whl/cu128` | Windows 驱动建议 572.61+，这是默认选项 |
| `cu130` | `https://download.pytorch.org/whl/cu130` | 建议 580+ 驱动，只在新驱动和新显卡上优先测试 |

默认用 `cu128`，因为它在兼容性和新模型性能之间比较均衡。

### 一键安装

默认安装：

```powershell
.\install-windows-nvidia-gpu.cmd
```

等价于：

```powershell
.\install-windows.cmd -Profile NvidiaCuda
```

指定 Python 3.12：

```powershell
.\install-windows-nvidia-gpu.cmd -PythonExe "C:\Users\you\AppData\Local\Programs\Python\Python312\python.exe"
```

切换 CUDA wheel：

```powershell
.\install-windows-nvidia-gpu.cmd -CudaWheel cu126
.\install-windows-nvidia-gpu.cmd -CudaWheel cu128
.\install-windows-nvidia-gpu.cmd -CudaWheel cu130
```

使用自建 PyTorch wheel 镜像：

```powershell
.\install-windows-nvidia-gpu.cmd -TorchIndexUrls "http://192.168.1.10/pytorch-wheels/cu128"
```

当前机器没有 NVIDIA GPU、只是准备离线包时可跳过显卡探测：

```powershell
.\install-windows-nvidia-gpu.cmd -SkipGpuCheck
```

安装完成后启动：

```powershell
.\start-windows.cmd
```

浏览器打开：

```text
http://127.0.0.1:8000
```

### NVIDIA 安装后验证

确认项目使用的 Python：

```powershell
.\.venv-nvidia-win\Scripts\python.exe -c "import sys; print(sys.executable)"
```

确认 PyTorch 走 CUDA：

```powershell
.\.venv-nvidia-win\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

正常结果应包含：

- `torch` 版本带 `+cu126` / `+cu128` / `+cu130`。
- `torch.version.cuda` 不为空。
- `torch.cuda.is_available()` 为 `True`。
- 能输出 NVIDIA 显卡名。

## AMD ROCm 部署

### 兼容机器

AMD Windows ROCm 7.2.1 官方兼容范围分为 Radeon 独显和 Ryzen AI APU。

Radeon 独显：

| ROCm | GPU 架构 | 官方列出的硬件 |
| --- | --- | --- |
| 7.2.1 | `gfx1201`, `gfx1200`, `gfx1100`, `gfx1101` | Radeon RX 9070, RX 9070 XT, Radeon AI PRO R9700, RX 9060 XT, RX 7900 XTX, Radeon PRO W7900, Radeon PRO W7900 Dual Slot, Radeon RX 7700 |

Ryzen AI APU：

| ROCm | GPU 架构 | 官方列出的硬件 |
| --- | --- | --- |
| 7.2.1 | `gfx1150`, `gfx1151` | Ryzen AI Max+ 395, Ryzen AI Max 390, Ryzen AI Max 385, Ryzen AI 9 HX 375, HX 370, AI 9 365, HX 475, HX 470, AI 9 465 |

你的 Ryzen AI Max+ 395 / Radeon 8060S 这类机器按官方矩阵属于 Ryzen AI Max+ 395 路线，走 `gfx1151`。如果机器只显示集显型号而没有处理器完整型号，优先用 AMD Adrenalin、设备管理器或厂商规格页确认 CPU/APU 型号。

### AMD 驱动和 Python 要求

AMD Windows ROCm PyTorch 7.2.1 要求：

- Python 3.12。
- AMD 26.2.2 图形驱动。
- 官方 ROCm 7.2.1 Windows wheel。

注意：AMD 官方说明里也强调 Windows 上不是完整 ROCm stack，本项目只依赖 AMD 发布的 Windows ROCm PyTorch wheel，不要求单独安装 Linux 那种完整 ROCm 环境。

### 一键安装

默认安装：

```powershell
.\install-windows-amd-gpu.cmd
```

等价于：

```powershell
.\install-windows.cmd -Profile AmdRocm
```

指定 Python 3.12：

```powershell
.\install-windows-amd-gpu.cmd -PythonExe "C:\Users\you\AppData\Local\Programs\Python\Python312\python.exe"
```

重建环境：

```powershell
.\install-windows-amd-gpu.cmd -RecreateVenv
```

强制重装 ROCm PyTorch：

```powershell
.\install-windows-amd-gpu.cmd -ForceTorch
```

安装完成后启动：

```powershell
.\start-windows.cmd
```

浏览器打开：

```text
http://127.0.0.1:8000
```

### AMD 下载加速和断点续传

ROCm Windows wheel 体积很大，目前没有已验证可用的公开国内镜像。脚本默认会复用：

```text
.download-cache\rocm-win-7.2.1
```

如果下载中断，直接重新执行同一条安装命令即可。

安装 aria2，多线程断点续传：

```powershell
.\install-windows-amd-gpu.cmd -InstallAria2
```

使用已有 aria2：

```powershell
.\install-windows-amd-gpu.cmd -Aria2cExe "D:\tools\aria2\aria2c.exe" -Aria2Connections 16
```

网络很慢但稳定时关闭低速失败阈值：

```powershell
.\install-windows-amd-gpu.cmd -Aria2LowestSpeedLimit 0
```

使用代理：

```powershell
.\install-windows-amd-gpu.cmd -Proxy "http://127.0.0.1:7890"
```

打印需要离线下载的 ROCm 文件：

```powershell
.\install-windows-amd-gpu.cmd -PrintRocmUrls
```

把打印出的 wheel/tar.gz 放到 `.download-cache\rocm-win-7.2.1`，再重新运行安装命令。

使用公司内网、NAS、Nginx 或对象存储自建镜像：

```powershell
.\install-windows-amd-gpu.cmd -RocmBaseUrls "http://192.168.1.10/rocm-win-7.2.1"
```

镜像目录必须保留 AMD 官方文件名，例如：

```text
rocm_sdk_core-7.2.1-py3-none-win_amd64.whl
rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl
rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl
rocm-7.2.1.tar.gz
torch-2.9.1+rocm7.2.1-cp312-cp312-win_amd64.whl
torchaudio-2.9.1+rocm7.2.1-cp312-cp312-win_amd64.whl
torchvision-0.24.1+rocm7.2.1-cp312-cp312-win_amd64.whl
```

### AMD 安装后验证

确认项目使用的 Python：

```powershell
.\.venv-rocm-win\Scripts\python.exe -c "import sys; print(sys.executable)"
```

确认 PyTorch 走 ROCm：

```powershell
.\.venv-rocm-win\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.version.hip); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

正常结果应包含：

- `torch` 版本带 `+rocm7.2.1`。
- `torch.version.hip` 不为空。
- `torch.cuda.is_available()` 为 `True`。
- 能输出 AMD GPU/APU 名称。

## 模型默认配置

GPU 安装完成后脚本会写入：

```dotenv
ASR_DEVICE=cuda
SPEAKER_DEVICE=cuda
PYANNOTE_DEVICE=cuda
SPEAKER_VECTOR_STORE=memory
DIARIZATION_ENGINE=funasr_campplus
DIARIZATION_PROVIDER=modelscope
```

同时会更新或创建：

```text
config\model-settings.json
```

设置页面里 ASR、声纹、说话人分离都可以继续切换模型。切换 CPU/GPU 后建议重启服务，确保已经加载的模型被释放。

## 常用维护命令

重新构建前端：

```powershell
.\install-windows.cmd -ForceFrontendBuild
```

重新安装 Python 依赖：

```powershell
.\install-windows.cmd -ForceDeps
```

删除并重建虚拟环境：

```powershell
.\install-windows-nvidia-gpu.cmd -RecreateVenv
.\install-windows-amd-gpu.cmd -RecreateVenv
```

不自动打开浏览器：

```powershell
.\start-windows.cmd -NoBrowser
```

指定启动虚拟环境：

```powershell
.\start-windows.cmd -VenvPath .venv-nvidia-win
.\start-windows.cmd -VenvPath .venv-rocm-win
```

## 打包给新电脑

在开发机或已经拉好代码的机器上生成 zip：

```powershell
.\package-windows.cmd
```

指定文件名：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\package-windows.ps1 -PackageName matrix-live-diarizer-windows-nvidia-api
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\package-windows.ps1 -PackageName matrix-live-diarizer-windows-amd-api
```

压缩包会排除：

- `.env`
- `config\model-settings.json`
- `.venv*`
- `web\node_modules`
- `models`
- `data`
- `logs`
- `.download-cache`

新电脑解压后重新运行对应一键安装脚本即可。

## 排错

### NVIDIA: `torch.cuda.is_available()` 是 `False`

按顺序检查：

```powershell
nvidia-smi
.\.venv-nvidia-win\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

处理：

- `nvidia-smi` 不存在或报错：重装 NVIDIA 驱动。
- `torch` 没有 `+cu...`：当前环境被 CPU 版 PyTorch 覆盖，运行 `.\install-windows-nvidia-gpu.cmd -ForceTorch`。
- 驱动太旧：升级驱动，或从 `cu128/cu130` 降到 `cu126`。

### AMD: ROCm 下载慢或反复下载

先看缓存目录：

```powershell
Get-ChildItem .\.download-cache\rocm-win-7.2.1
```

处理：

- 如果存在 `.part`，直接重跑安装命令继续断点续传。
- 如果下载到很小的 `.whl`，可能是镜像错误页，删除后重跑。
- 优先用 `-InstallAria2` 或 `-Proxy`。

### AMD: MIOpen / rocrand 编译错误

处理：

- 确认 AMD 驱动是 ROCm Windows PyTorch 要求的 26.2.2 或更新的兼容版本。
- 先使用默认 FunASR/CAM++ 分离，不要切到 pyannote。
- 如果只想让 ASR 跑 GPU，可以在设置页或 `.env` 里把 `PYANNOTE_DEVICE=cpu`，保留 `ASR_DEVICE=cuda`。

### 安装依赖时要 Visual Studio Build Tools

默认脚本会跳过 ChromaDB 和 pyannote，使用内置内存向量库，不需要 Visual Studio Build Tools。只有主动传下面参数时，才可能触发 Windows C++ 编译：

```powershell
.\install-windows.cmd -InstallChroma
.\install-windows.cmd -InstallPyannote
```

普通部署不要传这两个参数。

## 官方参考

- AMD Radeon Windows ROCm 兼容矩阵：<https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/windows/windows_compatibility.html>
- AMD Ryzen Windows ROCm 兼容矩阵：<https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityryz/windows/windows_compatibility.html>
- AMD Windows ROCm PyTorch 安装：<https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/windows/install-pytorch.html>
- PyTorch 安装选择器：<https://pytorch.org/get-started/locally/>
- PyTorch wheel 索引：<https://download.pytorch.org/whl/>
- NVIDIA CUDA Toolkit Release Notes：<https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html>
- NVIDIA CUDA Compatibility Guide：<https://docs.nvidia.com/deploy/cuda-compatibility/>
