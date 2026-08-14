# Windows 一键部署

本文面向新 Windows 机器部署。默认路线是 CPU + 中文 FunASR/ModelScope 模型，优先保证可安装、可启动、可切换模型；AMD ROCm GPU 和 NVIDIA CUDA GPU 路线作为可选 profile。

GPU 机器的兼容型号、驱动版本、完整 NVIDIA/AMD 命令和排错步骤见 [Windows GPU 部署指南](WINDOWS_GPU_DEPLOY.md)。

## 前置条件

必须安装：

1. Python 3.12 x64，并勾选 Add Python to PATH，或安装后使用 `-PythonExe` 指定路径。
2. Node.js LTS，用于构建前端。
3. FFmpeg，建议安装后确认新终端执行 `ffmpeg -version` 成功。

普通 CPU 部署不要求 Visual Studio Build Tools。脚本默认跳过 ChromaDB，使用项目内置的进程内内存向量库，避免 `chroma-hnswlib` 在 Windows 上触发 C++ 编译失败。

脚本默认也跳过 pyannote。中文会议说话人分离默认使用 FunASR/CAM++，不需要 Hugging Face gated 授权；需要 pyannote 对照测试时再单独安装。

## 一键安装

在项目根目录执行：

```powershell
.\install-windows.cmd
```

默认会做这些事：

- 创建 `.venv-win`
- 使用国内 PyPI/npm 镜像，失败时回退官方源
- 安装 CPU PyTorch 和项目依赖
- 创建 `.env`
- 创建 `config/model-settings.json`
- 构建 `web/dist`
- 生成 `start-windows.ps1`

启动：

```powershell
.\start-windows.cmd
```

打开：

```text
http://127.0.0.1:8000
```

## AMD ROCm GPU 部署

AMD Windows ROCm 路线推荐直接运行：

```powershell
.\install-windows-amd-gpu.cmd
```

它等价于下面的命令，会安装 ROCm runtime / ROCm PyTorch，并默认把 ASR、声纹 embedding、说话人分离设备设为 `cuda`。在 AMD ROCm PyTorch 中，`cuda` 是 PyTorch 兼容设备名，底层实际走 HIP/ROCm。

```powershell
.\install-windows.cmd -Profile AmdRocm
```

如果 Python 3.12 没在 PATH：

```powershell
.\install-windows.cmd -Profile AmdRocm -PythonExe "C:\Users\you\AppData\Local\Programs\Python\Python312\python.exe"
```

ROCm wheel 目前没有已验证的公开国内镜像。下载中断后重复执行同一命令，脚本会复用 `.download-cache\rocm-win-7.2.1` 和 `.part` 文件继续下载。
脚本会拒绝 `text/html` 或明显过小的镜像响应，避免把镜像站错误页当成 wheel 缓存；aria2 持续断流或低速时会失败退出并保留断点文件，不会无限卡住。
如果网络虽然慢但仍在稳定下载，可以关闭低速失败阈值：

```powershell
.\install-windows-amd-gpu.cmd -Aria2LowestSpeedLimit 0
```

推荐安装 aria2 后多线程断点续传：

```powershell
.\install-windows-amd-gpu.cmd -InstallAria2
```

如果已经有 `aria2c.exe`：

```powershell
.\install-windows-amd-gpu.cmd -Aria2cExe "D:\tools\aria2\aria2c.exe" -Aria2Connections 16
```

如果有代理：

```powershell
.\install-windows-amd-gpu.cmd -Proxy "http://127.0.0.1:7890"
```

如果想用 IDM、浏览器或另一台网络更快的机器下载：

```powershell
.\install-windows-amd-gpu.cmd -PrintRocmUrls
```

把打印出的 wheel/tar.gz 文件放到 `.download-cache\rocm-win-7.2.1`，再重新运行 `.\install-windows-amd-gpu.cmd`。脚本会优先复用缓存，不会重复下载完整文件。

如果你有公司内网、NAS、Nginx 或对象存储镜像，把这些文件按原文件名放在同一目录，然后传自定义源：

```powershell
.\install-windows-amd-gpu.cmd -RocmBaseUrls "http://192.168.1.10/rocm-win-7.2.1"
```

## NVIDIA CUDA GPU 部署

NVIDIA Windows 路线推荐直接运行：

```powershell
.\install-windows-nvidia-gpu.cmd
```

它等价于下面的命令，会创建 `.venv-nvidia-win`，安装 CUDA 版 PyTorch，并默认把 ASR、声纹 embedding、说话人分离设备设为 `cuda`：

```powershell
.\install-windows.cmd -Profile NvidiaCuda
```

前置要求是安装 NVIDIA 显卡驱动，并且新终端执行 `nvidia-smi` 能看到显卡。脚本默认使用 PyTorch CUDA 12.8 wheel：

```powershell
.\install-windows-nvidia-gpu.cmd -CudaWheel cu128
```

也可以切换到 PyTorch 提供的其他 CUDA wheel：

```powershell
.\install-windows-nvidia-gpu.cmd -CudaWheel cu126
.\install-windows-nvidia-gpu.cmd -CudaWheel cu130
```

如果遇到 `WinError 1114`、`c10.dll` 或 `torch import` 阶段失败，直接用稳定 PyTorch 组合重装：

```powershell
.\install-windows-nvidia-gpu.cmd -ForceTorch -TorchBuild stable
```

如果 Python 3.12 没在 PATH：

```powershell
.\install-windows-nvidia-gpu.cmd -PythonExe "C:\Users\you\AppData\Local\Programs\Python\Python312\python.exe"
```

PyTorch CUDA wheel 在 `MirrorMode=China` 时默认优先使用阿里文件镜像 `https://mirrors.aliyun.com/pytorch-wheels/<cu版本>`，普通依赖解析走清华 PyPI，失败后回退官方源 `https://download.pytorch.org/whl/<cu版本>`。公司内网或自建文件镜像可以这样传入：

```powershell
.\install-windows-nvidia-gpu.cmd -TorchFindLinks "http://192.168.1.10/pytorch-wheels/cu128"
```

如果只是准备离线包、当前机器没有 NVIDIA 显卡，可以跳过显卡探测：

```powershell
.\install-windows-nvidia-gpu.cmd -SkipGpuCheck
```

## 重新安装和常用参数

```powershell
.\install-windows.cmd -RecreateVenv
.\install-windows.cmd -ForceDeps
.\install-windows.cmd -ForceFrontendBuild
.\install-windows.cmd -MirrorMode Official
.\install-windows.cmd -StartServer
```

如果你已经安装了 Microsoft C++ Build Tools，且想使用 ChromaDB：

```powershell
.\install-windows.cmd -InstallChroma
```

如果你还要测试 pyannote：

```powershell
.\install-windows.cmd -InstallPyannote
```

如果 `web/dist` 已存在且新机器没有 Node.js：

```powershell
.\install-windows.cmd -SkipFrontendBuild
```

## 打包给新电脑

在已拉取代码的机器上执行：

```powershell
.\package-windows.cmd
```

产物在 `dist-packages\matrix-live-diarizer-windows-*.zip`。包内不包含：

- `.env` 和 `config/model-settings.json`
- `.venv*` / `.conda`
- `models`
- `data`
- `logs`
- `.download-cache`

新电脑解压后执行：

```powershell
.\install-windows.cmd
.\start-windows.cmd
```

## 推荐默认模型

中文会议建议先用：

- ASR: `sensevoice_zh`
- 声纹: `campplus`
- 说话人分离: `funasr_campplus`
- 向量库: `SPEAKER_VECTOR_STORE=memory`

需要测试更多说话人分离模型时，在设置页切换：

- `funasr_sensevoice_campplus`
- `funasr_campplus_cn_en`
- `funasr_paraformer_large_campplus`
- `funasr_eres2netv2`，实验项，先用“测试加载”确认本机 FunASR 支持

## 排错

- 仍然加载 pyannote：检查 `config/model-settings.json` 的 `diarization.engine` 是否是 `funasr_campplus`，改完后重启服务。
- pip 下载慢：默认普通依赖使用清华/阿里 PyPI，NVIDIA CUDA wheel 使用阿里 PyTorch wheel 镜像，可重跑安装脚本。
- ChromaDB 编译失败：普通 Windows 部署保持 `SPEAKER_VECTOR_STORE=memory`，不需要安装 Visual Studio Build Tools。
- 前端打不开：确认 `web/dist/index.html` 存在；不存在时运行 `.\install-windows.cmd -ForceFrontendBuild`。
