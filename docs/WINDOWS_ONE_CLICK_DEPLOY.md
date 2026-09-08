# Windows 一键部署

本文面向新 Windows 机器部署。项目提供三个独立入口：`install-windows.cmd` 只安装 CPU 版，`install-windows-amd-gpu.cmd` 只安装 AMD ROCm 版，`install-windows-nvidia-gpu.cmd` 只安装 NVIDIA CUDA 版。

GPU 机器的兼容型号、驱动版本、完整 NVIDIA/AMD 命令和排错步骤见 [Windows GPU 部署指南](WINDOWS_GPU_DEPLOY.md)。

## 前置条件

必须安装：

1. Python 运行时。三个安装版本都只使用项目内 `.runtime\python-3.12`；没有时优先用 `offline\python\python.3.12.x.nupkg`，Git clone 场景缺少离线资产时会自动下载 `python.3.12.10.nupkg` 到 `.download-cache\python` 再解压。不扫描主机 Python、不修改系统 PATH。
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
- 安装 InsightFace + CPU ONNXRuntime 人脸签到依赖
- 创建 `.env`
- 创建 `config/model-settings.json`
- 构建 `web/dist`
- 生成 `start-windows.ps1`

启动：

```powershell
.\start-windows.cmd
```

启动失败时窗口会停住，并显示最新日志路径，例如：

```text
logs\start-windows-20260903-141808.log
```

常见启动失败原因：

- `.venv-rocm-win` / `.venv-win` 里缺依赖，例如 `ModuleNotFoundError: uvicorn`，或日志里的 `Using Python:` 指向了错误虚拟环境；重新运行对应安装脚本。
- `.env` 被记事本或其它工具保存成 Windows ANSI/GBK。新版本会兼容读取并提示，建议进入设置页保存一次，让它写回 UTF-8。
- `PORT` 被其它程序占用，修改 `.env` 的 `PORT` 后再启动。
- `config\model-settings.json` 保存了不可用模型。启动默认不下载 Qwen，会自动退到 `sensevoice_zh,paraformer_full,paraformer`；如果手动要求 Qwen，请先保证模型缓存或网络可访问。
- `DEPLOYMENT_MODE=lan/public` 但没有配置 `JWT_SECRET` 或可信 `ALLOWED_ORIGINS`。
- `ENABLE_HTTPS=true` 但 `data\ssl\selfsigned.crt` / `.key` 不存在。
- AMD GPU 驱动、ROCm PyTorch 或模型推理库初始化失败；详细 traceback 会写入 `logs\start-windows-*.log`。
- 退出码 `-1073741819` / `0xC0000005` 是 Windows native DLL 访问冲突，不是普通 Python 异常。旧日志显示 `funasr==1.4.1` 可加载同一套 Paraformer/CAM++ 模型，`funasr==1.4.13` 在 AMD Windows ROCm 包里加载同一权重后崩溃。新安装脚本会固定并修复到 `funasr==1.4.1`；如果设置页提示 FunASR 版本不稳，重新运行一键安装。

打开：

```text
http://127.0.0.1:8000
```

## 端口冲突

服务端口由 `.env` 控制：

```env
HOST=127.0.0.1
PORT=8000
```

如果 8000 被其它程序占用，改成空闲端口后重新启动即可：

```env
PORT=8001
```

`start-windows.ps1` 默认会读取 `.env` 的 `HOST`、`PORT`、`ENABLE_HTTPS` 来打开浏览器；`HOST=0.0.0.0` 时浏览器仍会打开本机 `127.0.0.1`。如果检测到端口已被本项目占用，会直接打开已有服务；如果被其它程序占用，会提示修改 `.env` 的 `PORT`。

## AMD ROCm GPU 部署

AMD Windows ROCm 路线推荐直接运行：

```powershell
.\install-windows-amd-gpu.cmd
```

它会安装 ROCm runtime / ROCm PyTorch，并默认把 ASR、声纹 embedding、说话人分离设备设为 `cuda`。在 AMD ROCm PyTorch 中，`cuda` 是 PyTorch 兼容设备名，底层实际走 HIP/ROCm。不要用 `install-windows.cmd` 安装 AMD 版。

AMD 包会默认写入 `ASR_FUNASR_ALLOW_ROCM_GPU=false`。这是为了规避 FunASR 在部分 Windows ROCm 机器上加载 Paraformer/SenseVoice 时的 native 崩溃；PyTorch ROCm 仍安装可用，非 FunASR 路径可以继续按设置页测试 GPU。

AMD 包还会默认写入 `ASR_FUNASR_ALLOW_ROCM_PARAFORMER=false` 和 `ASR_FUNASR_ROCM_SAFE_VERSIONS=1.4.1`。因此 `paraformer`、`paraformer_full`、`paraformer_large`、`paraformer_spk`、`paraformer_streaming` 在 FunASR 版本不是 1.4.1 时会被标记为不可用，避免点击切换后服务直接退出。安装脚本会把依赖修复到 1.4.1；要强制测试其它 FunASR 版本，把 `.env` 改为：

```env
ASR_FUNASR_ALLOW_ROCM_PARAFORMER=true
```

人脸签到在 AMD Windows 包中走 `onnxruntime-directml`，默认 `FACE_DEVICE=directml`。这与 ASR/说话人分离的 ROCm PyTorch 互不混装。

AMD 不使用主机 Python。zip 离线包内应有可解压 Python 3.12 x64 运行时：

```text
offline\python\python.3.12.x.nupkg
```

安装脚本会解压到项目相对目录 `.runtime\python-3.12`，不会写 PATH，不扫描 Anaconda/系统 Python，不影响主机其它 Python/conda 服务。通过 Git 拉代码时通常不会带这个离线文件，安装脚本会自动下载到 `.download-cache\python`；如果你加了 `-OfflineOnly`，则必须手动放入该文件。AMD 安装会默认重建 `.venv-rocm-win`。

ROCm wheel 默认从包内缓存读取：

```text
.download-cache\rocm-win-7.2.1
```

AMD 一键 zip 应在打包阶段内置这个目录；缺文件时打包会失败并列出缺失文件，避免新机器安装时重新慢速下载。临时联网安装时，下载中断后重复执行同一命令，脚本会复用 `.download-cache\rocm-win-7.2.1` 和 `.part` 文件继续下载。
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

制作内置 ROCm 的 AMD 一键包：

```powershell
.\package-windows.cmd -PackageName matrix-live-diarizer-windows-amd-api -Profile AmdRocm
```

需要 VPN 或外网下载的 Python wheel 可以放到 `offline\wheels`，安装器会优先用本地 wheel，再走清华/阿里 PyPI。

如果你有公司内网、NAS、Nginx 或对象存储镜像，把这些文件按原文件名放在同一目录，然后传自定义源：

```powershell
.\install-windows-amd-gpu.cmd -RocmBaseUrls "http://192.168.1.10/rocm-win-7.2.1"
```

## NVIDIA CUDA GPU 部署

NVIDIA Windows 路线推荐直接运行：

```powershell
.\install-windows-nvidia-gpu.cmd
```

它会创建 `.venv-nvidia-win`，安装 CUDA 版 PyTorch，并默认把 ASR、声纹 embedding、说话人分离设备设为 `cuda`。不要用 `install-windows.cmd` 安装 NVIDIA 版。

人脸签到在 NVIDIA 包中走 `onnxruntime-gpu`，默认 `FACE_DEVICE=cuda`。如果 ONNXRuntime CUDA provider 不可用，设置页的人脸识别测试会显示 provider 诊断信息。

前置要求是安装 NVIDIA 显卡驱动，并且新终端执行 `nvidia-smi` 能看到显卡。脚本默认使用 PyTorch CUDA 12.8 wheel 和 stable PyTorch 组合：

```powershell
.\install-windows-nvidia-gpu.cmd -CudaWheel cu128
```

也可以切换到 PyTorch 提供的其他 CUDA wheel：

```powershell
.\install-windows-nvidia-gpu.cmd -CudaWheel cu126
.\install-windows-nvidia-gpu.cmd -CudaWheel cu130
```

如果想测试最新 PyTorch 组合，可以手动切换；遇到 `WinError 1114`、`c10.dll` 或 `torch import` 阶段失败时仍回到 stable：

```powershell
.\install-windows-nvidia-gpu.cmd -TorchBuild latest
.\install-windows-nvidia-gpu.cmd -ForceTorch -TorchBuild stable
```

NVIDIA 安装也不使用 PATH 或主机 Python。包内必须有 `offline\python\python.3.12.x.nupkg`，脚本会解压到 `.runtime\python-3.12`。

PyTorch CUDA wheel 在 `MirrorMode=China` 时默认优先使用国内文件镜像：南京大学 `https://mirror.nju.edu.cn/pytorch/whl/<cu版本>`、阿里 `https://mirrors.aliyun.com/pytorch-wheels/<cu版本>`、上海交大 `https://mirror.sjtu.edu.cn/pytorch-wheels/<cu版本>`；普通依赖解析走清华 PyPI，失败后回退官方源 `https://download.pytorch.org/whl/<cu版本>`。公司内网或自建文件镜像可以这样传入：

```powershell
.\install-windows-nvidia-gpu.cmd -TorchFindLinks "http://192.168.1.10/pytorch-wheels/cu128"
```

如果已开始从慢源下载，保留 `.download-cache\nvidia-wheels\cu128\*.part`，直接中断后用更快源重跑即可断点续传：

```powershell
.\install-windows-nvidia-gpu.cmd -TorchFindLinks "https://mirror.nju.edu.cn/pytorch/whl/cu128" -InstallAria2
```

CUDA 版 PyTorch 大 wheel 会先进入 `.download-cache\nvidia-wheels\<cu版本>`，再从本地缓存安装。断网或 `IncompleteRead` 后重复运行同一命令即可续传，不会因为小依赖失败而重新下载完整 torch wheel。

日志里的 `win_amd64.whl` 是 Python/Windows x64 平台标签，适用于 Intel x64 和 AMD x64 CPU，不代表 AMD GPU 包。NVIDIA/AMD GPU 路线的真正区别是 PyTorch 使用 `+cu*` CUDA wheel 还是 `+rocm*` ROCm wheel，人脸签到使用 `onnxruntime-gpu` 还是 `onnxruntime-directml`。

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

上面这些命令只针对 CPU 入口。AMD/NVIDIA 请把脚本名分别换成 `install-windows-amd-gpu.cmd` 或 `install-windows-nvidia-gpu.cmd`。三个安装入口都会默认重建各自虚拟环境；`-RecreateVenv` 保留为兼容参数，不再需要手动传。

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
.\package-windows.cmd -PackageName matrix-live-diarizer-windows-cpu -Profile Cpu
.\package-windows.cmd -PackageName matrix-live-diarizer-windows-nvidia-api -Profile NvidiaCuda
.\package-windows.cmd -PackageName matrix-live-diarizer-windows-amd-api -Profile AmdRocm
```

产物在 `dist-packages\matrix-live-diarizer-windows-*.zip`。三个包都会带 `offline\python\python.3.12.x.nupkg`，目标机器安装时再解压到项目内 `.runtime\python-3.12`。CPU/NVIDIA 普通包默认不包含：

- `.env` 和 `config/model-settings.json`
- `.venv*` / `.conda`
- `models`
- `data`
- `logs`

AMD ROCm 包会要求 `.download-cache\rocm-win-7.2.1` 完整，并把 ROCm 7.2.1 wheel/tar.gz 一起打进 zip：

人脸识别依赖不内置到 zip 的 venv 中，目标机器首次安装时会从清华/阿里 PyPI 镜像安装；如果目标网络受限，可提前把 `insightface`、`onnxruntime`/`onnxruntime-gpu`/`onnxruntime-directml`、`opencv-python`、`onnx`、`Pillow`、`scikit-image` 等 wheel 放入 `offline\wheels`，安装脚本会优先使用本地 wheel。

新电脑解压后执行：

```powershell
.\install-windows.cmd
.\start-windows.cmd
```

AMD 机器改为执行 `.\install-windows-amd-gpu.cmd`；NVIDIA 机器改为执行 `.\install-windows-nvidia-gpu.cmd`。三者只选一个入口，不混用。

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
