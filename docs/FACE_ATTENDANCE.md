# 人脸识别签到

本功能用于会议签到辅助：先在“人物”页面为参会人上传单人正脸照片，会议进行中或会议详情页打开“人脸签到”摄像头，系统会用 InsightFace 提取人脸向量并与本地照片库匹配，匹配成功后写入会议签到记录。

## 模型与运行方式

- 引擎：InsightFace + ONNXRuntime。
- 默认模型：`buffalo_l`，精度优先；设置页也可切换 `buffalo_m`、`buffalo_s`。
- CPU 包：安装 `onnxruntime`，`FACE_DEVICE=cpu`。
- NVIDIA 包：安装 `onnxruntime-gpu`，`FACE_DEVICE=cuda`。
- AMD Windows 包：安装 `onnxruntime-directml`，`FACE_DEVICE=directml`。

InsightFace 模型包会在首次“测试加载”或首次上传人脸照片时自动下载到：

```text
models\face\insightface
```

## 使用步骤

1. 运行对应的一键安装脚本：`install-windows.cmd`、`install-windows-nvidia-gpu.cmd` 或 `install-windows-amd-gpu.cmd`。
2. 打开“设置 → 人脸识别签到”，确认 Provider 为 `InsightFace`，设备按机器类型选择 `cpu`、`cuda` 或 `directml`。
3. 点击“测试加载”。首次使用会下载 InsightFace 模型包。
4. 到“人物”页面，为每个人上传至少 1 张清晰单人正脸照片。
5. 实时会议中打开“人脸签到”摄像头，或会后在会议详情页进入“人脸签到”标签页。

## 常见错误

如果上传人脸样本时报：

```text
No module named 'cv2'
```

说明当前项目虚拟环境里没有安装 OpenCV。重新运行对应的一键安装脚本即可；安装器会验证 `import cv2`，失败会直接停止安装：

```powershell
.\install-windows.cmd -ForceDeps
```

也可以只修复当前 CPU venv：

```powershell
.\.venv-win\Scripts\python.exe -m pip uninstall -y opencv-python opencv-python-headless
.\.venv-win\Scripts\python.exe -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --prefer-binary opencv-python-headless==4.14.0.94
.\.venv-win\Scripts\python.exe -c "import cv2; print(cv2.__version__)"
```

如果继续报：

```text
No module named 'insightface'
```

直接运行项目内修复脚本，它会按当前 venv 自动补齐 InsightFace、ONNXRuntime 和 OpenCV：

```powershell
.\repair-face-runtime.cmd
```

也可以手动补 CPU venv：

```powershell
.\.venv-win\Scripts\python.exe -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --prefer-binary onnxruntime==1.29.0 numpy scipy onnx==1.22.0 Pillow==12.3.0 scikit-image==0.26.0 tqdm requests
.\.venv-win\Scripts\python.exe -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --prefer-binary insightface==1.0.1 --no-deps
.\.venv-win\Scripts\python.exe -c "import cv2; from insightface.app import FaceAnalysis; import onnxruntime as ort; print(ort.get_available_providers())"
```

## 调参建议

- `匹配阈值` 默认 `0.55`。误识别时调高到 `0.60` 到 `0.70`。
- `歧义间隔` 默认 `0.08`。多人长相接近或照片质量差时调高。
- `识别间隔秒` 默认 `5`。GPU 机器可以调低，CPU 机器建议维持默认。

人脸签到只用于会议出勤辅助，不应作为门禁、支付、身份认证或合规考勤的唯一依据。
