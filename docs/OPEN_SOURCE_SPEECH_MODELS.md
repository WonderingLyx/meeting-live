# 开源 ASR 与声纹模型候选

更新时间：2026-08-11。GitHub star 数通过 GitHub API 查询，数值会随时间变化。

## ASR 优先测试顺序

| 优先级 | 项目 / 模型 | 当前 star | 适合场景 | 本项目接入方式 | AMD Windows 注意点 |
| --- | --- | ---: | --- | --- | --- |
| 1 | [FunASR](https://github.com/modelscope/FunASR) / SenseVoice / Paraformer | 19,763 | 中文会议、访谈、普通话优先 | 已内置 `sensevoice`、`paraformer`、`paraformer_streaming`；也可用插件接 FunASR CLI | 中文生态最好，先测 CPU 稳定性，再测 ROCm |
| 2 | [Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) | HF 模型页 | 高质量中文/多语种离线转写 | 已内置 `qwen3` | 质量基线，但在 AMD Windows ROCm 上可能遇到算子/MIOpen 兼容问题 |
| 3 | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) + Whisper large-v3/turbo | 24,847 | 多语种对照、word timestamps、成熟社区 | 已提供 `scripts/asr_plugins/faster_whisper_cli.py` 插件包装 | GPU 主要面向 CUDA/CTranslate2；AMD Windows 先按 CPU 测 |
| 4 | [whisper.cpp](https://github.com/ggml-org/whisper.cpp) | 52,799 | 轻量 CLI、本地二进制、无 Python 依赖 | 用 `plugin:<id>` 接 whisper.cpp 命令或包装脚本 | CPU 很方便；Vulkan/其他后端可单独实验 |
| 5 | [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) | 14,087 | 离线/流式 ASR、移动端/边缘部署 | 先用命令插件接 CLI，稳定后再做原生适配 | ONNX 路线适合部署测试，中文模型选择要单独评测 |
| 6 | [PaddleSpeech](https://github.com/PaddlePaddle/PaddleSpeech) | 12,662 | 中文语音工具链、研究和工程实验 | 命令插件或独立服务 | 依赖较重，Windows 环境要先验证安装 |
| 7 | [WeNet](https://github.com/wenet-e2e/wenet) | 5,221 | 工程化中文 ASR、生产式实验 | 命令插件或独立服务 | 更适合深入改造，不适合最快速试用 |
| 8 | [ESPnet](https://github.com/espnet/espnet) | 9,917 | 学术/多模型基准 | 命令插件或独立服务 | 依赖重，不建议作为第一批会议转写方案 |
| 9 | [NVIDIA NeMo Speech](https://github.com/NVIDIA-NeMo/Speech) | 18,086 | NVIDIA CUDA 环境、研究模型对照 | 命令插件或独立服务 | 不适合你的 AMD Windows 作为主路线 |

如果目标是中文会议，推荐先按这个顺序测：`sensevoice`、`paraformer`、`qwen3`、`plugin:faster_whisper_large_v3`、sherpa-onnx 中文模型。不要同时改 ASR 和声纹模型，否则很难判断问题来自哪一侧。

## 声纹 / 说话人识别候选

| 优先级 | 项目 / 模型 | 当前 star | 适合场景 | 本项目接入方式 | 注意点 |
| --- | --- | ---: | --- | --- | --- |
| 1 | [3D-Speaker](https://github.com/modelscope/3D-Speaker) CAM++ | 3,096 | 中文声纹、实时说话人匹配 | 已内置 `campplus` | 当前最适合作默认声纹模型 |
| 2 | [3D-Speaker](https://github.com/modelscope/3D-Speaker) ERes2NetV2 | 3,096 | 更高精度的中文声纹 embedding | 已内置 `eres2net` | 离线或更强机器优先测 |
| 3 | [WeSpeaker](https://github.com/wenet-e2e/wespeaker) | 1,378 | 声纹验证、说话人识别、中文生态 | 已内置/可扩展 `wespeaker` | 值得作为第三个中文声纹基线 |
| 4 | [SpeechBrain](https://github.com/speechbrain/speechbrain) ECAPA-TDNN | 11,745 | 通用 speaker verification 基线 | 后续用声纹插件或独立包装脚本接入 | VoxCeleb 偏英文/公开视频，中文会议要实测 |
| 5 | [pyannote-audio](https://github.com/pyannote/pyannote-audio) embedding / diarization | 10,403 | 完整说话人分离、分段、embedding | 已有 pyannote 离线路径，可继续增强 | HF gated 模型需要授权；AMD Windows ROCm 可能遇到 MIOpen 问题 |
| 6 | [NVIDIA NeMo Speech](https://github.com/NVIDIA-NeMo/Speech) TitaNet / diarization | 18,086 | NVIDIA CUDA 声纹/分离对照 | 命令插件或独立服务 | NVIDIA 路线，不建议作为你的 AMD Windows 主路线 |
| 7 | [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) speaker models | 14,087 | 离线 speaker identification / verification | 后续可加声纹插件 | 适合部署型验证 |

这里的“声纹分类”一般不是 ASR 模型直接输出人名，而是：先注册人员声音样本，模型提取 embedding，再用相似度把新语音匹配到已注册人员。多说话人“分离/分段”则是 diarization，通常需要 pyannote 或 sherpa-onnx 这类独立模块。

## 推荐评测表

| 目标 | 第一选择 | 对照选择 |
| --- | --- | --- |
| 中文会议准确率 | `qwen3`、`sensevoice`、`paraformer` | faster-whisper large-v3 |
| 本地速度 | `sensevoice`、`paraformer` | whisper.cpp、sherpa-onnx |
| 字/词级时间戳 | `qwen3` 开启 forced aligner、faster-whisper | whisper.cpp JSON 输出 |
| 实时字幕 | `paraformer_streaming` 当前仍按 VAD segment 调用 | sherpa-onnx streaming |
| 说话人匹配 | `campplus` | `eres2net`、`wespeaker` |
| 完整多人分离 | pyannote | sherpa-onnx diarization |

## 已记录的 star 快照

| 仓库 | star | license | 最近 push |
| --- | ---: | --- | --- |
| [openai/whisper](https://github.com/openai/whisper) | 107,039 | MIT | 2026-07-28 |
| [ggml-org/whisper.cpp](https://github.com/ggml-org/whisper.cpp) | 52,799 | MIT | 2026-08-07 |
| [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) | 24,847 | MIT | 2025-11-19 |
| [modelscope/FunASR](https://github.com/modelscope/FunASR) | 19,763 | MIT | 2026-08-05 |
| [NVIDIA-NeMo/Speech](https://github.com/NVIDIA-NeMo/Speech) | 18,086 | Apache-2.0 | 2026-08-11 |
| [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) | 14,087 | Apache-2.0 | 2026-08-11 |
| [PaddlePaddle/PaddleSpeech](https://github.com/PaddlePaddle/PaddleSpeech) | 12,662 | Apache-2.0 | 2026-08-02 |
| [speechbrain/speechbrain](https://github.com/speechbrain/speechbrain) | 11,745 | Apache-2.0 | 2026-06-15 |
| [pyannote/pyannote-audio](https://github.com/pyannote/pyannote-audio) | 10,403 | MIT | 2026-08-04 |
| [espnet/espnet](https://github.com/espnet/espnet) | 9,917 | Apache-2.0 | 2026-08-10 |
| [QwenAudio/SenseVoice](https://github.com/QwenAudio/SenseVoice) | 9,048 | MIT | 2026-07-27 |
| [wenet-e2e/wenet](https://github.com/wenet-e2e/wenet) | 5,221 | Apache-2.0 | 2026-06-15 |
| [modelscope/3D-Speaker](https://github.com/modelscope/3D-Speaker) | 3,096 | Apache-2.0 | 2025-12-08 |
| [wenet-e2e/wespeaker](https://github.com/wenet-e2e/wespeaker) | 1,378 | Apache-2.0 | 2026-07-08 |
