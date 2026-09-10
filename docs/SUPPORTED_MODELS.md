# 当前内置模型列表

本文只列当前代码已经内置、可在设置页选择并自动下载/加载的 ASR 与声纹模型。更多候选模型和 star 快照见 `docs/OPEN_SOURCE_SPEECH_MODELS.md`。

## ASR 模型

| 设置值 | 显示名 | 上游模型 | 语言/场景 | 依赖 | 备注 |
| --- | --- | --- | --- | --- | --- |
| `sensevoice_zh` | SenseVoice-Small Chinese | `iic/SenseVoiceSmall` | 中文/粤语会议快速转写 | `funasr` | 中文固定语言，中文会议优先测试 |
| `sensevoice` | SenseVoice-Small | `iic/SenseVoiceSmall` | 中文/英语/粤语/日语/韩语 | `funasr` | 多语种轻量对照 |
| `sensevoice_spk` | SenseVoice-Small + Speaker Tags | `iic/SenseVoiceSmall + fsmn-vad + cam++` | 中文快速转写 + 匿名段级说话人标签 | `funasr` | 只作对照，不替代项目声纹/分离 |
| `paraformer` | Paraformer | `paraformer-zh` | 中文会议/访谈离线转写 | `funasr` | 稳定中文基线 |
| `paraformer_full` | Paraformer + VAD + Punc | `paraformer-zh + fsmn-vad + ct-punc` | 较长中文会议、自动断句标点 | `funasr` | 首次会下载 ASR、VAD、标点模型 |
| `seaco_paraformer` | SeACo Paraformer Hotword | `iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch` | 人名、部门名、项目名较多的中文会议 | `funasr` | 用 `ASR_FUNASR_HOTWORDS` 配热词 |
| `paraformer_large` | Paraformer Large | `iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch` | 高质量普通话离线转写 | `funasr` | 适合质量对照 |
| `paraformer_spk` | Paraformer + Speaker Tags | `paraformer-zh + fsmn-vad + ct-punc + cam++` | 带 FunASR 内部 speaker 标签的转写 | `funasr` | 只作对照，不替代项目声纹/pyannote |
| `paraformer_online` | Paraformer Online | `iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online` | 中文实时低延迟对照 | `funasr` | 当前适配层仍按 VAD 段调用 |
| `paraformer_streaming` | Paraformer Streaming | `paraformer-zh-streaming` | 低延迟中文实时字幕 | `funasr` | 当前适配层仍按 VAD 段调用 |
| `qwen3` | Qwen3-ASR | `Qwen/Qwen3-ASR-0.6B` | 高质量中文/多语种离线转写 | `qwen_asr` 等 | 可选 `ASR_WORD_TIMESTAMPS=true` 启用 forced aligner |
| `plugin:<id>` | 外部 ASR 插件 | 由 `config/asr_plugins*.json` 定义 | 自定义 | 自定义 | 用于接入 faster-whisper、whisper.cpp、sherpa-onnx 等 |

## 声纹模型

| 设置值 | 显示名 | 上游模型 | 参数/维度 | 推荐场景 | 备注 |
| --- | --- | --- | --- | --- | --- |
| `campplus` | CamPlus | `damo/speech_campplus_sv_zh-cn_16k-common` | 7.2M / 192d | 默认实时中文声纹 | 速度快，适合实时 |
| `campplus_cn_en` | CamPlus Chinese-English | `iic/speech_campplus_sv_zh_en_16k-common_advanced` | 7.2M / 512d | 中英混合会议 | 适合普通话夹英文 |
| `eres2net` | ERes2NetV2 | `iic/speech_eres2netv2_sv_zh-cn_16k-common` | 17.8M / 192d | 中文高精度对照 | 精度较高 |
| `eres2net_base` | ERes2Net Base | `iic/speech_eres2net_base_sv_zh-cn_3dspeaker_16k` | 6.61M / 512d | 速度/精度折中 | 3D-Speaker 中文基线 |
| `eres2net_large` | ERes2Net Large | `iic/speech_eres2net_large_sv_zh-cn_3dspeaker_16k` | 22.46M / 512d | 离线高精度声纹 | 适合强机器质量对照 |
| `ecapa_tdnn` | ECAPA-TDNN | `iic/speech_ecapa-tdnn_sv_zh-cn_3dspeaker_16k` | 20.8M / 192d | 经典中文声纹基线 | 适合和 CAM++ / ERes2Net 对照 |
| `wespeaker` | ResNet34 | `iic/speech_resnet34_sv_zh-cn_3dspeaker_16k` | 6.34M / 256d | 稳定轻量基线 | WeSpeaker 路线 |

## 说话人分离模型

说话人分离负责把完整会议切成“谁在什么时候说话”的时间段，和“声纹模型”不是同一个任务。声纹模型负责把某段声音匹配到已注册人员，分离模型负责先把多人会议分段。

| 设置值 | 显示名 | 上游模型/接口 | 语言/场景 | 依赖 | 备注 |
| --- | --- | --- | --- | --- | --- |
| `funasr_campplus` | FunASR Paraformer + CAM++ | `paraformer-zh + fsmn-vad + ct-punc + cam++` | 中文会议优先 | `funasr`, `modelscope` | 本地自动下载；无需 Hugging Face gated 授权；属于 ASR 辅助 speaker 标签 |
| `funasr_sensevoice_campplus` | FunASR SenseVoice + CAM++ | `iic/SenseVoiceSmall + fsmn-vad + cam++` | 中文/多语种对照 | `funasr`, `modelscope` | 本地自动下载；适合和 Paraformer 栈对照中文会议效果 |
| `funasr_campplus_cn_en` | FunASR Paraformer + CAM++ CN/EN | `iic/speech_campplus_sv_zh_en_16k-common_advanced` | 中英混合会议 | `funasr`, `modelscope` | 本地自动下载；会议里夹杂英文时优先测试 |
| `funasr_paraformer_large_campplus` | FunASR Paraformer Large + CAM++ | `iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch + cam++` | 中文离线高质量对照 | `funasr`, `modelscope` | 本地自动下载；比默认栈更慢，适合离线对比 |
| `funasr_eres2netv2` | FunASR Paraformer + ERes2NetV2 | `iic/speech_eres2netv2_sv_zh-cn_16k-common` | 中文实验对照 | `funasr`, `modelscope` | 实验项；通过 FunASR `spk_model` 接 3D-Speaker 声纹模型，需以本机加载测试为准 |
| `pyannote_community` | pyannote Community-1 | `pyannote/speaker-diarization-community-1` | 多语种完整录音 | `pyannote.audio`, `HF_TOKEN` | 准确率基线强；需要接受 Hugging Face 模型条款 |
| `pyannote_custom` | pyannote Custom Pipeline | 自定义 pyannote pipeline/model id | 多语种对照测试 | `pyannote.audio`, `HF_TOKEN` | 用于测试 `pyannote/speaker-diarization-3.1` 等其它 pipeline |
| `sherpa_onnx_cli` | sherpa-onnx / external CLI | 本地命令输出 JSON 或 RTTM | 取决于本地模型 | 外部命令 | 用来接 sherpa-onnx、3D-Speaker 或其它离线分离程序 |

## 统一配置文件

设置页保存后会写入 `config/model-settings.json`，模板见 `config/model-settings.example.json`。真实配置文件已加入 `.gitignore`，因为里面可能包含 API key。

支持的配置段：

| 段 | 字段 | 说明 |
| --- | --- | --- |
| `asr` | `provider`, `endpoint`, `api_key`, `model`, `device`, `word_timestamps`, `load_timeout_sec` | ASR 模型源、模型选择和运行设备 |
| `speaker` | `provider`, `endpoint`, `api_key`, `model`, `device` | 声纹 embedding 模型源和模型选择 |
| `diarization` | `engine`, `provider`, `endpoint`, `api_key`, `model`, `command`, `device` | 完整会议多人分离模型 |
| `llm` | `provider`, `endpoint`, `api_key`, `model`, `enabled`, `allow_public`, `timeout_sec`, `max_input_tokens`, `mock` | OpenAI-compatible LLM 配置 |

## 中文会议推荐顺序

1. ASR 先试 `sensevoice_zh`：下载小、速度快、中文固定语言，适合 AMD Windows 先验证链路。
2. 专名、人名识别不稳时试 `seaco_paraformer`，并在 `.env` 写 `ASR_FUNASR_HOTWORDS`。
3. 实时延迟对照试 `paraformer_online` 或 `paraformer_streaming`；当前项目仍按 VAD 语音段调用，不是逐 token 真流式。
4. 上传长会议对照试 `paraformer_full` / `paraformer_large`；说话人标签对照试 `sensevoice_spk` / `paraformer_spk`。
5. 说话人分离先试 `funasr_campplus`：对中文友好，沿用项目已有 FunASR 依赖和 ModelScope 缓存，不需要 HF_TOKEN。
6. 再试 `pyannote_community`：分离能力强，但需要 Hugging Face gated 模型授权；AMD Windows ROCm 环境下建议先用 CPU 验证。
