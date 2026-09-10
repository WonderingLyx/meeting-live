# 模型选型参考

本项目以"小、快、可在消费级硬件跑"为原则做模型选型。本文档说明当前支持的模型、能力边界和推荐场景。

运行时能力以 `/v1/models` 为准。设置页读取后端返回的 `asr_engines.engines[*].capabilities`,不会在前端硬编码能力矩阵。部署方可以用 `ASR_CAPABILITIES_JSON` 或 `ASR_CAPABILITIES_FILE` 覆盖某个模型的能力说明,例如关闭字级时间戳、标记自研适配器支持真流式、补充本部署限制等。

---

## 当前可用

### ASR: SenseVoice / Paraformer / Paraformer Streaming（默认 SenseVoice Chinese）
- **来源**: ModelScope / FunASR
- **依赖**: `funasr==1.4.1`（Windows ROCm 已验证版本）
- **切换**: 设置页或 `PUT /v1/asr/engine`
- **行为**: 新 ASR 下载/加载完成前继续使用旧 ASR;加载失败不会影响当前引擎
- **启动降级**: 默认 `ASR_STARTUP_ALLOW_DOWNLOAD=false`,启动时不会因为 Qwen 未缓存而阻塞,会优先尝试 `sensevoice_zh,paraformer_full,paraformer`
- **能力来源**: 后端 `/v1/models` 返回,可通过 `ASR_CAPABILITIES_JSON` / `ASR_CAPABILITIES_FILE` 配置覆盖
- **适用**:
  - SenseVoice Chinese (`sensevoice_zh`): 中文固定语言快速转写,中文会议优先测试
  - SenseVoice Speaker Tags (`sensevoice_spk`): SenseVoice + VAD + CAM++ 匿名 speaker 标签,用于和独立说话人分离对照
  - SenseVoice-Small: 多语种上传转写,模型更轻
  - Paraformer: 中文会议/访谈离线转写
  - Paraformer Full (`paraformer_full`): Paraformer + VAD + 标点,适合较长中文会议
  - SeACo Paraformer (`seaco_paraformer`): 热词/专名增强中文模型,适合人名、部门名、项目名较多的会议
  - Paraformer Large (`paraformer_large`): ModelScope Paraformer Large,适合离线质量对照
  - Paraformer Speaker Tags (`paraformer_spk`): FunASR 内部 speaker 标签,只作对照
  - Paraformer Online (`paraformer_online`): 明确使用 ModelScope 在线低延迟模型,用于实时延迟对照
  - Paraformer Streaming (`paraformer_streaming`): FunASR 内置流式别名,低延迟实时字幕对照
- **能力差异**:
  - FunASR 系列当前按段级结果接入,不提供 Qwen3 ForcedAligner 的 word timestamps
  - `sensevoice_spk` / `paraformer_spk` 的 speaker 标签来自 FunASR 当前音频块,不替代项目的 pyannote 分离或注册声纹身份识别
  - `seaco_paraformer` 可通过 `ASR_FUNASR_HOTWORDS` 输入热词,适合中文会议里的姓名、组织和业务词
  - Paraformer Online / Streaming 表示模型适合流式/低延迟场景,但当前服务端展示不是 token-level 真流式逐 token 输出
  - 切换 ASR 只改变转写模型,不会改变说话人识别算法;实时说话人由 Speaker Engine 完成,上传离线 diarization 可走 pyannote

### ASR: Qwen3-ASR-0.6B（高质量可选）
- **来源**: Hugging Face `Qwen/Qwen3-ASR-0.6B`
- **大小**: 1.8GB
- **设备**: MPS / CUDA / CPU(默认 auto,mps 优先,90s 超时回退)
- **能力**: 多语种(中英日韩等)、50+ 语言识别、自动语言检测、长音频(分段)
- **可选**: Qwen3-ForcedAligner-0.6B(600MB),给字级时间戳用,`ASR_WORD_TIMESTAMPS=true` 启用
  - **开启后效果**:
    - 实时响应和 SQLite 会议文稿可保存 `words: [{text, start, end}]`
    - SRT/VTT 字幕按字切分(0.3s/字 vs 默认 3s/段),适合视频剪辑/卡拉 OK
    - 会议详情和字幕导出可使用更精确的时间
  - **代价**:
    - 首次启动多下载 600MB 模型(国内需 HF 镜像)
    - ASR 加载多 5-10s(MPS 上偶发死锁,90s 超时回退 CPU)
    - 每次推理多 50-200ms(对齐计算)
  - **推荐**: 个人学习保持 false(轻量);需要精确字幕/视频剪辑场景开 true
- **优点**: 中文识别强,适合做质量对照
- **缺点**: 体积较大,低端机器加载慢;HF 镜像依赖

> 能力边界：word timestamps / 字级时间戳当前只在 Qwen3-ASR + `ASR_WORD_TIMESTAMPS=true` 路径可用。说话人识别不是 Qwen3-ASR 自带能力,由声纹引擎或上传离线 pyannote 负责。Qwen 模型依赖 Hugging Face/镜像访问,不再作为 Windows 一键包默认启动模型。

### 声纹: 3D-Speaker / ModelScope Speaker Verification
- **来源**: ModelScope `damo/speech_campplus_sv_zh-cn_16k-common`、`iic/speech_eres2net_large_sv_zh-cn_3dspeaker_16k` 等
- **大小**: 6-23M 参数级别,首次切换到未缓存模型时自动下载
- **能力**: 说话人识别(谁在说话)、声纹库累积、cosine 距离比对
- **切换**: `SPEAKER_ENGINE=campplus|campplus_cn_en|eres2net|eres2net_base|eres2net_large|ecapa_tdnn|wespeaker` 或设置页运行时可切
- **推荐顺序**: 中文会议先测 `campplus`,中英混合测 `campplus_cn_en`,离线质量对照测 `eres2net_large`,基线对照测 `ecapa_tdnn` / `wespeaker`
- **embedding_dim / model_id**: 不同模型的 embedding 空间不同。项目会把模型 id、revision、维度和归一化方式写入 voice sample,切引擎后旧样本保留但不会跨模型误匹配。

### VAD: Silero VAD
- **来源**: torch.hub `snakers4/silero-vad`
- **大小**: < 2MB
- **设备**: CPU(很小,不值得放 GPU)
- **作用**: 流式状态机 SILENCE ↔ SPEECH 切换,触发 ASR

---

## 可选 / 候选模型说明

### SenseVoice-Small
- **大小**: ~250MB(对比 Qwen3-ASR-0.6B 的 1.8GB)
- **来源**: ModelScope `iic/SenseVoiceSmall`
- **能力**: ASR + 语种识别 + 情感识别 + 声音事件检测(AED),多任务
- **优点**: 体积小 7x,速度快(实测 ~2x),多语种 50+
- **缺点**:
  - **不支持流式**:SenseVoice 是非自回归,完整段输入才能识别(WebSocket 实时场景不适用)
  - **中文方言仅普通话 + 粤语**:`labels` 字段 50+ 语言是指"语种",不是"方言"。粤语/闽南语/上海话等不支持
  - **中文表现通常弱于 Qwen3-ASR-0.6B**:适合轻量部署,高质量中文转写可手动切换 Qwen3-ASR 对照
  - **声学事件 + 情感不是本项目目标**:多任务反而拖慢主任务
- **结论**: 已作为默认 ASR 集成,适合"批量处理短音频 + 多语种"场景;Qwen3-ASR 保留为高质量可选项。

### ZipEnhancer (speech_zipenhancer_ans_multiloss_16k_base)
- **大小**: 2.04M 参数(约 8MB,极小)
- **来源**: ModelScope `iic/speech_zipenhancer_ans_multiloss_16k_base`
- **能力**: **单通道语音降噪 / 增强** — 16kHz 噪声音频 → 16kHz 干净人声
- **论文**: ICASSP 2025 [arxiv:2501.05183](https://arxiv.org/abs/2501.05183)
- **基准**: DNS Challenge 2020 上 PESQ 3.69(SOTA 同规模)
- **优点**:
  - 极小(2M 参数)、SOTA 降噪质量
  - 16kHz in/out,与本项目采样率匹配
  - CPU/MPS 都能跑
- **缺点 / 风险**:
  - **不是 ASR,不是声纹** — 是前端信号处理
  - **职责冲突**: 插进来变成 `VAD → ZipEnhancer → ASR → Speaker` 四级流水线,违反"单进程"原则
  - **过增强失真**: 干净语音可能被当作噪声削掉
  - **声纹退化**: CamPlus 训练在 clean 上,降噪会破坏说话人特征,反而降低 EER
  - **延迟代价**: 2-3s 一段推理 50-100ms(CPU/MPS),实时流必须串行
- **结论**: 暂不集成。如果未来要做"嘈杂环境优化":
  1. 先用真实噪声样本评估 ASR 和声纹识别的变化
  2. 默认关闭,`.env` 加 `NOISE_ENHANCEMENT=zipenhancer`
  3. 声纹提取强制走**原始**音频(跳过降噪)
  4. 加 PESQ 量化指标,让用户看降噪前后质量差

### 其他小 ASR 候选
- **SeACo Paraformer / Paraformer Online / SenseVoice Speaker Tags** (FunASR): 已作为可选 ASR 集成。
- **Fun-ASR-Nano-2512 / Fun-ASR-MLT-Nano-2512** (FunASR): 上游标注为 800M 级模型，中文/多语种能力更强，但对当前 Windows AMD 包固定的 `funasr==1.4.1`、`transformers<5` 和 ROCm 稳定性风险更高。暂不内置为可切换项，建议先用独立环境验证后再接插件。
- **Whisper-tiny**: 75MB,英文强,中文弱,延迟高。不推荐。
- **Whisper-base**: 150MB,中文一般,延迟高。不推荐。
- **WenetSpeech** 系列: 工业级,通常 1GB+,与本项目"小"原则不符。

---

## 选型原则

1. **< 2GB 模型优先**: 1.8GB 的 Qwen3-ASR-0.6B 是上限(MPS 容易 OOM)
2. **中文为第一优先级**: 项目主要服务中文场景(README/i18n 都是简体中文)
3. **流式友好**: WebSocket 实时流优先选择低延迟模型;离线上传可使用非流式模型
4. **可热切换**: ASR 与声纹引擎均支持运行时切换
5. **离线 / 国产化**: ModelScope 镜像 + 阿里生态(Qwen/CamPlus/ERes2Net)优先

## 许可证与供应链边界

仓库的 MIT License 只覆盖项目代码，不覆盖 Qwen、FunASR、pyannote、CamPlus、ERes2Net、Wespeaker、Silero 或其他模型权重。模型可能有独立许可证、访问授权和使用限制，部署者必须在下载页面核对当前条款。

Qwen ASR / ForcedAligner、pyannote Community-1、Silero VAD 和 ModelScope 声纹引擎使用固定 revision 或固定仓库版本；可通过 `.env.example` 中对应变量显式覆盖。FunASR 的内置别名模型（例如 `paraformer-zh`、`fsmn-vad`、`ct-punc`、`cam++`）由 FunASR/ModelScope 内部解析，本项目通过 `funasr==1.4.1`、`disable_update=True` 和 Windows ROCm 版本保护降低漂移风险。复现实验仍应记录模型仓库、revision、缓存文件哈希、Python 环境和硬件；不要把模型缓存作为项目代码重新分发。ModelScope 模型可能加载上游自定义代码，只应使用已审核来源并在低权限隔离环境运行。

### 版本漂移检查

| 能力 | 当前保护 | 剩余风险 |
|---|---|---|
| Qwen3 ASR / ForcedAligner | 代码固定 Hugging Face revision；`qwen-asr==0.0.6` | 首次下载仍依赖 HF/镜像；未缓存时启动不会阻塞，会降级到本地 FunASR 候选 |
| FunASR SenseVoice / Paraformer ASR | `funasr==1.4.1`；`disable_update=True`；Windows ROCm 对 Paraformer 系做安全版本拦截 | FunASR 内置别名模型不是显式 revision，需保留安装包依赖固定 |
| 上传会议 FunASR 说话人分离 | 同 ASR 的 FunASR 版本保护；Paraformer 系分离在不安全版本下标记不可用并停止加载 | SenseVoice + CAM++ 仍保留为对照路径；效果需按真实会议样本评估 |
| pyannote diarization | `pyannote.audio==4.0.4`；Community-1 固定 revision | Hugging Face gated 模型仍需账号授权和 `HF_TOKEN` |
| 声纹识别 | 每个 ModelScope speaker engine 固定 `model_revision`，样本按 model id/revision/维度隔离 | 用户自定义 revision 会产生新的声纹空间，旧样本不会跨模型误匹配 |
| 人脸签到 | `insightface==1.0.1`；安装脚本固定 ONNXRuntime/OpenCV/Pillow/scikit-image 版本 | InsightFace 模型包首次下载后本地缓存；发布包默认不再分发上游权重 |

### 模型许可与联网矩阵

“项目 MIT License”不等于“模型可任意商用或再分发”。下表描述应用的
访问行为，不替代上游条款；发布部署包前应再次核对对应模型卡的当前
license、地域和用途限制。

| 能力 / 默认仓库 | 托管方 | 首次使用是否联网 | Token / 门控 | 许可责任 |
|---|---|---:|---|---|
| Qwen3 ASR / ForcedAligner (`Qwen/...`) | ModelScope（及所配置镜像） | 是 | 通常不需要 | 以 Qwen 模型卡为准，不随本仓库 MIT 授权 |
| SenseVoice / Paraformer (`iic/...`) | ModelScope | 是 | 通常不需要 | 代码与权重条款可能不同，逐个模型卡核对 |
| CamPlus / ERes2Net / ResNet34 (`damo/...`, `iic/...`) | ModelScope | 是 | 通常不需要 | 声纹权重及上游自定义代码不随项目再许可 |
| pyannote Community-1 | Hugging Face | 是 | 需要 `HF_TOKEN` 且先接受模型条款 | 模型卡标注 CC-BY-4.0；仍须遵守门控条款和署名要求 |
| Silero VAD (`snakers4/silero-vad`) | Torch Hub / GitHub | 是 | 不需要 | 以对应版本仓库和权重说明为准 |
| 已缓存模型 | 本机用户缓存 | 否 | pyannote 首次授权仍需预先完成 | 缓存不得被默认打入本项目发布包 |

模型下载是可选网络访问；转写推理、说话人分离和声纹比对在模型就绪后
均可本机执行。外部 LLM 是另一条独立数据通路，详见 `docs/PRIVACY.md`。

## 处理状态与来源

- 实时结果首先是 `draft`，适合即时阅读，不应被描述为最终说话人结果。
- 完整录音处理成功后为 `refined`。字级时间戳存在时优先按词对齐；否则只能按语句时间重叠近似分配，绝不按字符比例伪造边界。
- 会议详情的 `processing_manifest` 记录本次实际使用的 ASR、分离、对齐和身份建议模型；设置页展示的是下一次处理的当前配置，两者含义不同。
- `SPEAKER_XX` 只表示本场会议中的匿名聚类。严格通过模型兼容、语音时长、双样本、相似度与候选差距校验时可自动显示已登记人物，并标注为自动匹配；较弱结果只给建议，低置信度保持匿名。

---

## 升级路径(v0.4+ 候选)

| 维度 | 当前 | 候选 | 收益 | 代价 |
|---|---|---|---|---|
| ASR | Qwen3-ASR-0.6B | Paraformer-streaming | 流式延迟 ↓50% | 中文 WER 略升 |
| ASR | Qwen3-ASR-0.6B | Qwen3-ASR-1.7B | WER ↓2-3% | 模型 3GB+,MPS 易 OOM |
| 声纹 | CamPlus | ERes2NetV2 (默认) | CN-Celeb EER 6.14% (vs 6.78%) | 慢一些(17M vs 7M 参数) |
| 降噪 | 无 | ZipEnhancer | 嘈杂环境 +5% WER | 加 50ms 延迟,有声纹退化风险 |
| 对齐 | 无(可选 Qwen3-ForcedAligner) | 直接用 Qwen3-ASR 自身的 timestamp | 简化流程 | Qwen3-ASR 原生时间戳不够准 |

新模型接入前,应优先确认它是否能解决明确的使用问题,而不是只看模型规模或榜单指标。
