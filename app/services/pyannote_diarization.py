"""pyannote 离线说话人分离服务

定位: 上传录音文件时的"高准确度"说话人识别模式。
- 延迟: 2-5 秒 (需整段音频后处理,不能实时)
- 依赖: pyannote.audio >= 4.0.4 + HF_TOKEN 环境变量
- 模型: pyannote/speaker-diarization-community-1 (CC-BY-4.0)

调研依据:
- Community-1: 使用完整音频上下文的本地离线 diarization pipeline
- 公开会议集 DER 约 12-20%，且会随麦克风、噪声和重叠语音显著变化
- 本项目仅在完整文件上传路径启用，不影响实时 WebSocket 延迟

用法:
    diar = PyannoteDiarizer()
    segments = diar.diarize("audio.wav")  # [(start, end, "SPEAKER_00"), ...]
"""
import logging
import os
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np

# Backward-compatible public import.  The alignment implementation lives in a
# dependency-free module so meeting processing can use it without initializing
# pyannote, but existing integrations historically imported it from here.
from app.services.speaker_alignment import (
    align_speakers_to_segments as align_speakers_to_segments,
)

# pyannote/audio 在边界帧(样本数<=1、空切片)上会触发数值统计退化:
#   - std(): degrees of freedom <= 0  (pooling.py:103)
#   - Mean of empty slice / invalid value encountered in divide
# 这些是 pyannote 内部已知行为,结果用 0/NaN 兜底,不影响最终分离。
# 加载模型前过滤掉,避免污染日志、干扰排查真问题。
warnings.filterwarnings("ignore", message="std\\(\\).*degrees of freedom", category=UserWarning)
warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="invalid value encountered in divide", category=RuntimeWarning)


logger = logging.getLogger("Matrix_Pyannote")
DEFAULT_PYANNOTE_MODEL = "pyannote/speaker-diarization-community-1"
DEFAULT_PYANNOTE_REVISION = "3533c8cf8e369892e6b79ff1bf80f7b0286a54ee"


def _pyannote_model_id() -> str:
    return (os.environ.get("PYANNOTE_MODEL") or DEFAULT_PYANNOTE_MODEL).strip() or DEFAULT_PYANNOTE_MODEL


def _pyannote_revision(model_id: str) -> str | None:
    explicit = (os.environ.get("PYANNOTE_MODEL_REVISION") or "").strip()
    if explicit:
        return explicit
    if model_id == DEFAULT_PYANNOTE_MODEL:
        return DEFAULT_PYANNOTE_REVISION
    return None


def _hf_cache_repo_dir(model_id: str) -> str:
    return "models--" + model_id.replace("/", "--")


def _format_pyannote_error(exc: Exception, model_id: str) -> str:
    raw = str(exc)
    lower = raw.lower()
    if "403" in raw or "gated repo" in lower or "authorized list" in lower:
        return (
            f"{model_id} 访问被拒绝: Hugging Face 返回 403。"
            f"请用 HF_TOKEN 所属账号打开 https://huggingface.co/{model_id} "
            "接受/申请访问，并确认 token 具备 Read 权限和公开 gated repo 访问权限。"
            f"原始错误: {raw}"
        )
    if "401" in raw or "invalid token" in lower:
        return f"{model_id} token 无效或权限不足: 请重新创建 Read token。原始错误: {raw}"
    return f"{model_id} 加载失败: {raw}"


def _env_bool(key: str, default: bool) -> bool:
    value = os.environ.get(key)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _is_rocm_torch(torch) -> bool:
    return bool(getattr(torch.version, "hip", None))


def _iter_torch_modules(root, torch):
    seen: set[int] = set()
    stack = [root]
    while stack:
        obj = stack.pop()
        if obj is None:
            continue
        obj_id = id(obj)
        if obj_id in seen:
            continue
        seen.add(obj_id)

        if isinstance(obj, torch.nn.Module):
            yield obj
            stack.extend(obj.children())
            continue

        if isinstance(obj, dict):
            stack.extend(obj.values())
            continue
        if isinstance(obj, (list, tuple, set)):
            stack.extend(obj)
            continue

        attrs = getattr(obj, "__dict__", None)
        if isinstance(attrs, dict):
            for key, value in attrs.items():
                if key.startswith("__"):
                    continue
                stack.append(value)


def _disable_recurrent_dropout_for_rocm(pipeline, torch) -> int:
    """Work around ROCm/MIOpen JIT failures in pyannote inference.

    Some ROCm builds try to JIT-compile MIOpenDropoutHIP for recurrent layers
    even during inference.  On Windows ROCm this can fail when HIPRTC cannot
    resolve rocrand/hiprand headers.  Recurrent dropout is only a training-time
    regularizer, so setting it to 0.0 does not change inference behavior.
    """
    if not _is_rocm_torch(torch):
        return 0
    if not _env_bool("PYANNOTE_ROCM_DISABLE_LSTM_DROPOUT", True):
        return 0

    patched = 0
    recurrent_types = (torch.nn.RNN, torch.nn.GRU, torch.nn.LSTM)
    for module in _iter_torch_modules(pipeline, torch):
        if isinstance(module, recurrent_types) and float(getattr(module, "dropout", 0.0) or 0.0) != 0.0:
            module.dropout = 0.0
            patched += 1
    if patched:
        logger.info("[PYANNOTE] ROCm: 已关闭 %d 个 recurrent dropout，规避 MIOpenDropoutHIP 编译错误", patched)
    return patched


class PyannoteDiarizer:
    """pyannote 离线匿名说话人分离

    单例懒加载:首次调用 diarize() 时初始化 pipeline(下载模型 ~50MB)。
    失败时不抛异常,返回空列表,让上层保留匿名文稿。
    """

    _instance = None
    _pipeline = None
    _enabled = False
    _last_error = None
    _device = "cpu"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._pipeline is not None:
            return
        self._last_error = None
        # 检查 HF_TOKEN
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        if not token:
            logger.warning(
                "[PYANNOTE] HF_TOKEN 未设置，上传会议将保留匿名说话人"
            )
            self._last_error = "HF_TOKEN 未设置"
            self._enabled = False
            return
        try:
            from pyannote.audio import Pipeline
            import torch
        except Exception as e:
            self._last_error = f"pyannote.audio 无法加载: {e}"
            logger.warning("[PYANNOTE] %s", self._last_error)
            self._enabled = False
            return
        try:
            # 本地化:cache_dir 指向 models/pyannote/,HF pipeline 缓存于此
            # (hashed 结构 models--pyannote--.../snapshots/,在 models/ 下)。
            from app.services.model_resolver import local_path
            import os as _os
            import glob as _glob
            model_id = _pyannote_model_id()
            revision = _pyannote_revision(model_id)
            _pyannote_cache = local_path("pyannote", "_cache")
            _os.makedirs(_pyannote_cache, exist_ok=True)
            _repo_dir = _os.path.join(
                _pyannote_cache, _hf_cache_repo_dir(model_id)
            )
            _snapshots = _glob.glob(_os.path.join(_repo_dir, "snapshots", "*"))
            if _snapshots:
                logger.info(
                    "[PYANNOTE] 本地缓存命中,离线加载 pyannote/speaker-diarization-community-1 (cache=%s)",
                    _pyannote_cache,
                )
            else:
                logger.info(
                    "[PYANNOTE] 本地无缓存,联网下载 pyannote/speaker-diarization-community-1 (~50MB) → %s",
                    _pyannote_cache,
                )
            pipeline_kwargs = {
                "token": token,
                "cache_dir": _pyannote_cache,
            }
            if revision:
                pipeline_kwargs["revision"] = revision
            self._pipeline = Pipeline.from_pretrained(model_id, **pipeline_kwargs)
            _disable_recurrent_dropout_for_rocm(self._pipeline, torch)
            from app.config import config

            requested = config.speaker.diarization_device
            device = "cuda" if requested == "auto" and torch.cuda.is_available() else requested
            if device == "auto":
                device = "cpu"
            if device == "cuda" and not torch.cuda.is_available():
                logger.warning("[PYANNOTE] CUDA 不可用，回退 CPU")
                device = "cpu"
            self._pipeline.to(torch.device(device))
            self._device = device
            self._enabled = True
            logger.info("[PYANNOTE] 加载完成，设备=%s", device)
        except Exception as e:
            self._last_error = _format_pyannote_error(e, _pyannote_model_id())
            logger.error("[PYANNOTE] %s", self._last_error)
            self._enabled = False

    @property
    def enabled(self) -> bool:
        """是否可用(检查 HF_TOKEN + 模型是否加载成功)"""
        return self._enabled

    @property
    def last_error(self) -> str | None:
        return self._last_error

    @property
    def device(self) -> str:
        return self._device

    def diarize(
        self,
        audio_path: str,
        *,
        waveform: "np.ndarray | None" = None,
        sample_rate: int | None = None,
    ) -> List[Tuple[float, float, str]]:
        """对音频做离线匿名说话人分离

        Args:
            audio_path: 本地音频路径(可由 librosa/soundfile 解码)。当
                waveform 未提供时用它解码。
            waveform: 调用方已加载的 1D float32 mono 波形。提供时直接
                复用,避免对同一段音频二次 librosa.load(长会议 ~576MB×2
                峰值)。采样率须与 sample_rate 一致。
            sample_rate: waveform 的采样率;未提供时回退到 16kHz。

        Returns:
            List of (start_sec, end_sec, speaker_id) — speaker_id 形如 "SPEAKER_00"
            失败时返回 []
        """
        if not self._enabled:
            return []
        if waveform is None and not Path(audio_path).exists():
            logger.warning(f"[PYANNOTE] 文件不存在: {audio_path}")
            return []
        try:
            import librosa
            import torch

            if waveform is not None:
                sr = int(sample_rate or 16000)
                raw_wave = np.asarray(waveform)
                if raw_wave.ndim != 1:
                    raise ValueError(
                        f"waveform must be 1D mono, got shape={raw_wave.shape}"
                    )
                if np.issubdtype(raw_wave.dtype, np.integer):
                    info = np.iinfo(raw_wave.dtype)
                    scale = float(max(abs(info.min), info.max))
                    wave = raw_wave.astype(np.float32) / scale
                else:
                    wave = raw_wave.astype(np.float32, copy=False)
            else:
                wave, sr = librosa.load(audio_path, sr=None, mono=True)
                wave = np.asarray(wave, dtype=np.float32)
            if int(sr) <= 0:
                raise ValueError(f"sample_rate must be positive, got {sr}")
            if wave.ndim != 1 or wave.size == 0:
                raise ValueError("waveform must be a non-empty 1D mono array")
            if not np.isfinite(wave).all():
                raise ValueError("waveform contains NaN or infinity")
            if float(np.max(np.abs(wave))) > 1.5:
                raise ValueError("float waveform level must be normalized near [-1, 1]")
            audio_input = {
                "waveform": torch.as_tensor(wave, dtype=torch.float32).unsqueeze(0),
                "sample_rate": int(sr),
            }
            output = self._pipeline(audio_input)
            annotation = getattr(
                output,
                "exclusive_speaker_diarization",
                None,
            )
            if annotation is None:
                annotation = getattr(output, "speaker_diarization", output)
            results = []
            for turn, _track, speaker in annotation.itertracks(yield_label=True):
                results.append((float(turn.start), float(turn.end), str(speaker)))
            logger.info(f"[PYANNOTE] {audio_path}: {len(results)} 个说话人段,识别出 {len(set(s[2] for s in results))} 个不同说话人")
            return results
        except Exception as e:
            self._last_error = f"说话人分离执行失败: {e}"
            logger.error("[PYANNOTE] %s", self._last_error)
            return []


def get_pyannote_diarizer() -> PyannoteDiarizer:
    return PyannoteDiarizer()


def reset_pyannote_diarizer() -> None:
    """Drop the cached pyannote pipeline so runtime settings can be reloaded."""
    PyannoteDiarizer._instance = None
    PyannoteDiarizer._pipeline = None
    PyannoteDiarizer._enabled = False
    PyannoteDiarizer._last_error = None
    PyannoteDiarizer._device = "cpu"
