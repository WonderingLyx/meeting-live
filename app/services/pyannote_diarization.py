"""Offline speaker diarization engines.

This module keeps the historical pyannote import path while exposing a generic
factory for local diarization backends used by uploaded meeting processing.
"""
from __future__ import annotations

import importlib.util
import json
import logging
import os
import re
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import Any, List, Tuple

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


logger = logging.getLogger("Matrix_Diarization")
DEFAULT_DIARIZATION_ENGINE = "pyannote_community"
DEFAULT_PYANNOTE_MODEL = "pyannote/speaker-diarization-community-1"
DEFAULT_PYANNOTE_REVISION = "3533c8cf8e369892e6b79ff1bf80f7b0286a54ee"
DEFAULT_FUNASR_DIARIZATION_MODEL = "paraformer-zh + fsmn-vad + ct-punc + cam++"

DIARIZATION_ENGINE_CONFIG: dict[str, dict[str, Any]] = {
    "pyannote_community": {
        "type": "pyannote_community",
        "name": "pyannote Community-1",
        "provider": "huggingface",
        "endpoint": "https://huggingface.co",
        "model": DEFAULT_PYANNOTE_MODEL,
        "dependency": "pyannote.audio",
        "token_env": "HF_TOKEN",
        "requires_token": True,
        "local_after_download": True,
        "languages": "多语种",
        "languages_en": "multilingual",
        "description": "完整录音离线多人分离，准确率基线最好，但 Hugging Face gated 模型需要授权。",
        "description_en": "Full-recording offline diarization. Strong baseline, but the Hugging Face gated model requires access approval.",
        "recommended_for": ["meeting_upload", "accuracy_baseline"],
        "terms_url": f"https://huggingface.co/{DEFAULT_PYANNOTE_MODEL}",
    },
    "pyannote_custom": {
        "type": "pyannote_custom",
        "name": "pyannote Custom Pipeline",
        "provider": "huggingface",
        "endpoint": "https://huggingface.co",
        "model": "pyannote/speaker-diarization-3.1",
        "dependency": "pyannote.audio",
        "token_env": "HF_TOKEN",
        "requires_token": True,
        "local_after_download": True,
        "languages": "多语种",
        "languages_en": "multilingual",
        "description": "用于测试其它 pyannote pipeline/model id，仍按本地缓存加载。",
        "description_en": "Use another pyannote pipeline/model id while keeping local cache loading.",
        "recommended_for": ["pyannote_ab_test"],
        "terms_url": "https://huggingface.co/pyannote/speaker-diarization-3.1",
    },
    "funasr_campplus": {
        "type": "funasr_campplus",
        "name": "FunASR Paraformer + CAM++",
        "provider": "modelscope",
        "endpoint": "https://modelscope.cn",
        "model": DEFAULT_FUNASR_DIARIZATION_MODEL,
        "dependency": "funasr",
        "token_env": "MODELSCOPE_API_TOKEN",
        "requires_token": False,
        "local_after_download": True,
        "languages": "中文",
        "languages_en": "Chinese",
        "description": "中文友好的本地 ASR 辅助说话人标签，复用 FunASR/ModelScope 自动下载链路。",
        "description_en": "Chinese-friendly local ASR-assisted speaker labels using FunASR/ModelScope downloads.",
        "recommended_for": ["chinese_meeting_upload", "no_hf_token"],
    },
    "sherpa_onnx_cli": {
        "type": "sherpa_onnx_cli",
        "name": "sherpa-onnx / external CLI",
        "provider": "local",
        "endpoint": "file://./models",
        "model": "sherpa-onnx diarization command",
        "dependency": "external_command",
        "token_env": "",
        "requires_token": False,
        "local_after_download": True,
        "languages": "取决于本地模型",
        "languages_en": "depends on local model",
        "description": "本地命令适配器，可接 sherpa-onnx、3D-Speaker 等输出 JSON 或 RTTM 的离线分离程序。",
        "description_en": "Local command adapter for sherpa-onnx, 3D-Speaker, or any offline tool producing JSON/RTTM turns.",
        "recommended_for": ["onnx_runtime", "custom_diarization"],
    },
}

_PYANNOTE_ENGINES = {"pyannote", "pyannote_community", "pyannote_custom"}


def _pyannote_model_id() -> str:
    return (
        os.environ.get("PYANNOTE_MODEL")
        or os.environ.get("DIARIZATION_MODEL")
        or DEFAULT_PYANNOTE_MODEL
    ).strip() or DEFAULT_PYANNOTE_MODEL


def normalize_diarization_engine(engine_type: str | None) -> str:
    value = (engine_type or "").strip().lower().replace("-", "_")
    aliases = {
        "": DEFAULT_DIARIZATION_ENGINE,
        "pyannote": "pyannote_community",
        "community_1": "pyannote_community",
        "community1": "pyannote_community",
        "funasr": "funasr_campplus",
        "funasr_spk": "funasr_campplus",
        "paraformer_spk": "funasr_campplus",
        "sherpa": "sherpa_onnx_cli",
        "sherpa_onnx": "sherpa_onnx_cli",
        "external": "sherpa_onnx_cli",
    }
    return aliases.get(value, value)


def configured_diarization_engine() -> str:
    raw = os.environ.get("DIARIZATION_ENGINE")
    if not raw:
        try:
            from app.services.model_config import read_model_settings

            settings = read_model_settings(include_env_secrets=True).get("diarization", {})
            raw = str(settings.get("engine") or "")
        except Exception:
            raw = ""
    return normalize_diarization_engine(raw)


def default_diarization_model(engine_type: str | None = None) -> str:
    engine = normalize_diarization_engine(engine_type or configured_diarization_engine())
    return str(DIARIZATION_ENGINE_CONFIG.get(engine, {}).get("model") or DEFAULT_PYANNOTE_MODEL)


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


def _diarization_settings() -> dict[str, Any]:
    try:
        from app.services.model_config import read_model_settings

        settings = read_model_settings(include_env_secrets=True).get("diarization", {})
        return dict(settings) if isinstance(settings, dict) else {}
    except Exception:
        return {}


def _diarization_command() -> str:
    settings = _diarization_settings()
    return (
        os.environ.get("DIARIZATION_COMMAND")
        or str(settings.get("command") or "")
    ).strip()


def _funasr_modelscope_cache() -> str:
    from app.services.model_resolver import models_root

    cache = os.path.join(models_root(), "funasr")
    os.makedirs(cache, exist_ok=True)
    return cache


def _dependency_state(info: dict[str, Any]) -> dict[str, Any]:
    dependency = str(info.get("dependency") or "")
    if dependency == "pyannote.audio":
        available = _module_available("pyannote") and _module_available("pyannote.audio")
        hint = "pip install pyannote.audio"
    elif dependency == "funasr":
        available = _module_available("funasr")
        hint = "pip install funasr modelscope"
    elif dependency == "external_command":
        available = bool(_diarization_command())
        hint = "在设置页填写 command，或设置 DIARIZATION_COMMAND；命令输出 JSON/RTTM。"
    else:
        available = True
        hint = ""
    return {
        "dependency_available": available,
        "install_hint": "" if available else hint,
    }


def get_diarization_engine_info(engine_type: str | None = None) -> dict[str, Any]:
    engine = normalize_diarization_engine(engine_type or configured_diarization_engine())
    base = dict(DIARIZATION_ENGINE_CONFIG.get(engine) or {})
    if not base:
        return {
            "type": engine,
            "name": engine,
            "provider": "custom",
            "model": default_diarization_model(DEFAULT_DIARIZATION_ENGINE),
            "available": False,
            "dependency_available": False,
            "install_hint": f"Unsupported diarization engine: {engine}",
        }
    base.update(_dependency_state(base))
    base["available"] = bool(base.get("dependency_available"))
    return base


def get_all_diarization_engines() -> dict[str, dict[str, Any]]:
    return {
        key: get_diarization_engine_info(key)
        for key in DIARIZATION_ENGINE_CONFIG
    }


def _coalesce_turns(
    turns: list[tuple[float, float, str]],
    *,
    max_gap_sec: float = 0.15,
) -> list[tuple[float, float, str]]:
    if not turns:
        return []
    sorted_turns = sorted(turns, key=lambda item: (item[0], item[1], item[2]))
    merged: list[tuple[float, float, str]] = [sorted_turns[0]]
    for start, end, speaker in sorted_turns[1:]:
        prev_start, prev_end, prev_speaker = merged[-1]
        if speaker == prev_speaker and start <= prev_end + max_gap_sec:
            merged[-1] = (prev_start, max(prev_end, end), prev_speaker)
        else:
            merged.append((start, end, speaker))
    return merged


def _speaker_label(value: Any) -> str:
    text = str(value if value is not None else "0").strip()
    if not text:
        text = "0"
    upper = text.upper()
    if upper.startswith(("SPEAKER", "SPK", "FUNASR")):
        return text
    if text.isdigit():
        return f"SPEAKER_{int(text):02d}"
    return f"SPEAKER_{re.sub(r'\\s+', '_', text)}"


def _seconds_from_provider(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number / 1000.0 if number > 300 else number


def _segment_turn(value: dict[str, Any]) -> tuple[float, float, str] | None:
    if not isinstance(value, dict):
        return None
    if "start_sec" in value:
        start = _seconds_from_provider(value.get("start_sec"))
    else:
        start = _seconds_from_provider(value.get("start", value.get("begin")))
    if "end_sec" in value:
        end = _seconds_from_provider(value.get("end_sec"))
    elif "end" in value:
        end = _seconds_from_provider(value.get("end"))
    else:
        duration = _seconds_from_provider(value.get("duration", value.get("dur")))
        end = start + duration
    speaker = value.get("speaker", value.get("spk", value.get("label", value.get("speaker_id"))))
    if speaker is None or end <= start:
        return None
    return (float(start), float(end), _speaker_label(speaker))


def parse_diarization_output(raw: str) -> list[tuple[float, float, str]]:
    """Parse external diarization output in JSON or RTTM format."""
    text = (raw or "").strip()
    if not text:
        return []
    if text[0] in "[{":
        payload = json.loads(text)
        if isinstance(payload, dict):
            for key in ("segments", "diarization", "turns", "items"):
                if isinstance(payload.get(key), list):
                    payload = payload[key]
                    break
        if not isinstance(payload, list):
            raise ValueError("JSON output must be a list or contain segments/diarization/turns")
        turns = []
        for item in payload:
            if isinstance(item, dict):
                turn = _segment_turn(item)
                if turn is not None:
                    turns.append(turn)
        return _coalesce_turns(turns)

    turns: list[tuple[float, float, str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if parts[0].upper() == "SPEAKER" and len(parts) >= 8:
            start = _seconds_from_provider(parts[3])
            duration = _seconds_from_provider(parts[4])
            turns.append((start, start + duration, _speaker_label(parts[7])))
            continue
        if len(parts) >= 3:
            try:
                start = _seconds_from_provider(parts[0])
                end = _seconds_from_provider(parts[1])
            except Exception:
                continue
            if end > start:
                turns.append((start, end, _speaker_label(parts[2])))
    return _coalesce_turns(turns)


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


class FunASRCampPlusDiarizer:
    """FunASR ASR-assisted speaker diarization for Chinese meetings."""

    _instance = None
    _model = None
    _enabled = False
    _last_error = None
    _device = "cpu"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._model is not None:
            return
        self._last_error = None
        try:
            from funasr import AutoModel
        except Exception as exc:
            self._enabled = False
            self._last_error = f"FunASR 未安装或无法加载: {exc}"
            logger.warning("[DIARIZATION:FunASR] %s", self._last_error)
            return

        from app.config import config
        from engine.asr.funasr_engine import FunASREngine

        self._device = FunASREngine._resolve_device(config.speaker.diarization_device)
        cache = _funasr_modelscope_cache()
        old_ms_cache = os.environ.get("MODELSCOPE_CACHE")
        os.environ["MODELSCOPE_CACHE"] = cache
        try:
            logger.info("[DIARIZATION:FunASR] loading %s device=%s cache=%s", DEFAULT_FUNASR_DIARIZATION_MODEL, self._device, cache)
            self._model = AutoModel(
                model="paraformer-zh",
                vad_model="fsmn-vad",
                punc_model="ct-punc",
                spk_model="cam++",
                vad_kwargs={"max_single_segment_time": 30000},
                device=self._device,
            )
            self._enabled = True
        except Exception as exc:
            self._enabled = False
            self._last_error = f"FunASR CAM++ 加载失败: {exc}"
            logger.error("[DIARIZATION:FunASR] %s", self._last_error)
        finally:
            if old_ms_cache is None:
                os.environ.pop("MODELSCOPE_CACHE", None)
            else:
                os.environ["MODELSCOPE_CACHE"] = old_ms_cache

    @property
    def enabled(self) -> bool:
        return bool(self._enabled)

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
        if not self._enabled or self._model is None:
            return []
        if waveform is None and not Path(audio_path).exists():
            self._last_error = f"文件不存在: {audio_path}"
            logger.warning("[DIARIZATION:FunASR] %s", self._last_error)
            return []
        try:
            import librosa

            if waveform is None:
                wave, _sr = librosa.load(audio_path, sr=16000, mono=True)
                audio_input: Any = np.asarray(wave, dtype=np.float32)
            else:
                raw = np.asarray(waveform)
                if raw.ndim != 1:
                    raise ValueError(f"waveform must be 1D mono, got shape={raw.shape}")
                if np.issubdtype(raw.dtype, np.integer):
                    info = np.iinfo(raw.dtype)
                    scale = float(max(abs(info.min), info.max))
                    raw = raw.astype(np.float32) / scale
                audio_input = raw.astype(np.float32, copy=False)
                if int(sample_rate or 16000) != 16000:
                    audio_input = librosa.resample(audio_input, orig_sr=int(sample_rate or 16000), target_sr=16000)

            result = self._model.generate(
                input=audio_input,
                batch_size_s=60,
                merge_vad=True,
                merge_length_s=15,
            )
            turns = self._turns_from_funasr_result(result)
            if not turns:
                self._last_error = "FunASR 未返回 speaker 标签；确认 spk_model=cam++ 是否成功下载/加载"
            else:
                self._last_error = None
            logger.info("[DIARIZATION:FunASR] %s: %d turns, %d speakers", audio_path, len(turns), len({t[2] for t in turns}))
            return turns
        except Exception as exc:
            self._last_error = f"FunASR CAM++ 分离失败: {exc}"
            logger.error("[DIARIZATION:FunASR] %s", self._last_error)
            return []

    @classmethod
    def _turns_from_funasr_result(cls, result: Any) -> list[tuple[float, float, str]]:
        if not result:
            return []
        items = result if isinstance(result, list) else [result]
        turns: list[tuple[float, float, str]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            sentence_info = item.get("sentence_info")
            if isinstance(sentence_info, list):
                for sentence in sentence_info:
                    if isinstance(sentence, dict):
                        turn = _segment_turn(sentence)
                        if turn is not None:
                            turns.append(turn)
                continue
            turn = _segment_turn(item)
            if turn is not None:
                turns.append(turn)
        return _coalesce_turns(turns)


class CommandDiarizer:
    """Run an external local diarization command and parse JSON or RTTM turns."""

    _instance = None
    _enabled = False
    _last_error = None
    _device = "external"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self._command = _diarization_command()
        self._enabled = bool(self._command)
        self._last_error = None if self._enabled else "未配置 DIARIZATION_COMMAND"

    @property
    def enabled(self) -> bool:
        return bool(self._enabled)

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
        if not self._enabled:
            return []
        if not Path(audio_path).exists():
            self._last_error = f"文件不存在: {audio_path}"
            return []
        try:
            fd, temp_name = tempfile.mkstemp(prefix="matrix_diar_", suffix=".json")
            os.close(fd)
            output_path = Path(temp_name)
            try:
                command = self._command.format(
                    input=audio_path,
                    output=str(output_path),
                    sample_rate=int(sample_rate or 16000),
                )
                completed = subprocess.run(
                    command,
                    shell=True,
                    check=False,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=int(os.environ.get("DIARIZATION_COMMAND_TIMEOUT_SEC", "1800")),
                )
                stdout = completed.stdout or ""
                stderr = completed.stderr or ""
                if completed.returncode != 0:
                    raise RuntimeError((stderr or stdout or f"exit code {completed.returncode}")[:1000])
                raw = output_path.read_text(encoding="utf-8") if output_path.exists() and output_path.stat().st_size else stdout
                turns = parse_diarization_output(raw)
                if not turns:
                    self._last_error = "本地分离命令未输出可解析的 JSON/RTTM 片段"
                else:
                    self._last_error = None
                return turns
            finally:
                output_path.unlink(missing_ok=True)
        except Exception as exc:
            self._last_error = f"本地分离命令执行失败: {exc}"
            logger.error("[DIARIZATION:CLI] %s", self._last_error)
            return []


def get_pyannote_diarizer() -> PyannoteDiarizer:
    return PyannoteDiarizer()


def get_diarization_engine():
    engine = configured_diarization_engine()
    if engine in _PYANNOTE_ENGINES:
        return PyannoteDiarizer()
    if engine == "funasr_campplus":
        return FunASRCampPlusDiarizer()
    if engine == "sherpa_onnx_cli":
        return CommandDiarizer()
    fallback = CommandDiarizer()
    fallback._enabled = False
    fallback._last_error = f"不支持的说话人分离引擎: {engine}"
    return fallback


def reset_pyannote_diarizer() -> None:
    """Drop the cached pyannote pipeline so runtime settings can be reloaded."""
    PyannoteDiarizer._instance = None
    PyannoteDiarizer._pipeline = None
    PyannoteDiarizer._enabled = False
    PyannoteDiarizer._last_error = None
    PyannoteDiarizer._device = "cpu"


def reset_diarization_engine() -> None:
    reset_pyannote_diarizer()
    FunASRCampPlusDiarizer._instance = None
    FunASRCampPlusDiarizer._model = None
    FunASRCampPlusDiarizer._enabled = False
    FunASRCampPlusDiarizer._last_error = None
    FunASRCampPlusDiarizer._device = "cpu"
    CommandDiarizer._instance = None
    CommandDiarizer._enabled = False
    CommandDiarizer._last_error = None
