"""ASR 引擎工厂.

现有 Qwen3-ASR 保持在 engine/asr_engine.py 中,这里只负责按配置延迟加载。
新增引擎都是可选依赖,避免未安装时影响默认 Qwen 路径。
"""
from __future__ import annotations

import logging
import importlib.util
import sys
import threading
import json
import os
from typing import Any

from .plugin_engine import (
    ExternalCommandASREngine,
    get_all_asr_plugin_infos,
    get_asr_plugin_info,
    is_plugin_engine_type,
    plugin_dependency_status,
    plugin_engine_type,
    plugin_id_from_engine_type,
)

logger = logging.getLogger("ASR_Engine")

_CAPABILITY_OVERRIDES_CACHE: dict[str, Any] | None = None
_CAPABILITY_OVERRIDES_CACHE_KEY: tuple[str, str] | None = None
_CAPABILITY_OVERRIDES_WARNED: set[str] = set()


ASR_ENGINE_CONFIG: dict[str, dict[str, Any]] = {
    "qwen3": {
        "name": "Qwen3-ASR",
        "model": "Qwen/Qwen3-ASR-0.6B",
        "description": "高质量中文/多语言 ASR,适合离线高质量转写",
        "description_en": "High-quality Chinese/multilingual ASR for offline transcription",
        "languages": "52种语言/方言",
        "languages_en": "52 languages / dialects",
        "supports_streaming": False,
        "supports_words": True,
        "capabilities": {
            "transcription": True,
            "upload": True,
            "realtime_segmented": True,
            "true_streaming": False,
            "word_timestamps": True,
            "speaker_diarization": False,
            "result_contract": "asr-result-v1",
            "timestamp_granularity": "word_optional",
            "native_metadata": False,
            "recommended_for": ["high_quality_upload", "general_realtime"],
            "notes": "字级时间戳仅在 ASR_WORD_TIMESTAMPS=true 且 forced aligner 加载成功时可用。",
            "notes_en": "Word timestamps require ASR_WORD_TIMESTAMPS=true and a loaded forced aligner.",
        },
    },
    "sensevoice": {
        "name": "SenseVoice-Small",
        "model": "iic/SenseVoiceSmall",
        "description": "快速多语种 ASR,适合上传文件和普通机器快速转写",
        "description_en": "Fast multilingual ASR for upload transcription and everyday local use",
        "languages": "中文/英语/粤语/日语/韩语",
        "languages_en": "Chinese / English / Cantonese / Japanese / Korean",
        "supports_streaming": False,
        "supports_words": False,
        "optional_dependency": "funasr",
        "capabilities": {
            "transcription": True,
            "upload": True,
            "realtime_segmented": True,
            "true_streaming": False,
            "word_timestamps": False,
            "speaker_diarization": False,
            "result_contract": "asr-result-v1",
            "timestamp_granularity": "dynamic",
            "native_metadata": True,
            "recommended_for": ["fast_upload", "lightweight_multilingual"],
            "notes": "不返回字级时间戳;说话人识别仍由独立声纹/pyannote 模块完成。",
            "notes_en": "No word timestamps; speaker diarization is provided by separate speaker/pyannote modules.",
        },
    },
    "sensevoice_zh": {
        "name": "SenseVoice-Small Chinese",
        "model": "iic/SenseVoiceSmall",
        "description": "SenseVoice 中文固定语言模式,适合普通话/粤语会议快速转写,避免自动语种识别抖动",
        "description_en": "SenseVoice with Chinese language fixed, useful for Chinese meeting transcription",
        "languages": "中文/粤语",
        "languages_en": "Chinese / Cantonese",
        "supports_streaming": False,
        "supports_words": False,
        "optional_dependency": "funasr",
        "capabilities": {
            "transcription": True,
            "upload": True,
            "realtime_segmented": True,
            "true_streaming": False,
            "word_timestamps": False,
            "speaker_diarization": False,
            "result_contract": "asr-result-v1",
            "timestamp_granularity": "dynamic",
            "native_metadata": True,
            "recommended_for": ["chinese_fast_upload", "chinese_realtime"],
            "notes": "与 SenseVoice-Small 使用同一模型,但推理时固定 language=zh,中文会议优先测试。",
            "notes_en": "Uses the same SenseVoice-Small weights, but fixes language=zh for Chinese meetings.",
        },
    },
    "paraformer": {
        "name": "Paraformer",
        "model": "paraformer-zh",
        "description": "中文稳定 ASR,适合会议/访谈离线转写",
        "description_en": "Stable Chinese ASR for meeting/interview transcription",
        "languages": "中文/英语",
        "languages_en": "Chinese / English",
        "supports_streaming": False,
        "supports_words": False,
        "optional_dependency": "funasr",
        "capabilities": {
            "transcription": True,
            "upload": True,
            "realtime_segmented": True,
            "true_streaming": False,
            "word_timestamps": False,
            "speaker_diarization": False,
            "result_contract": "asr-result-v1",
            "timestamp_granularity": "dynamic",
            "native_metadata": True,
            "recommended_for": ["chinese_meeting_upload"],
            "notes": "偏中文会议/访谈;导出字幕使用 segment 级时间戳。",
            "notes_en": "Best for Chinese meeting/interview transcription; exports use segment-level timestamps.",
        },
    },
    "paraformer_full": {
        "name": "Paraformer + VAD + Punc",
        "model": "paraformer-zh + fsmn-vad + ct-punc",
        "description": "中文会议完整 FunASR 组合,自动断句并补标点,适合上传文件和较长访谈",
        "description_en": "Full Chinese FunASR stack with VAD and punctuation for longer meetings",
        "languages": "中文/英语",
        "languages_en": "Chinese / English",
        "supports_streaming": False,
        "supports_words": False,
        "optional_dependency": "funasr",
        "capabilities": {
            "transcription": True,
            "upload": True,
            "realtime_segmented": True,
            "true_streaming": False,
            "word_timestamps": False,
            "speaker_diarization": False,
            "result_contract": "asr-result-v1",
            "timestamp_granularity": "dynamic",
            "native_metadata": True,
            "recommended_for": ["chinese_meeting_upload", "punctuated_transcript"],
            "notes": "首次会下载 Paraformer、FSMN-VAD 和 CT-Punc。说话人仍由项目独立声纹/pyannote 流程处理。",
            "notes_en": "Downloads Paraformer, FSMN-VAD, and CT-Punc on first use. Speaker ID remains separate.",
        },
    },
    "paraformer_large": {
        "name": "Paraformer Large",
        "model": "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        "description": "ModelScope Paraformer Large 中文模型,偏离线高质量普通话转写",
        "description_en": "ModelScope Paraformer Large Chinese model for higher-quality offline Mandarin ASR",
        "languages": "中文/英语",
        "languages_en": "Chinese / English",
        "supports_streaming": False,
        "supports_words": False,
        "optional_dependency": "funasr",
        "capabilities": {
            "transcription": True,
            "upload": True,
            "realtime_segmented": True,
            "true_streaming": False,
            "word_timestamps": False,
            "speaker_diarization": False,
            "result_contract": "asr-result-v1",
            "timestamp_granularity": "dynamic",
            "native_metadata": True,
            "recommended_for": ["chinese_high_quality_upload"],
            "notes": "偏离线质量测试;实时场景先用 sensevoice_zh 或 paraformer_full 对照。",
            "notes_en": "Use mainly for offline quality tests; compare with sensevoice_zh or paraformer_full for realtime.",
        },
    },
    "paraformer_spk": {
        "name": "Paraformer + Speaker Tags",
        "model": "paraformer-zh + fsmn-vad + ct-punc + cam++",
        "description": "FunASR 中文转写并输出模型内说话人标签,用于和项目独立说话人分离做对照",
        "description_en": "Chinese FunASR stack with provider speaker tags for comparison with app diarization",
        "languages": "中文/英语",
        "languages_en": "Chinese / English",
        "supports_streaming": False,
        "supports_words": False,
        "optional_dependency": "funasr",
        "capabilities": {
            "transcription": True,
            "upload": True,
            "realtime_segmented": True,
            "true_streaming": False,
            "word_timestamps": False,
            "speaker_diarization": "provider_chunk_labels",
            "result_contract": "asr-result-v1",
            "timestamp_granularity": "dynamic",
            "native_metadata": True,
            "recommended_for": ["diarization_baseline", "chinese_meeting_upload"],
            "notes": "FunASR 内部 speaker 标签按音频块产生,不替代项目的 pyannote/声纹身份识别。",
            "notes_en": "Provider speaker tags are chunk-local and do not replace pyannote or registered speaker identity.",
        },
    },
    "paraformer_streaming": {
        "name": "Paraformer Streaming",
        "model": "paraformer-zh-streaming",
        "description": "中文实时低延迟 ASR,适合 WebSocket 实时字幕",
        "description_en": "Low-latency Chinese streaming ASR for live captions",
        "languages": "中文/英语",
        "languages_en": "Chinese / English",
        "supports_streaming": True,
        "supports_words": False,
        "optional_dependency": "funasr",
        "capabilities": {
            "transcription": True,
            "upload": True,
            "realtime_segmented": True,
            "true_streaming": "adapter_not_yet",
            "word_timestamps": False,
            "speaker_diarization": False,
            "result_contract": "asr-result-v1",
            "timestamp_granularity": "dynamic",
            "native_metadata": True,
            "recommended_for": ["low_latency_future"],
            "notes": "模型支持流式,但当前项目适配层仍按 VAD segment 调用,不是 token-level 真流式输出。",
            "notes_en": "The model supports streaming, but this app currently calls it on VAD segments, not token-level streaming.",
        },
    },
}


FUNASR_ENGINE_TYPES = {
    "sensevoice",
    "sensevoice_zh",
    "paraformer",
    "paraformer_full",
    "paraformer_large",
    "paraformer_spk",
    "paraformer_streaming",
}


VALID_ASR_ENGINE_TYPES = set(ASR_ENGINE_CONFIG)


def _valid_asr_engine_types() -> set[str]:
    return set(ASR_ENGINE_CONFIG) | set(get_all_asr_plugin_infos())


def _is_valid_asr_engine_type(engine_type: str) -> bool:
    return engine_type in ASR_ENGINE_CONFIG or engine_type in get_all_asr_plugin_infos()


ASR_DOWNLOAD_ESTIMATES: dict[str, dict[str, Any]] = {
    # Current pinned HF revision of Qwen/Qwen3-ASR-0.6B.
    "qwen3": {
        "label": "Qwen3-ASR",
        "category": "asr",
        "name": "Qwen3-ASR-0.6B",
        "partial_prefix": ".Qwen3-ASR-0.6B.partial-",
        "total_bytes": 1_880_619_678,
    },
}


def _safe_tree_size(path: str) -> int:
    total = 0
    if not os.path.exists(path):
        return 0
    if os.path.isfile(path):
        try:
            return os.path.getsize(path)
        except OSError:
            return 0
    for root, _, files in os.walk(path):
        for filename in files:
            if filename.endswith((".lock", ".metadata")):
                continue
            try:
                total += os.path.getsize(os.path.join(root, filename))
            except OSError:
                continue
    return total


def _latest_partial_dir(parent: str, prefix: str) -> str | None:
    try:
        candidates = [
            os.path.join(parent, name)
            for name in os.listdir(parent)
            if name.startswith(prefix) and os.path.isdir(os.path.join(parent, name))
        ]
    except OSError:
        return None
    if not candidates:
        return None
    return max(candidates, key=lambda path: os.path.getmtime(path))


def _scan_asr_download_progress(engine_type: str | None, *, active: bool) -> dict[str, Any] | None:
    engine_type = _normalize_engine_type(engine_type)
    estimate = ASR_DOWNLOAD_ESTIMATES.get(engine_type)
    if not estimate:
        return None

    try:
        from app.services.model_resolver import models_root

        parent = os.path.join(models_root(), estimate["category"])
        final_path = os.path.join(parent, estimate["name"])
        partial_path = _latest_partial_dir(parent, estimate["partial_prefix"])
        if active and partial_path:
            source_path = partial_path
            status = "downloading"
        elif os.path.isdir(final_path):
            source_path = final_path
            status = "ready"
        elif partial_path:
            source_path = partial_path
            status = "partial"
        else:
            source_path = None
            status = "pending" if active else "missing"

        downloaded = _safe_tree_size(source_path) if source_path else 0
        total_bytes = int(estimate["total_bytes"])
        percent = min(100.0, round(downloaded / total_bytes * 100, 1)) if total_bytes else None
        return {
            "engine_type": engine_type,
            "label": estimate["label"],
            "status": status,
            "downloaded_bytes": downloaded,
            "total_bytes": total_bytes,
            "percent": percent,
        }
    except Exception as exc:
        logger.debug("[ASR] progress scan failed for %s: %s", engine_type, exc)
        return None


def _asr_download_progress(engine_type: str | None, *, active: bool) -> dict[str, Any] | None:
    engine_type = _normalize_engine_type(engine_type)
    scanned = _scan_asr_download_progress(engine_type, active=active)
    try:
        from app.services.download_progress import get_progress

        live = get_progress(f"asr:{engine_type}")
    except Exception:
        live = None
    if not live:
        return scanned

    estimate = ASR_DOWNLOAD_ESTIMATES.get(engine_type, {})
    total_bytes = live.get("total_bytes") or estimate.get("total_bytes")
    downloaded = max(
        int(live.get("downloaded_bytes") or 0),
        int((scanned or {}).get("downloaded_bytes") or 0),
    )
    percent = (
        min(100.0, round(downloaded / int(total_bytes) * 100, 1))
        if total_bytes
        else live.get("percent")
    )
    return {
        "engine_type": engine_type,
        "label": live.get("label") or estimate.get("label") or engine_type,
        "status": live.get("status") or ("downloading" if active else "partial"),
        "downloaded_bytes": downloaded,
        "total_bytes": total_bytes,
        "percent": percent,
        "message": live.get("message"),
        "error": live.get("error"),
    }


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = base.copy()
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_capability_overrides() -> dict[str, Any]:
    """读取用户自定义 ASR 能力覆盖.

    支持:
    - ASR_CAPABILITIES_JSON='{"qwen3": {"capabilities": {"word_timestamps": false}}}'
    - ASR_CAPABILITIES_FILE='./config/asr_capabilities.json'
    """
    global _CAPABILITY_OVERRIDES_CACHE, _CAPABILITY_OVERRIDES_CACHE_KEY

    raw = os.getenv("ASR_CAPABILITIES_JSON", "").strip()
    file_path = os.getenv("ASR_CAPABILITIES_FILE", "").strip()
    cache_key = (raw, file_path)
    if _CAPABILITY_OVERRIDES_CACHE_KEY == cache_key and _CAPABILITY_OVERRIDES_CACHE is not None:
        return _CAPABILITY_OVERRIDES_CACHE

    if not raw and file_path:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw = f.read().strip()
                cache_key = (raw, file_path)
        except OSError as e:
            _warn_capability_override_once(
                f"file:{file_path}",
                f"[ASR] 读取 ASR_CAPABILITIES_FILE 失败: {e}",
            )
            _CAPABILITY_OVERRIDES_CACHE_KEY = cache_key
            _CAPABILITY_OVERRIDES_CACHE = {}
            return _CAPABILITY_OVERRIDES_CACHE
    if not raw:
        _CAPABILITY_OVERRIDES_CACHE_KEY = cache_key
        _CAPABILITY_OVERRIDES_CACHE = {}
        return _CAPABILITY_OVERRIDES_CACHE
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        _warn_capability_override_once(
            f"json:{raw[:120]}",
            f"[ASR] ASR capabilities JSON 无效: {e}",
        )
        _CAPABILITY_OVERRIDES_CACHE_KEY = cache_key
        _CAPABILITY_OVERRIDES_CACHE = {}
        return _CAPABILITY_OVERRIDES_CACHE
    if not isinstance(data, dict):
        _warn_capability_override_once(
            f"type:{type(data).__name__}",
            "[ASR] ASR capabilities override 必须是 object",
        )
        _CAPABILITY_OVERRIDES_CACHE_KEY = cache_key
        _CAPABILITY_OVERRIDES_CACHE = {}
        return _CAPABILITY_OVERRIDES_CACHE
    _CAPABILITY_OVERRIDES_CACHE_KEY = cache_key
    _CAPABILITY_OVERRIDES_CACHE = data
    return _CAPABILITY_OVERRIDES_CACHE


def _warn_capability_override_once(key: str, message: str) -> None:
    if key in _CAPABILITY_OVERRIDES_WARNED:
        return
    _CAPABILITY_OVERRIDES_WARNED.add(key)
    logger.warning(message)


def _engine_info_with_overrides(engine_type: str) -> dict[str, Any]:
    if is_plugin_engine_type(engine_type):
        info = get_asr_plugin_info(engine_type)
        if info is None:
            return {
                "type": engine_type,
                "name": engine_type,
                "model": engine_type,
                "description": "未找到该外部 ASR 插件",
                "description_en": "External ASR plugin was not found.",
                "customized": False,
            }
    else:
        info = ASR_ENGINE_CONFIG.get(engine_type, ASR_ENGINE_CONFIG["qwen3"]).copy()
    overrides = _load_capability_overrides()
    override = overrides.get(engine_type) or overrides.get(_normalize_engine_type(engine_type))
    if isinstance(override, dict):
        info = _deep_merge(info, override)
        info["customized"] = True
    else:
        info["customized"] = False
    return info


def _dependency_status(engine_type: str) -> dict[str, Any]:
    """检查 ASR 引擎运行时依赖是否可用."""
    engine_type = _normalize_engine_type(engine_type)
    if is_plugin_engine_type(engine_type):
        return plugin_dependency_status(engine_type)
    info = _engine_info_with_overrides(engine_type)
    dep = info.get("optional_dependency")
    if not dep:
        return {"available": True}

    if importlib.util.find_spec(dep) is not None:
        return {"available": True, "dependency": dep}

    pyver = f"{sys.version_info.major}.{sys.version_info.minor}"
    hint = f"请先安装依赖: pip install {dep}"
    hint_en = f"Install dependency first: pip install {dep}"
    if dep == "funasr" and sys.version_info >= (3, 13):
        hint = (
            "当前 Python 3.13 环境未安装 funasr。FunASR 在 Windows/Python 3.13 "
            "可能因 editdistance 无预编译 wheel 安装失败;建议使用 Python 3.10-3.12 "
            "环境安装 requirements.txt 后再切换。"
        )
        hint_en = (
            "FunASR is not installed in the current Python 3.13 environment. "
            "On Windows/Python 3.13, FunASR may fail to install because editdistance "
            "has no prebuilt wheel. Use Python 3.10-3.12, install requirements.txt, "
            "then switch again."
        )
    return {
        "available": False,
        "dependency": dep,
        "python": pyver,
        "reason": f"missing dependency: {dep}",
        "install_hint": hint,
        "install_hint_en": hint_en,
    }


def _normalize_engine_type(engine_type: str | None) -> str:
    if not engine_type:
        return "qwen3"
    normalized = engine_type.lower().strip()
    if normalized.startswith("external:"):
        return plugin_engine_type(normalized.split(":", 1)[1])
    if normalized.startswith("plugin:"):
        return plugin_engine_type(normalized.split(":", 1)[1])
    normalized = normalized.replace("-", "_")
    aliases = {
        "qwen": "qwen3",
        "qwen3_asr": "qwen3",
        "funasr_sensevoice": "sensevoice",
        "sensevoice_small": "sensevoice",
        "sensevoice_cn": "sensevoice_zh",
        "sensevoice_chinese": "sensevoice_zh",
        "sensevoice_zh_cn": "sensevoice_zh",
        "funasr_sensevoice_zh": "sensevoice_zh",
        "funasr_paraformer": "paraformer",
        "paraformer_zh": "paraformer",
        "paraformer_punc": "paraformer_full",
        "paraformer_vad_punc": "paraformer_full",
        "paraformer_full_stack": "paraformer_full",
        "paraformer_large_vad_punc": "paraformer_large",
        "paraformer_speaker": "paraformer_spk",
        "paraformer_spk": "paraformer_spk",
        "paraformer_speaker_tags": "paraformer_spk",
        "funasr_streaming": "paraformer_streaming",
        "streaming": "paraformer_streaming",
    }
    return aliases.get(normalized, normalized)


def get_asr_engine(engine_type: str | None = None):
    """按配置创建 ASR 引擎实例."""
    if engine_type is None:
        from app.config import config
        engine_type = config.audio.asr_engine

    engine_type = _normalize_engine_type(engine_type)
    if not _is_valid_asr_engine_type(engine_type):
        raise ValueError(
            f"Invalid ASR_ENGINE={engine_type!r}. Valid: {sorted(_valid_asr_engine_types())}"
        )

    logger.info(f"[ASR] 选择引擎: {engine_type}")
    if is_plugin_engine_type(engine_type):
        return ExternalCommandASREngine(engine_type)
    if engine_type == "qwen3":
        from engine.asr_engine import ASREngine
        return ASREngine()
    if engine_type in FUNASR_ENGINE_TYPES:
        from .funasr_engine import FunASREngine
        return FunASREngine(kind=engine_type)
    raise AssertionError(f"Unhandled ASR engine: {engine_type}")


def get_asr_engine_info(engine_type: str | None = None) -> dict[str, Any]:
    """获取某个或当前 ASR 引擎信息."""
    if engine_type is None:
        engine_type = get_asr_manager().current_type
    engine_type = _normalize_engine_type(engine_type)
    info = _engine_info_with_overrides(engine_type)
    info["type"] = engine_type
    if engine_type == "qwen3":
        try:
            from app.config import config
            info["word_timestamps_enabled"] = bool(config.audio.asr_word_timestamps)
        except Exception:
            info["word_timestamps_enabled"] = False
    else:
        info["word_timestamps_enabled"] = False
    info.update(_dependency_status(engine_type))
    return info


def get_all_asr_engines() -> dict[str, Any]:
    engines = {}
    for key in ASR_ENGINE_CONFIG:
        info = _engine_info_with_overrides(key)
        info["type"] = key
        info.update(_dependency_status(key))
        engines[key] = info
    engines.update(get_all_asr_plugin_infos())
    manager = get_asr_manager()
    pending = manager.pending_type
    return {
        "current": manager.current_type,
        "engines": engines,
        "switching": manager.switching,
        "pending": pending,
        "cached": manager.cached_types,
        "progress": _asr_download_progress(pending, active=manager.switching) if pending else None,
    }


class ASREngineManager:
    """ASR 引擎管理器.

    切换策略:
    - 旧引擎继续作为 current 服务请求。
    - 新引擎在 switch lock 内下载/加载。
    - 加载成功后才替换 current。
    - 加载失败时 current 不变。
    """

    _instance: "ASREngineManager | None" = None
    _initialized = False
    _lock = threading.Lock()

    def __new__(cls) -> "ASREngineManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if ASREngineManager._initialized:
            return
        with ASREngineManager._lock:
            if ASREngineManager._initialized:
                return
            from app.config import config
            self._engine_cache: dict[str, Any] = {}
            self._current_type = _normalize_engine_type(
                os.environ.get("ASR_ENGINE") or config.audio.asr_engine
            )
            self._current_engine: Any | None = None
            self._switch_lock = threading.Lock()
            self._state_lock = threading.Lock()
            self._switching = False
            self._pending_type: str | None = None
            self._max_cache = 1
            ASREngineManager._initialized = True
            logger.info(f"[ASR] Manager 初始化完成, 默认引擎: {self._current_type}")

    def get_engine(self) -> Any:
        """获取当前 ASR 引擎."""
        if self._current_engine is None:
            with self._switch_lock:
                if self._current_engine is None:
                    self._current_engine = self._load_engine(self._current_type)
        return self._current_engine

    def _load_engine(self, engine_type: str) -> Any:
        engine_type = _normalize_engine_type(engine_type)
        if engine_type in self._engine_cache:
            logger.info(f"[ASR] 使用缓存引擎: {engine_type}")
            return self._engine_cache[engine_type]
        engine = get_asr_engine(engine_type)
        if not getattr(engine, "initialized", True):
            raise RuntimeError(f"ASR 引擎 {engine_type} 初始化失败,不可用")
        self._engine_cache[engine_type] = engine
        return engine

    def _evict_except_current(self) -> None:
        if len(self._engine_cache) <= self._max_cache:
            return
        for key in list(self._engine_cache.keys()):
            if key == self._current_type:
                continue
            logger.info(f"[ASR] evict 引擎缓存: {key}")
            self._engine_cache.pop(key, None)
        try:
            import gc
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        except Exception:
            pass

    def _forget_engine_singleton(self, engine_type: str) -> None:
        """Drop provider-level singleton references so a same-type reload is real."""
        engine_type = _normalize_engine_type(engine_type)
        try:
            if is_plugin_engine_type(engine_type):
                return
            if engine_type == "qwen3":
                from engine.asr_engine import ASREngine

                ASREngine._instance = None
                return
            if engine_type in FUNASR_ENGINE_TYPES:
                from .funasr_engine import FunASREngine

                FunASREngine._instances.pop(engine_type, None)
        except Exception as exc:
            logger.debug("[ASR] forget singleton failed for %s: %s", engine_type, exc)

    def switch_engine(self, engine_type: str) -> dict[str, Any]:
        """下载/加载新 ASR,成功后原子切换."""
        engine_type = _normalize_engine_type(engine_type)
        previous_type = self._current_type

        if not _is_valid_asr_engine_type(engine_type):
            error = f"Invalid ASR engine type: {engine_type}. Valid: {sorted(_valid_asr_engine_types())}"
            return {"success": False, "error": error, "engine_type": engine_type}

        dep_status = _dependency_status(engine_type)
        if not dep_status.get("available", True):
            error = dep_status.get("install_hint") or dep_status.get("reason") or "ASR dependency unavailable"
            logger.warning(f"[ASR] 切换前依赖检查失败: {engine_type}: {error}")
            return {
                "success": False,
                "error": error,
                "engine_type": engine_type,
                "previous_type": previous_type,
                "current": self._current_type,
                "dependency": dep_status.get("dependency"),
                "available": False,
            }

        if engine_type == previous_type:
            return {
                "success": True,
                "engine_type": engine_type,
                "previous_type": previous_type,
                "engine_info": get_asr_engine_info(engine_type),
                "already_active": True,
                "downloaded": engine_type in self._engine_cache,
            }

        if not self._switch_lock.acquire(blocking=False):
            return {
                "success": False,
                "error": f"ASR engine switch already in progress: {self._pending_type}",
                "engine_type": engine_type,
                "current": self._current_type,
                "pending": self._pending_type,
            }

        try:
            with self._state_lock:
                self._switching = True
                self._pending_type = engine_type

            was_cached = engine_type in self._engine_cache
            logger.info(f"[ASR] 开始切换: {previous_type} -> {engine_type} cached={was_cached}")
            new_engine = self._load_engine(engine_type)
            self._current_engine = new_engine
            self._current_type = engine_type
            self._evict_except_current()

            logger.info(f"[ASR] 切换成功: {previous_type} -> {engine_type}")
            return {
                "success": True,
                "engine_type": engine_type,
                "previous_type": previous_type,
                "engine_info": get_asr_engine_info(engine_type),
                "downloaded": was_cached,
                "switched": True,
            }
        except Exception as e:
            logger.exception(f"[ASR] 切换失败: {previous_type} -> {engine_type}: {e}")
            return {
                "success": False,
                "error": str(e),
                "engine_type": engine_type,
                "previous_type": previous_type,
                "current": self._current_type,
            }
        finally:
            with self._state_lock:
                self._switching = False
                self._pending_type = None
            self._switch_lock.release()

    def reload_current(self) -> dict[str, Any]:
        """Reload the active ASR type, used when runtime settings such as device change."""
        engine_type = self._current_type

        dep_status = _dependency_status(engine_type)
        if not dep_status.get("available", True):
            error = dep_status.get("install_hint") or dep_status.get("reason") or "ASR dependency unavailable"
            logger.warning("[ASR] reload dependency check failed: %s: %s", engine_type, error)
            return {
                "success": False,
                "error": error,
                "engine_type": engine_type,
                "current": self._current_type,
                "dependency": dep_status.get("dependency"),
                "available": False,
            }

        if not self._switch_lock.acquire(blocking=False):
            return {
                "success": False,
                "error": f"ASR engine reload already in progress: {self._pending_type}",
                "engine_type": engine_type,
                "current": self._current_type,
                "pending": self._pending_type,
            }

        try:
            with self._state_lock:
                self._switching = True
                self._pending_type = engine_type

            logger.info("[ASR] reload current engine: %s", engine_type)
            self._engine_cache.pop(engine_type, None)
            self._forget_engine_singleton(engine_type)
            new_engine = self._load_engine(engine_type)
            self._current_engine = new_engine
            self._evict_except_current()

            logger.info("[ASR] reload current engine success: %s", engine_type)
            return {
                "success": True,
                "engine_type": engine_type,
                "previous_type": engine_type,
                "engine_info": get_asr_engine_info(engine_type),
                "downloaded": False,
                "reloaded": True,
            }
        except Exception as e:
            logger.exception("[ASR] reload current engine failed: %s: %s", engine_type, e)
            return {
                "success": False,
                "error": str(e),
                "engine_type": engine_type,
                "current": self._current_type,
            }
        finally:
            with self._state_lock:
                self._switching = False
                self._pending_type = None
            self._switch_lock.release()

    def get_all_engines_info(self) -> dict[str, Any]:
        return get_all_asr_engines()

    def get_engine_info(self) -> dict[str, Any]:
        return get_asr_engine_info(self._current_type)

    @property
    def current_type(self) -> str:
        return self._current_type

    @property
    def switching(self) -> bool:
        return self._switching

    @property
    def pending_type(self) -> str | None:
        return self._pending_type

    @property
    def cached_types(self) -> list[str]:
        return list(self._engine_cache.keys())

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls._instance = None
            cls._initialized = False


def get_asr_manager() -> ASREngineManager:
    return ASREngineManager()
