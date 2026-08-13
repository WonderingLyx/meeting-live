"""Unified model source configuration.

The runtime still supports the legacy .env and settings database values, but
new settings saved from the UI are written to config/model-settings.json.  API
keys stay out of SQLite and are masked before they are returned to the
frontend.
"""
from __future__ import annotations

import json
import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger("Matrix_Model_Config")

MODEL_SETTINGS_FILE_ENV = "MODEL_SETTINGS_FILE"
_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_PATH = _ROOT / "config" / "model-settings.json"

MODEL_SOURCE_PROVIDERS = [
    {
        "key": "modelscope",
        "label": "ModelScope",
        "endpoint": "https://modelscope.cn",
        "token_env": "MODELSCOPE_API_TOKEN",
        "scope": "asr,speaker",
    },
    {
        "key": "huggingface",
        "label": "Hugging Face",
        "endpoint": "https://huggingface.co",
        "token_env": "HF_TOKEN",
        "scope": "asr,speaker,diarization",
    },
    {
        "key": "local",
        "label": "Local cache/path",
        "endpoint": "file://./models",
        "token_env": "",
        "scope": "asr,speaker",
    },
    {
        "key": "custom",
        "label": "Custom",
        "endpoint": "",
        "token_env": "",
        "scope": "asr,speaker,diarization",
    },
]


def model_settings_path() -> Path:
    raw = os.environ.get(MODEL_SETTINGS_FILE_ENV)
    if raw:
        return Path(raw).expanduser().resolve()
    return _DEFAULT_PATH


def model_settings_exists() -> bool:
    return model_settings_path().is_file()


def default_model_settings() -> dict[str, Any]:
    """Return non-secret defaults derived from the current runtime config."""
    from app.config import config

    return {
        "version": 1,
        "asr": {
            "provider": os.environ.get("ASR_PROVIDER", "modelscope"),
            "endpoint": os.environ.get("ASR_ENDPOINT", "https://modelscope.cn"),
            "api_key": "",
            "model": config.audio.asr_engine,
            "device": config.audio.asr_device,
            "word_timestamps": bool(config.audio.asr_word_timestamps),
            "load_timeout_sec": int(config.audio.asr_load_timeout_sec),
        },
        "speaker": {
            "provider": os.environ.get("SPEAKER_PROVIDER", "modelscope"),
            "endpoint": os.environ.get("SPEAKER_ENDPOINT", "https://modelscope.cn"),
            "api_key": "",
            "model": config.speaker.engine_type,
            "device": os.environ.get("SPEAKER_DEVICE", "auto"),
        },
        "diarization": {
            "provider": os.environ.get("DIARIZATION_PROVIDER", "huggingface"),
            "endpoint": os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
            "api_key": "",
            "model": os.environ.get(
                "PYANNOTE_MODEL",
                "pyannote/speaker-diarization-community-1",
            ),
            "device": config.speaker.diarization_device,
        },
        "llm": {
            "provider": os.environ.get("LLM_PROVIDER", "ollama"),
            "endpoint": config.llm.endpoint,
            "api_key": "",
            "model": config.llm.model,
            "enabled": bool(config.llm.enabled),
            "allow_public": bool(config.llm.allow_public),
            "timeout_sec": int(config.llm.timeout_sec),
            "max_input_tokens": int(config.llm.max_input_tokens),
            "mock": bool(config.llm.mock),
        },
    }


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def read_raw_model_settings() -> dict[str, Any]:
    path = model_settings_path()
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取模型配置文件 {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"模型配置文件必须是 JSON object: {path}")
    return payload


def _read_raw_or_empty() -> dict[str, Any]:
    try:
        return read_raw_model_settings()
    except ValueError as exc:
        logger.warning("%s", exc)
        return {}


def _env_secret_for(section: str, provider: str) -> str:
    provider = (provider or "").strip().lower()
    if section == "llm":
        return os.environ.get("LLM_API_KEY", "")
    if section == "diarization" or provider == "huggingface":
        return (
            os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGINGFACE_TOKEN")
            or os.environ.get("HUGGINGFACE_HUB_TOKEN")
            or ""
        )
    if provider == "modelscope" or section in {"asr", "speaker"}:
        return (
            os.environ.get(f"{section.upper()}_API_KEY")
            or os.environ.get("MODELSCOPE_API_TOKEN")
            or os.environ.get("MODELSCOPE_TOKEN")
            or ""
        )
    return ""


def read_model_settings(*, include_env_secrets: bool = False) -> dict[str, Any]:
    settings = _deep_merge(default_model_settings(), _read_raw_or_empty())
    if include_env_secrets:
        for section_name in ("asr", "speaker", "diarization", "llm"):
            section = settings.get(section_name)
            if not isinstance(section, dict):
                continue
            env_secret = _env_secret_for(section_name, str(section.get("provider") or ""))
            if env_secret:
                section["api_key"] = env_secret
    return settings


def section_has_file_config(section: str) -> bool:
    return isinstance(_read_raw_or_empty().get(section), dict)


def _trimmed(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value).strip()


def _to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "on"}


def _mask_secret(value: str | None) -> str | None:
    if not value:
        return None
    if len(value) <= 10:
        return value[:2] + "***"
    return value[:6] + "***" + value[-4:]


def public_model_settings() -> dict[str, Any]:
    raw = _read_raw_or_empty()
    settings = read_model_settings(include_env_secrets=True)
    public: dict[str, Any] = {
        "config_path": str(model_settings_path()),
        "providers": MODEL_SOURCE_PROVIDERS,
    }
    for section_name in ("asr", "speaker", "diarization", "llm"):
        section = dict(settings.get(section_name) or {})
        api_key = _trimmed(section.pop("api_key", ""))
        section["api_key_configured"] = bool(api_key)
        section["api_key_preview"] = _mask_secret(api_key)
        section["config_source"] = "model-settings" if isinstance(raw.get(section_name), dict) else "env"
        public[section_name] = section
    return public


def update_model_section(
    section: str,
    values: Mapping[str, Any],
    *,
    clear_api_key: bool = False,
) -> dict[str, Any]:
    if section not in {"asr", "speaker", "diarization", "llm"}:
        raise ValueError(f"unsupported model settings section: {section}")

    raw = _read_raw_or_empty()
    raw["version"] = int(raw.get("version") or 1)
    base_section = default_model_settings()[section]
    existing = raw.get(section) if isinstance(raw.get(section), dict) else {}
    merged = _deep_merge(base_section, existing)

    updates = dict(values)
    incoming_key = updates.pop("api_key", None)
    for key, value in updates.items():
        if value is None:
            continue
        if isinstance(value, str):
            merged[key] = value.strip()
        else:
            merged[key] = value

    if clear_api_key:
        merged["api_key"] = ""
    elif incoming_key is not None:
        candidate = _trimmed(incoming_key)
        if candidate:
            merged["api_key"] = candidate

    raw[section] = merged
    write_model_settings(raw)
    return read_model_settings(include_env_secrets=True)[section]


def write_model_settings(payload: Mapping[str, Any]) -> None:
    path = model_settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    data = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    tmp.write_text(data + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _set_env_if_value(key: str, value: Any) -> None:
    text = _trimmed(value)
    if text:
        os.environ[key] = text


def _set_provider_environment(section_name: str, section: Mapping[str, Any]) -> None:
    provider = _trimmed(section.get("provider")).lower()
    endpoint = _trimmed(section.get("endpoint")).rstrip("/")
    api_key = _trimmed(section.get("api_key"))
    if endpoint:
        os.environ[f"{section_name.upper()}_ENDPOINT"] = endpoint
    if api_key:
        os.environ[f"{section_name.upper()}_API_KEY"] = api_key

    if provider == "huggingface":
        _set_env_if_value("HF_ENDPOINT", endpoint)
        _set_env_if_value("HF_TOKEN", api_key)
        _set_env_if_value("HUGGINGFACE_HUB_TOKEN", api_key)
        return

    if provider == "modelscope":
        _set_env_if_value("MODELSCOPE_ENDPOINT", endpoint)
        _set_env_if_value("MODELSCOPE_API_TOKEN", api_key)
        _set_env_if_value("MODELSCOPE_TOKEN", api_key)


def apply_model_settings_to_runtime(settings: Mapping[str, Any] | None = None) -> None:
    """Apply saved model settings to process config/env before engines load."""
    from app.config import config

    try:
        effective = settings or read_model_settings(include_env_secrets=True)
    except Exception as exc:
        logger.warning("模型配置加载失败，继续使用 .env/default: %s", exc)
        return

    asr = effective.get("asr") if isinstance(effective.get("asr"), Mapping) else {}
    speaker = effective.get("speaker") if isinstance(effective.get("speaker"), Mapping) else {}
    diarization = (
        effective.get("diarization")
        if isinstance(effective.get("diarization"), Mapping)
        else {}
    )
    llm = effective.get("llm") if isinstance(effective.get("llm"), Mapping) else {}

    if asr:
        _set_provider_environment("asr", asr)
        model = _trimmed(asr.get("model") or asr.get("engine"))
        device = _trimmed(asr.get("device"), config.audio.asr_device).lower()
        if model:
            config.audio.asr_engine = model.lower()
            os.environ["ASR_ENGINE"] = config.audio.asr_engine
        if device:
            config.audio.asr_device = device
            os.environ["ASR_DEVICE"] = device
        if "word_timestamps" in asr:
            config.audio.asr_word_timestamps = _to_bool(asr.get("word_timestamps"))
            os.environ["ASR_WORD_TIMESTAMPS"] = str(config.audio.asr_word_timestamps).lower()
        if asr.get("load_timeout_sec") is not None:
            try:
                config.audio.asr_load_timeout_sec = int(asr["load_timeout_sec"])
                os.environ["ASR_LOAD_TIMEOUT_SEC"] = str(config.audio.asr_load_timeout_sec)
            except (TypeError, ValueError):
                pass

    if speaker:
        _set_provider_environment("speaker", speaker)
        model = _trimmed(speaker.get("model") or speaker.get("engine"))
        if model:
            config.speaker.engine_type = model.lower()
            os.environ["SPEAKER_ENGINE"] = config.speaker.engine_type
        _set_env_if_value("SPEAKER_DEVICE", speaker.get("device"))

    if diarization:
        _set_provider_environment("diarization", diarization)
        model = _trimmed(diarization.get("model"))
        device = _trimmed(diarization.get("device"), config.speaker.diarization_device).lower()
        if model:
            os.environ["PYANNOTE_MODEL"] = model
        if device:
            config.speaker.diarization_device = device
            os.environ["PYANNOTE_DEVICE"] = device

    if llm:
        _set_provider_environment("llm", llm)
        if "enabled" in llm:
            config.llm.enabled = _to_bool(llm.get("enabled"), config.llm.enabled)
        config.llm.endpoint = _trimmed(llm.get("endpoint"), config.llm.endpoint)
        config.llm.model = _trimmed(llm.get("model"), config.llm.model)
        api_key = _trimmed(llm.get("api_key"))
        if api_key:
            config.llm.api_key = api_key
            os.environ["LLM_API_KEY"] = api_key
        if llm.get("timeout_sec") is not None:
            try:
                config.llm.timeout_sec = int(llm["timeout_sec"])
            except (TypeError, ValueError):
                pass
        if llm.get("max_input_tokens") is not None:
            try:
                config.llm.max_input_tokens = int(llm["max_input_tokens"])
            except (TypeError, ValueError):
                pass
        if "mock" in llm:
            config.llm.mock = _to_bool(llm.get("mock"), config.llm.mock)
        if "allow_public" in llm:
            config.llm.allow_public = _to_bool(llm.get("allow_public"), config.llm.allow_public)
