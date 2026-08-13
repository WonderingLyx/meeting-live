"""设置/存储相关端点"""
import asyncio
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from app.config import config
from app.services.model_config import (
    apply_model_settings_to_runtime,
    public_model_settings,
    read_raw_model_settings,
    read_model_settings,
    section_has_file_config,
    update_model_section,
    write_model_settings,
)

router = APIRouter()


class AsrSwitchRequest(BaseModel):
    """ASR 切换请求."""
    engine_type: str = Field(..., min_length=1, max_length=128, description="ASR 引擎类型")


class AsrSettingsRequest(BaseModel):
    """ASR runtime settings."""
    provider: Optional[str] = Field(None, min_length=1, max_length=40)
    endpoint: Optional[str] = Field(None, max_length=500)
    api_key: Optional[str] = Field(None, max_length=500)
    model: Optional[str] = Field(None, min_length=1, max_length=200)
    device: str = Field("auto", min_length=2, max_length=16, description="auto | cpu | cuda")
    word_timestamps: Optional[bool] = None
    load_timeout_sec: Optional[int] = Field(None, ge=10, le=600)
    reload_current: bool = True


class SpeakerSettingsRequest(BaseModel):
    """Speaker embedding model settings."""
    provider: str = Field("modelscope", min_length=1, max_length=40)
    endpoint: Optional[str] = Field(None, max_length=500)
    api_key: Optional[str] = Field(None, max_length=500)
    model: str = Field("campplus", min_length=1, max_length=128)
    device: str = Field("auto", min_length=2, max_length=16)


class ModelSourceTestRequest(BaseModel):
    section: str = Field(..., pattern="^(asr|speaker|diarization)$")
    provider: str = Field("custom", min_length=1, max_length=40)
    endpoint: Optional[str] = Field(None, max_length=500)
    api_key: Optional[str] = Field(None, max_length=500)


class DiarizationSettingsRequest(BaseModel):
    engine: Optional[str] = Field(None, min_length=1, max_length=80)
    provider: Optional[str] = Field(None, min_length=1, max_length=40)
    endpoint: Optional[str] = Field(None, max_length=500)
    api_key: Optional[str] = Field(None, max_length=500)
    hf_token: Optional[str] = Field(None, max_length=500)
    device: str = Field("auto", pattern="^(auto|cpu|cuda)$")
    model_id: Optional[str] = Field(None, min_length=1, max_length=200)
    command: Optional[str] = Field(None, max_length=1200)


class AsrSwitchResponse(BaseModel):
    success: bool
    engine_type: str
    previous_type: Optional[str] = None
    engine_info: Optional[Dict[str, Any]] = None
    already_active: bool = False
    downloaded: bool = False
    switched: bool = False
    error: Optional[str] = None


ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
PYANNOTE_DEFAULT_MODEL = "pyannote/speaker-diarization-community-1"
PYANNOTE_TERMS_URL = "https://huggingface.co/pyannote/speaker-diarization-community-1"
HF_TOKEN_URL = "https://huggingface.co/settings/tokens"


def _mask_secret(value: str | None) -> str | None:
    if not value:
        return None
    if len(value) <= 10:
        return value[:2] + "***"
    return value[:6] + "***" + value[-4:]


def _read_env_value(key: str) -> str | None:
    if not ENV_PATH.is_file():
        return None
    try:
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            k, v = stripped.split("=", 1)
            if k.strip() == key:
                return v.strip().strip('"').strip("'")
    except OSError:
        return None
    return None


def _write_env_values(values: dict[str, str]) -> None:
    lines = ENV_PATH.read_text(encoding="utf-8").splitlines() if ENV_PATH.exists() else []
    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in values:
                out.append(f"{key}={values[key]}")
                seen.add(key)
                continue
        out.append(line)
    if out and out[-1].strip():
        out.append("")
    for key, value in values.items():
        if key not in seen:
            out.append(f"{key}={value}")
    ENV_PATH.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")


def _diarization_status(*, load: bool = False) -> dict[str, Any]:
    model_settings = read_model_settings(include_env_secrets=True)
    diarization_settings = model_settings.get("diarization", {})
    env_token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("MODELSCOPE_API_TOKEN")
        or os.environ.get("DIARIZATION_API_KEY")
    )
    configured_token = (diarization_settings.get("api_key") or "").strip()
    file_token = (
        _read_env_value("DIARIZATION_API_KEY")
        or _read_env_value("HF_TOKEN")
        or _read_env_value("MODELSCOPE_API_TOKEN")
    )
    token = env_token or configured_token or file_token
    try:
        from app.services.pyannote_diarization import (
            DEFAULT_DIARIZATION_ENGINE,
            default_diarization_model,
            get_all_diarization_engines,
            get_diarization_engine,
            get_diarization_engine_info,
            normalize_diarization_engine,
            PyannoteDiarizer,
            FunASRCampPlusDiarizer,
            CommandDiarizer,
        )
    except Exception as exc:
        return {
            "enabled": False,
            "engine": diarization_settings.get("engine") or "pyannote_community",
            "engine_info": {},
            "engines": {},
            "token_configured": bool(token),
            "token_preview": _mask_secret(token),
            "provider": diarization_settings.get("provider") or "huggingface",
            "endpoint": diarization_settings.get("endpoint") or os.environ.get("HF_ENDPOINT") or "https://huggingface.co",
            "api_key_configured": bool(token),
            "api_key_preview": _mask_secret(token),
            "config_path": public_model_settings()["config_path"],
            "device": diarization_settings.get("device") or config.speaker.diarization_device,
            "loaded_device": "cpu",
            "model_id": diarization_settings.get("model") or PYANNOTE_DEFAULT_MODEL,
            "command": diarization_settings.get("command") or "",
            "model_revision": None,
            "last_error": str(exc),
            "env_path": str(ENV_PATH),
            "terms_url": PYANNOTE_TERMS_URL,
            "token_url": HF_TOKEN_URL,
        }
    has_diarization_file_config = section_has_file_config("diarization")
    engine = normalize_diarization_engine(
        (diarization_settings.get("engine") if has_diarization_file_config else None)
        or os.environ.get("DIARIZATION_ENGINE")
        or diarization_settings.get("engine")
        or DEFAULT_DIARIZATION_ENGINE
    )
    engine_info = get_diarization_engine_info(engine)
    model_id = (
        (diarization_settings.get("model") if has_diarization_file_config else None)
        or os.environ.get("DIARIZATION_MODEL")
        or os.environ.get("PYANNOTE_MODEL")
        or diarization_settings.get("model")
        or engine_info.get("model")
        or _read_env_value("PYANNOTE_MODEL")
        or default_diarization_model(engine)
    ).strip() or default_diarization_model(engine)
    model_revision = (
        os.environ.get("PYANNOTE_MODEL_REVISION")
        or _read_env_value("PYANNOTE_MODEL_REVISION")
        or None
    )
    try:
        if load:
            diarizer = get_diarization_engine()
            enabled = bool(diarizer.enabled)
            last_error = diarizer.last_error
            loaded_device = diarizer.device
        else:
            if engine in {"pyannote", "pyannote_community", "pyannote_custom"}:
                enabled = bool(PyannoteDiarizer._enabled)
                last_error = PyannoteDiarizer._last_error
                loaded_device = PyannoteDiarizer._device
            elif engine_info.get("runtime") == "funasr":
                state = FunASRCampPlusDiarizer.state_for(engine)
                enabled = bool(state.get("enabled"))
                last_error = state.get("last_error")
                loaded_device = str(state.get("device") or "cpu")
            else:
                enabled = bool(CommandDiarizer._enabled)
                last_error = CommandDiarizer._last_error
                loaded_device = CommandDiarizer._device
            if engine_info.get("requires_token") and not token and not last_error:
                last_error = "HF_TOKEN 未设置"
            if not engine_info.get("dependency_available") and not last_error:
                last_error = engine_info.get("install_hint") or "当前分离引擎依赖不可用"
    except Exception as exc:
        enabled = False
        last_error = str(exc)
        loaded_device = "cpu"
    return {
        "enabled": enabled,
        "engine": engine,
        "engine_info": engine_info,
        "engines": get_all_diarization_engines(),
        "token_configured": bool(env_token or configured_token or file_token),
        "token_preview": _mask_secret(env_token or configured_token or file_token),
        "token_source": (
            "environment"
            if env_token
            else ("model-settings" if configured_token else ("env_file" if file_token else None))
        ),
        "provider": (
            diarization_settings.get("provider")
            or engine_info.get("provider")
            or "huggingface"
        ),
        "endpoint": (
            diarization_settings.get("endpoint")
            or (None if has_diarization_file_config else os.environ.get("DIARIZATION_ENDPOINT"))
            or os.environ.get("HF_ENDPOINT")
            or engine_info.get("endpoint")
            or "https://huggingface.co"
        ),
        "api_key_configured": bool(env_token or configured_token or file_token),
        "api_key_preview": _mask_secret(env_token or configured_token or file_token),
        "config_path": public_model_settings()["config_path"],
        "device": diarization_settings.get("device") or config.speaker.diarization_device,
        "loaded_device": loaded_device,
        "model_id": model_id,
        "command": diarization_settings.get("command") or os.environ.get("DIARIZATION_COMMAND") or "",
        "model_revision": model_revision,
        "last_error": last_error,
        "env_path": str(ENV_PATH),
        "terms_url": engine_info.get("terms_url") or PYANNOTE_TERMS_URL,
        "token_url": HF_TOKEN_URL,
    }


def _torch_device_status() -> dict[str, Any]:
    status: dict[str, Any] = {
        "torch_available": False,
        "torch_version": None,
        "backend": "cpu",
        "cuda_available": False,
        "cuda_version": None,
        "hip_version": None,
        "mps_available": False,
        "devices": [],
        "error": None,
    }
    try:
        import torch

        hip_version = getattr(torch.version, "hip", None)
        cuda_version = getattr(torch.version, "cuda", None)
        cuda_available = bool(torch.cuda.is_available())
        mps_available = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        devices: list[dict[str, Any]] = []
        if cuda_available:
            for index in range(torch.cuda.device_count()):
                try:
                    name = torch.cuda.get_device_name(index)
                except Exception:
                    name = f"cuda:{index}"
                devices.append({"index": index, "name": name})
        status.update(
            {
                "torch_available": True,
                "torch_version": getattr(torch, "__version__", None),
                "backend": "rocm" if hip_version else ("cuda" if cuda_version else ("mps" if mps_available else "cpu")),
                "cuda_available": cuda_available,
                "cuda_version": cuda_version,
                "hip_version": hip_version,
                "mps_available": mps_available,
                "devices": devices,
            }
        )
    except Exception as exc:
        status["error"] = str(exc)
    return status


def _normalize_asr_device(raw: str | None) -> str:
    value = (raw or "auto").strip().lower()
    aliases = {
        "gpu": "cuda",
        "rocm": "cuda",
        "hip": "cuda",
        "amd": "cuda",
        "nvidia": "cuda",
    }
    value = aliases.get(value, value)
    if value not in {"auto", "cpu", "cuda", "mps"}:
        raise HTTPException(status_code=400, detail="ASR_DEVICE 只支持 auto / cpu / cuda / mps")
    return value


def _validate_asr_device(device: str, device_status: dict[str, Any]) -> None:
    if device == "cuda" and not device_status.get("cuda_available"):
        raise HTTPException(
            status_code=400,
            detail=(
                "当前 PyTorch 没有检测到可用 GPU。AMD ROCm 可用时 torch.version.hip "
                "应不为空，且 torch.cuda.is_available() 应为 True；请先安装 ROCm 版 PyTorch。"
            ),
        )
    if device == "mps" and not device_status.get("mps_available"):
        raise HTTPException(status_code=400, detail="当前 PyTorch 没有检测到可用 MPS 设备。")


def _default_probe_url(provider: str, endpoint: str) -> str:
    provider = provider.strip().lower()
    endpoint = endpoint.strip().rstrip("/")
    if provider == "huggingface":
        return f"{endpoint}/api/whoami-v2"
    if provider == "modelscope":
        return f"{endpoint}/api/v1/models"
    return endpoint


async def _probe_model_source(section: str, provider: str, endpoint: str, api_key: str | None) -> dict[str, Any]:
    provider = (provider or "custom").strip().lower()
    endpoint = (endpoint or "").strip()
    if provider == "local" or endpoint.startswith("file://"):
        raw_path = endpoint.removeprefix("file://").strip() or "./models"
        path = (Path(__file__).resolve().parents[2] / raw_path).resolve() if raw_path.startswith(".") else Path(raw_path).expanduser()
        return {
            "ok": path.exists(),
            "section": section,
            "provider": provider,
            "endpoint": endpoint,
            "status_code": None,
            "message": "本地路径可用" if path.exists() else f"本地路径不存在: {path}",
        }

    if not endpoint:
        raise HTTPException(status_code=400, detail="Endpoint 不能为空")
    parsed = urlparse(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise HTTPException(status_code=400, detail="Endpoint 必须是 http:// 或 https:// 开头的完整地址")

    url = _default_probe_url(provider, endpoint)
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key.strip()}"
    try:
        async with httpx.AsyncClient(timeout=8.0, follow_redirects=False) as client:
            response = await client.get(url, headers=headers)
        ok = response.status_code < 500 and response.status_code not in {401, 403}
        if response.status_code in {401, 403}:
            message = "服务可达，但 API Key/权限未通过"
        elif response.status_code >= 500:
            message = f"服务返回 {response.status_code}"
        else:
            message = "连接成功"
        return {
            "ok": ok,
            "section": section,
            "provider": provider,
            "endpoint": endpoint,
            "probe_url": url,
            "status_code": response.status_code,
            "message": message,
        }
    except Exception as exc:
        return {
            "ok": False,
            "section": section,
            "provider": provider,
            "endpoint": endpoint,
            "probe_url": url,
            "status_code": None,
            "message": f"连接失败: {exc}",
        }


def _asr_settings_status(*, reload_result: dict[str, Any] | None = None) -> dict[str, Any]:
    public_settings = public_model_settings()
    asr_source = public_settings.get("asr", {})
    loaded_device = None
    try:
        from engine.asr import get_asr_manager

        manager = get_asr_manager()
        current_engine = getattr(manager, "_current_engine", None)
        loaded_device = getattr(current_engine, "device", None) if current_engine is not None else None
    except Exception:
        loaded_device = None
    result = {
        **asr_source,
        "model": asr_source.get("model") or config.audio.asr_engine,
        "device": asr_source.get("device") or config.audio.asr_device,
        "env_device": os.environ.get("ASR_DEVICE") or _read_env_value("ASR_DEVICE") or config.audio.asr_device,
        "loaded_device": loaded_device,
        "env_path": str(ENV_PATH),
        "config_path": public_settings["config_path"],
        "device_status": _torch_device_status(),
    }
    if reload_result is not None:
        result["reload_result"] = reload_result
    return result


def _speaker_settings_status(*, switch_result: dict[str, Any] | None = None) -> dict[str, Any]:
    public_settings = public_model_settings()
    speaker_source = public_settings.get("speaker", {})
    try:
        from engine.speaker.speaker_factory import get_engine_manager

        manager = get_engine_manager()
        current = manager.current_type
        current_engine = getattr(manager, "_current_engine", None)
        loaded_device = getattr(current_engine, "device", None) if current_engine is not None else None
    except Exception:
        current = config.speaker.engine_type
        loaded_device = None
    result = {
        **speaker_source,
        "model": speaker_source.get("model") or current,
        "device": speaker_source.get("device") or os.environ.get("SPEAKER_DEVICE") or "auto",
        "env_device": os.environ.get("SPEAKER_DEVICE") or _read_env_value("SPEAKER_DEVICE") or "auto",
        "loaded_device": loaded_device,
        "current": current,
        "config_path": public_settings["config_path"],
        "device_status": _torch_device_status(),
    }
    if switch_result is not None:
        result["switch_result"] = switch_result
    return result


@router.get("/v1/model-config")
async def get_model_config():
    """Return the unified model source config with masked secrets."""
    from engine.asr.factory import ASR_ENGINE_CONFIG
    from engine.speaker.speaker_factory import ENGINE_CONFIG
    from app.services.pyannote_diarization import get_all_diarization_engines

    return {
        **public_model_settings(),
        "supported": {
            "asr": ASR_ENGINE_CONFIG,
            "speaker": ENGINE_CONFIG,
            "diarization": get_all_diarization_engines(),
            "llm_providers": [
                {"key": "ollama", "label": "Ollama", "endpoint": "http://127.0.0.1:11434/v1"},
                {"key": "lmstudio", "label": "LM Studio", "endpoint": "http://127.0.0.1:1234/v1"},
                {"key": "localai", "label": "LocalAI", "endpoint": "http://127.0.0.1:8080/v1"},
                {"key": "vllm", "label": "vLLM", "endpoint": "http://127.0.0.1:8000/v1"},
                {"key": "openai", "label": "OpenAI-compatible", "endpoint": "https://api.openai.com/v1"},
                {"key": "custom", "label": "Custom", "endpoint": ""},
            ],
        },
    }


@router.post("/v1/model-source/test")
async def test_model_source(body: ModelSourceTestRequest):
    settings = read_model_settings(include_env_secrets=True)
    current = settings.get(body.section, {}) if isinstance(settings.get(body.section), dict) else {}
    provider = body.provider or current.get("provider") or "custom"
    endpoint = body.endpoint if body.endpoint is not None else current.get("endpoint") or ""
    api_key = (body.api_key or "").strip() or current.get("api_key") or ""
    return await _probe_model_source(body.section, provider, endpoint, api_key)


@router.get("/v1/asr/engines")
async def list_asr_engines():
    """获取 ASR 引擎列表和切换状态."""
    from engine.asr import get_asr_manager
    return get_asr_manager().get_all_engines_info()


@router.get("/v1/asr/plugins")
async def list_asr_plugins():
    """List external ASR plugin registrations."""
    from engine.asr.plugin_engine import get_all_asr_plugin_infos, plugin_config_path

    return {
        "config_path": str(plugin_config_path()),
        "plugins": get_all_asr_plugin_infos(),
        "engine_type_format": "plugin:<id>",
    }


@router.post("/v1/asr/test")
async def test_asr_engine(
    engine_type: str = Form(..., min_length=1, max_length=128),
    file: UploadFile = File(...),
):
    """Run one uploaded audio sample through a selected ASR engine."""
    from engine.asr import get_asr_engine, get_asr_engine_info
    from engine.asr.contracts import normalize_asr_result

    suffix = Path(file.filename or "sample.wav").suffix or ".wav"
    tmpdir = tempfile.mkdtemp(prefix="asr_test_")
    target = Path(tmpdir) / f"input{suffix}"
    try:
        size = 0
        with target.open("wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > 100 * 1024 * 1024:
                    raise HTTPException(status_code=413, detail="测试音频不能超过 100MB")
                out.write(chunk)

        import librosa

        audio, sample_rate = await asyncio.to_thread(
            librosa.load, str(target), sr=config.audio.sample_rate, mono=True
        )
        if len(audio) < 1600:
            raise HTTPException(status_code=400, detail="测试音频太短")
        engine = await asyncio.to_thread(get_asr_engine, engine_type)
        result = await engine.run_asr(audio, use_preprocessing=True)
        normalized = normalize_asr_result(result, audio_duration=len(audio) / float(sample_rate))
        return {
            "engine_type": get_asr_engine_info(engine_type)["type"],
            "engine_info": get_asr_engine_info(engine_type),
            "sample_rate": int(sample_rate),
            "duration": len(audio) / float(sample_rate),
            "result": normalized,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"ASR 测试失败: {exc}") from exc
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@router.get("/v1/asr/settings")
async def get_asr_settings():
    """Get ASR runtime settings and local acceleration availability."""
    return _asr_settings_status()


@router.put("/v1/asr/settings")
async def update_asr_settings(body: AsrSettingsRequest, request: Request):
    """Update ASR model source/device and optionally reload the active ASR engine."""
    device = _normalize_asr_device(body.device)
    device_status = _torch_device_status()
    _validate_asr_device(device, device_status)

    previous_device = config.audio.asr_device or "auto"
    previous_engine = config.audio.asr_engine
    previous_model_settings = read_raw_model_settings()
    current_model = (body.model or previous_engine).strip()
    values: dict[str, Any] = {
        "device": device,
        "model": current_model,
    }
    if body.provider is not None:
        values["provider"] = body.provider
    if body.endpoint is not None:
        values["endpoint"] = body.endpoint
    if body.api_key is not None:
        values["api_key"] = body.api_key
    if body.word_timestamps is not None:
        values["word_timestamps"] = body.word_timestamps
    if body.load_timeout_sec is not None:
        values["load_timeout_sec"] = body.load_timeout_sec
    env_values = {
        "ASR_ENGINE": current_model,
        "ASR_DEVICE": device,
        **(
            {"ASR_WORD_TIMESTAMPS": str(body.word_timestamps).lower()}
            if body.word_timestamps is not None
            else {}
        ),
        **(
            {"ASR_LOAD_TIMEOUT_SEC": str(body.load_timeout_sec)}
            if body.load_timeout_sec is not None
            else {}
        ),
    }

    try:
        update_model_section("asr", values)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Unable to update .env/model settings: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    apply_model_settings_to_runtime()

    reload_result: dict[str, Any] | None = None
    if body.reload_current:
        from engine.asr import get_asr_manager

        manager = get_asr_manager()
        if current_model and current_model != manager.current_type:
            reload_result = await asyncio.to_thread(manager.switch_engine, current_model)
        else:
            reload_result = await asyncio.to_thread(manager.reload_current)
        if not reload_result.get("success"):
            os.environ["ASR_DEVICE"] = previous_device
            os.environ["ASR_ENGINE"] = previous_engine
            config.audio.asr_device = previous_device
            config.audio.asr_engine = previous_engine
            write_model_settings(previous_model_settings)
            raise HTTPException(
                status_code=400,
                detail=(
                    f"ASR 配置加载失败，已回滚运行时到 {previous_engine}/{previous_device}: "
                    f"{reload_result.get('error', 'ASR 重载失败')}"
                ),
            )

        new_engine = manager.get_engine()
        runtime = getattr(request.app.state, "runtime", None)
        if runtime is not None:
            runtime.set_asr(new_engine)

    try:
        _write_env_values(env_values)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Unable to update .env: {exc}") from exc

    return _asr_settings_status(reload_result=reload_result)


@router.get("/v1/speaker/settings")
async def get_speaker_settings():
    """Get speaker embedding model source settings."""
    return _speaker_settings_status()


@router.put("/v1/speaker/settings")
async def update_speaker_settings(body: SpeakerSettingsRequest, request: Request):
    """Update speaker embedding source/model and switch runtime engine."""
    model = body.model.strip()
    previous_model_settings = read_raw_model_settings()
    try:
        update_model_section(
            "speaker",
            {
                "provider": body.provider,
                "endpoint": body.endpoint,
                "api_key": body.api_key,
                "model": model,
                "device": body.device,
            },
        )
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Unable to update .env/model settings: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    apply_model_settings_to_runtime()

    from engine.speaker.speaker_factory import get_engine_manager

    manager = get_engine_manager()
    result = await asyncio.to_thread(manager.switch_engine, model, force_reload=True)
    if not result.get("success"):
        write_model_settings(previous_model_settings)
        apply_model_settings_to_runtime()
        raise HTTPException(status_code=400, detail=result.get("error", "声纹引擎切换失败"))

    new_engine = manager.get_engine()
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is not None:
        runtime.set_speaker(new_engine)
    if hasattr(request.app.state, "spk_engine"):
        request.app.state.spk_engine = new_engine

    try:
        _write_env_values(
            {
                "SPEAKER_ENGINE": result.get("engine_type") or model,
                "SPEAKER_DEVICE": body.device,
                "SPEAKER_PROVIDER": body.provider,
                "SPEAKER_ENDPOINT": body.endpoint or "",
            }
        )
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Unable to update .env: {exc}") from exc

    return _speaker_settings_status(switch_result=result)


@router.get("/v1/diarization/settings")
async def get_diarization_settings():
    return _diarization_status()


@router.get("/v1/diarization/engines")
async def list_diarization_engines():
    from app.services.pyannote_diarization import get_all_diarization_engines

    return {"engines": get_all_diarization_engines()}


@router.put("/v1/diarization/settings")
async def update_diarization_settings(body: DiarizationSettingsRequest):
    from app.services.pyannote_diarization import (
        default_diarization_model,
        get_diarization_engine_info,
        normalize_diarization_engine,
        reset_diarization_engine,
    )

    current = read_model_settings(include_env_secrets=True).get("diarization", {})
    engine = normalize_diarization_engine(body.engine or current.get("engine"))
    engine_info = get_diarization_engine_info(engine)
    previous_engine = normalize_diarization_engine(current.get("engine"))
    requested_model = (body.model_id or "").strip()
    current_model = str(current.get("model") or "").strip()
    if body.engine and engine != previous_engine and (not requested_model or requested_model == current_model):
        model_id = str(engine_info.get("model") or default_diarization_model(engine))
    elif requested_model:
        model_id = requested_model
    else:
        model_id = str(
            current.get("model")
            or engine_info.get("model")
            or default_diarization_model(engine)
        )
    model_id = model_id.strip() or default_diarization_model(engine)
    token = (body.api_key or body.hf_token or "").strip()
    provider = (body.provider or current.get("provider") or engine_info.get("provider") or "huggingface").strip()
    endpoint = (body.endpoint or current.get("endpoint") or engine_info.get("endpoint") or "").strip()
    command = (body.command if body.command is not None else current.get("command") or "").strip()
    try:
        update_model_section(
            "diarization",
            {
                "engine": engine,
                "provider": provider,
                "endpoint": endpoint,
                "api_key": token,
                "model": model_id,
                "command": command,
                "device": body.device,
            },
        )
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    values = {
        "DIARIZATION_ENGINE": engine,
        "DIARIZATION_MODEL": model_id,
        "DIARIZATION_PROVIDER": provider,
        "DIARIZATION_ENDPOINT": endpoint,
        "DIARIZATION_COMMAND": command,
        "PYANNOTE_DEVICE": body.device,
        "PYANNOTE_MODEL": model_id,
    }
    if token:
        os.environ["DIARIZATION_API_KEY"] = token
        if provider == "huggingface":
            os.environ["HF_TOKEN"] = token
        elif provider == "modelscope":
            os.environ["MODELSCOPE_API_TOKEN"] = token
    os.environ["DIARIZATION_ENGINE"] = engine
    os.environ["DIARIZATION_MODEL"] = model_id
    os.environ["DIARIZATION_PROVIDER"] = provider
    os.environ["DIARIZATION_ENDPOINT"] = endpoint
    if command:
        os.environ["DIARIZATION_COMMAND"] = command
    else:
        os.environ.pop("DIARIZATION_COMMAND", None)
    os.environ["PYANNOTE_DEVICE"] = body.device
    os.environ["PYANNOTE_MODEL"] = model_id
    config.speaker.diarization_device = body.device
    apply_model_settings_to_runtime()
    try:
        _write_env_values(values)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Unable to update .env: {exc}") from exc

    reset_diarization_engine()
    return _diarization_status()


@router.post("/v1/diarization/test")
async def test_diarization_settings():
    from app.services.pyannote_diarization import (
        get_diarization_engine,
        reset_diarization_engine,
    )

    reset_diarization_engine()
    await asyncio.to_thread(get_diarization_engine)
    return _diarization_status(load=True)


@router.put("/v1/asr/engine", response_model=AsrSwitchResponse)
async def switch_asr_engine(body: AsrSwitchRequest, request: Request):
    """受控切换 ASR 引擎.

    下载/加载新模型期间旧 ASR 继续服务;只有新模型加载成功后才切换。
    """
    from engine.asr import get_asr_manager

    manager = get_asr_manager()
    result = await asyncio.to_thread(manager.switch_engine, body.engine_type)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "ASR 切换失败"))

    try:
        update_model_section("asr", {"model": result.get("engine_type") or body.engine_type})
        _write_env_values({"ASR_ENGINE": result.get("engine_type") or body.engine_type})
        apply_model_settings_to_runtime()
    except Exception:
        # Runtime switch already succeeded; persistence failure should not mask it.
        pass

    new_engine = manager.get_engine()
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is not None:
        runtime.set_asr(new_engine)
    # ASR 引擎统一由 runtime 管理。

    return AsrSwitchResponse(**result)
