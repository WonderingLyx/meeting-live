"""设置/存储相关端点"""
import asyncio
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from app.config import config

router = APIRouter()


class AsrSwitchRequest(BaseModel):
    """ASR 切换请求."""
    engine_type: str = Field(..., min_length=1, max_length=128, description="ASR 引擎类型")


class AsrSettingsRequest(BaseModel):
    """ASR runtime settings."""
    device: str = Field("auto", min_length=2, max_length=16, description="auto | cpu | cuda")
    reload_current: bool = True


class DiarizationSettingsRequest(BaseModel):
    hf_token: Optional[str] = Field(None, max_length=500)
    device: str = Field("auto", pattern="^(auto|cpu|cuda)$")
    model_id: Optional[str] = Field(None, min_length=1, max_length=200)


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
    env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    file_token = _read_env_value("HF_TOKEN")
    token = env_token or file_token
    model_id = (
        os.environ.get("PYANNOTE_MODEL")
        or _read_env_value("PYANNOTE_MODEL")
        or PYANNOTE_DEFAULT_MODEL
    ).strip() or PYANNOTE_DEFAULT_MODEL
    model_revision = (
        os.environ.get("PYANNOTE_MODEL_REVISION")
        or _read_env_value("PYANNOTE_MODEL_REVISION")
        or None
    )
    try:
        from app.services.pyannote_diarization import PyannoteDiarizer, get_pyannote_diarizer

        if load:
            diarizer = get_pyannote_diarizer()
            enabled = bool(diarizer.enabled)
            last_error = diarizer.last_error
            loaded_device = diarizer.device
        else:
            enabled = bool(PyannoteDiarizer._enabled)
            last_error = PyannoteDiarizer._last_error
            loaded_device = PyannoteDiarizer._device
            if not token and not last_error:
                last_error = "HF_TOKEN 未设置"
    except Exception as exc:
        enabled = False
        last_error = str(exc)
        loaded_device = "cpu"
    return {
        "enabled": enabled,
        "token_configured": bool(token),
        "token_preview": _mask_secret(token),
        "token_source": "environment" if env_token else ("env_file" if file_token else None),
        "device": config.speaker.diarization_device,
        "loaded_device": loaded_device,
        "model_id": model_id,
        "model_revision": model_revision,
        "last_error": last_error,
        "env_path": str(ENV_PATH),
        "terms_url": PYANNOTE_TERMS_URL,
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


def _asr_settings_status(*, reload_result: dict[str, Any] | None = None) -> dict[str, Any]:
    loaded_device = None
    try:
        from engine.asr import get_asr_manager

        manager = get_asr_manager()
        current_engine = getattr(manager, "_current_engine", None)
        loaded_device = getattr(current_engine, "device", None) if current_engine is not None else None
    except Exception:
        loaded_device = None
    result = {
        "device": config.audio.asr_device,
        "env_device": os.environ.get("ASR_DEVICE") or _read_env_value("ASR_DEVICE") or config.audio.asr_device,
        "loaded_device": loaded_device,
        "env_path": str(ENV_PATH),
        "device_status": _torch_device_status(),
    }
    if reload_result is not None:
        result["reload_result"] = reload_result
    return result


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
    """Update ASR device and optionally reload the active ASR engine."""
    device = _normalize_asr_device(body.device)
    device_status = _torch_device_status()
    _validate_asr_device(device, device_status)

    previous_device = config.audio.asr_device or "auto"
    try:
        _write_env_values({"ASR_DEVICE": device})
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Unable to update .env: {exc}") from exc

    os.environ["ASR_DEVICE"] = device
    config.audio.asr_device = device

    reload_result: dict[str, Any] | None = None
    if body.reload_current:
        from engine.asr import get_asr_manager

        manager = get_asr_manager()
        reload_result = await asyncio.to_thread(manager.reload_current)
        if not reload_result.get("success"):
            os.environ["ASR_DEVICE"] = previous_device
            config.audio.asr_device = previous_device
            try:
                _write_env_values({"ASR_DEVICE": previous_device})
            except OSError:
                pass
            raise HTTPException(
                status_code=400,
                detail=(
                    f"ASR_DEVICE={device} 加载失败，已回滚到 {previous_device}: "
                    f"{reload_result.get('error', 'ASR 重载失败')}"
                ),
            )

        new_engine = manager.get_engine()
        runtime = getattr(request.app.state, "runtime", None)
        if runtime is not None:
            runtime.set_asr(new_engine)

    return _asr_settings_status(reload_result=reload_result)


@router.get("/v1/diarization/settings")
async def get_diarization_settings():
    return _diarization_status()


@router.put("/v1/diarization/settings")
async def update_diarization_settings(body: DiarizationSettingsRequest):
    model_id = (body.model_id or PYANNOTE_DEFAULT_MODEL).strip() or PYANNOTE_DEFAULT_MODEL
    values = {
        "PYANNOTE_DEVICE": body.device,
        "PYANNOTE_MODEL": model_id,
    }
    token = (body.hf_token or "").strip()
    if token:
        values["HF_TOKEN"] = token
        os.environ["HF_TOKEN"] = token
    os.environ["PYANNOTE_DEVICE"] = body.device
    os.environ["PYANNOTE_MODEL"] = model_id
    config.speaker.diarization_device = body.device
    try:
        _write_env_values(values)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Unable to update .env: {exc}") from exc

    from app.services.pyannote_diarization import reset_pyannote_diarizer

    reset_pyannote_diarizer()
    return _diarization_status()


@router.post("/v1/diarization/test")
async def test_diarization_settings():
    from app.services.pyannote_diarization import (
        get_pyannote_diarizer,
        reset_pyannote_diarizer,
    )

    reset_pyannote_diarizer()
    await asyncio.to_thread(get_pyannote_diarizer)
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

    new_engine = manager.get_engine()
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is not None:
        runtime.set_asr(new_engine)
    # ASR 引擎统一由 runtime 管理。

    return AsrSwitchResponse(**result)
