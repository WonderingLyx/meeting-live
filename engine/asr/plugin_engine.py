"""External ASR plugin adapter.

Plugins are configured by JSON and executed as local commands.  This gives the
project a stable adapter boundary for testing ASR models that do not deserve a
first-class in-process integration yet.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from app.config import config
from .common import evaluate_audio_quality, filter_hallucinations, rms_is_silent
from .contracts import empty_asr_result, normalize_asr_result

logger = logging.getLogger("ASR_Engine")

PLUGIN_PREFIX = "plugin:"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLUGINS_FILE = PROJECT_ROOT / "config" / "asr_plugins.json"
PLUGIN_ID_RE = re.compile(r"^[a-z0-9_][a-z0-9_.]*$")


def normalize_plugin_id(value: str | None) -> str:
    normalized = (value or "").strip().lower().replace("-", "_")
    normalized = re.sub(r"[^a-z0-9_.]", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_.")
    if not normalized:
        raise ValueError("ASR plugin id cannot be empty")
    if not PLUGIN_ID_RE.match(normalized):
        raise ValueError(
            "ASR plugin id must contain lowercase letters, numbers, underscore, or dot"
        )
    return normalized


def plugin_engine_type(plugin_id: str) -> str:
    return f"{PLUGIN_PREFIX}{normalize_plugin_id(plugin_id)}"


def is_plugin_engine_type(engine_type: str | None) -> bool:
    return bool(engine_type and str(engine_type).startswith(PLUGIN_PREFIX))


def plugin_id_from_engine_type(engine_type: str) -> str:
    if not is_plugin_engine_type(engine_type):
        raise ValueError(f"Not an ASR plugin engine type: {engine_type}")
    return normalize_plugin_id(str(engine_type).split(":", 1)[1])


def plugin_config_path() -> Path:
    raw = (os.environ.get("ASR_PLUGINS_FILE") or "").strip()
    return Path(raw).expanduser() if raw else DEFAULT_PLUGINS_FILE


def _read_plugin_config() -> dict[str, Any]:
    path = plugin_config_path()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("[ASR Plugin] failed to read %s: %s", path, exc)
        return {}
    if isinstance(data, dict) and isinstance(data.get("plugins"), dict):
        data = data["plugins"]
    if not isinstance(data, dict):
        logger.warning("[ASR Plugin] %s must be an object or {'plugins': object}", path)
        return {}
    return data


def _normalize_command(command: object) -> list[str]:
    if isinstance(command, list):
        return [str(item) for item in command if str(item)]
    if isinstance(command, str) and command.strip():
        return shlex.split(command, posix=os.name != "nt")
    return []


def _plugin_working_dir(spec: dict[str, Any]) -> Path:
    raw = str(spec.get("working_dir") or "").strip()
    if not raw:
        return PROJECT_ROOT
    path = Path(os.path.expandvars(raw)).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _plugin_info(plugin_id: str, raw: dict[str, Any]) -> dict[str, Any]:
    engine_type = plugin_engine_type(plugin_id)
    name = str(raw.get("name") or plugin_id)
    capabilities = {
        "transcription": True,
        "upload": True,
        "realtime_segmented": bool(raw.get("realtime_segmented", True)),
        "true_streaming": bool(raw.get("true_streaming", False)),
        "word_timestamps": bool(raw.get("supports_words", False)),
        "speaker_diarization": bool(raw.get("speaker_diarization", False)),
        "result_contract": "asr-result-v1",
        "timestamp_granularity": str(raw.get("timestamp_granularity") or "dynamic"),
        "native_metadata": True,
        "recommended_for": raw.get("recommended_for") or ["custom_local_asr"],
        "notes": str(raw.get("notes") or "外部命令 ASR 插件; stdout 必须输出 ASR JSON 或纯文本。"),
        "notes_en": str(raw.get("notes_en") or "External command ASR plugin; stdout must be ASR JSON or plain text."),
    }
    return {
        "type": engine_type,
        "plugin_id": plugin_id,
        "plugin": True,
        "adapter": str(raw.get("adapter") or "command"),
        "name": name,
        "model": str(raw.get("model") or raw.get("model_id") or name),
        "description": str(raw.get("description") or "自定义本地 ASR 插件"),
        "description_en": str(raw.get("description_en") or "Custom local ASR plugin"),
        "languages": str(raw.get("languages") or "自定义"),
        "languages_en": str(raw.get("languages_en") or raw.get("languages") or "Custom"),
        "supports_streaming": bool(raw.get("supports_streaming", False)),
        "supports_words": bool(raw.get("supports_words", False)),
        "capabilities": capabilities,
        "customized": True,
        "config_path": str(plugin_config_path()),
    }


def get_all_asr_plugin_specs() -> dict[str, dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {}
    for raw_id, raw_spec in _read_plugin_config().items():
        if not isinstance(raw_spec, dict):
            logger.warning("[ASR Plugin] plugin %s must be an object", raw_id)
            continue
        if raw_spec.get("enabled") is False:
            continue
        try:
            plugin_id = normalize_plugin_id(str(raw_spec.get("id") or raw_id))
        except ValueError as exc:
            logger.warning("[ASR Plugin] invalid plugin id %s: %s", raw_id, exc)
            continue
        spec = dict(raw_spec)
        spec["id"] = plugin_id
        specs[plugin_id] = spec
    return specs


def get_asr_plugin_spec(engine_type: str) -> dict[str, Any] | None:
    plugin_id = plugin_id_from_engine_type(engine_type)
    return get_all_asr_plugin_specs().get(plugin_id)


def get_all_asr_plugin_infos() -> dict[str, dict[str, Any]]:
    infos: dict[str, dict[str, Any]] = {}
    for plugin_id, spec in get_all_asr_plugin_specs().items():
        info = _plugin_info(plugin_id, spec)
        info.update(plugin_dependency_status(plugin_engine_type(plugin_id)))
        infos[plugin_engine_type(plugin_id)] = info
    return infos


def get_asr_plugin_info(engine_type: str) -> dict[str, Any] | None:
    spec = get_asr_plugin_spec(engine_type)
    if not spec:
        return None
    plugin_id = plugin_id_from_engine_type(engine_type)
    info = _plugin_info(plugin_id, spec)
    info.update(plugin_dependency_status(engine_type))
    return info


def plugin_dependency_status(engine_type: str) -> dict[str, Any]:
    spec = get_asr_plugin_spec(engine_type)
    if not spec:
        return {
            "available": False,
            "reason": "plugin not found",
            "install_hint": f"请在 {plugin_config_path()} 配置该 ASR 插件",
            "install_hint_en": f"Configure this ASR plugin in {plugin_config_path()}",
        }
    if str(spec.get("adapter") or "command") != "command":
        return {
            "available": False,
            "reason": f"unsupported adapter: {spec.get('adapter')}",
            "install_hint": "当前只支持 command 适配器",
            "install_hint_en": "Only the command adapter is currently supported.",
        }
    command = _normalize_command(spec.get("command"))
    if not command:
        return {
            "available": False,
            "reason": "command is missing",
            "install_hint": "请为插件配置 command 数组",
            "install_hint_en": "Configure a command array for the plugin.",
        }
    executable = command[0]
    if "{" not in executable and "}" not in executable:
        expanded = os.path.expandvars(os.path.expanduser(executable))
        found = shutil.which(expanded) or (expanded if os.path.exists(expanded) else None)
        if not found:
            return {
                "available": False,
                "reason": f"command not found: {executable}",
                "install_hint": f"命令不可用: {executable}",
                "install_hint_en": f"Command is not available: {executable}",
            }

    modules = spec.get("required_python_modules") or []
    if isinstance(modules, str):
        modules = [modules]
    if isinstance(modules, list):
        for module in modules:
            module_name = str(module).strip()
            if not module_name:
                continue
            if importlib.util.find_spec(module_name) is None:
                return {
                    "available": False,
                    "reason": f"missing python module: {module_name}",
                    "install_hint": f"缺少 Python 依赖: {module_name}",
                    "install_hint_en": f"Missing Python module: {module_name}",
                }

    required_files = spec.get("required_files") or []
    if isinstance(required_files, str):
        required_files = [required_files]
    if isinstance(required_files, list):
        base_dir = _plugin_working_dir(spec)
        for item in required_files:
            raw_path = str(item).strip()
            if not raw_path:
                continue
            candidate = Path(os.path.expandvars(raw_path)).expanduser()
            if not candidate.is_absolute():
                candidate = base_dir / candidate
            if not candidate.exists():
                return {
                    "available": False,
                    "reason": f"required file not found: {raw_path}",
                    "install_hint": f"插件依赖文件不存在: {raw_path}",
                    "install_hint_en": f"Required plugin file was not found: {raw_path}",
                }
    return {"available": True, "dependency": "external-command"}


def _resolve_device() -> str:
    requested = (config.audio.asr_device or "auto").lower()
    if requested == "auto":
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"
    if requested == "cuda":
        return "cuda"
    return requested


def _render_command(command: list[str], *, audio_path: str, device: str, sample_rate: int, language: str) -> list[str]:
    values = {
        "audio": audio_path,
        "device": device,
        "sample_rate": str(sample_rate),
        "language": language,
        "python": sys.executable,
    }
    rendered = []
    for item in command:
        value = str(item)
        for key, replacement in values.items():
            value = value.replace("{" + key + "}", replacement)
        rendered.append(os.path.expandvars(value))
    return rendered


def _decode_stdout(stdout: str) -> object:
    text = (stdout or "").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return {"text": text}


class ExternalCommandASREngine:
    """ASR engine backed by a local command that emits ASR JSON."""

    def __init__(self, engine_type: str):
        self.engine_type = plugin_engine_type(plugin_id_from_engine_type(engine_type))
        self.plugin_id = plugin_id_from_engine_type(self.engine_type)
        self.spec = get_asr_plugin_spec(self.engine_type)
        if not self.spec:
            raise RuntimeError(f"ASR plugin not found: {self.engine_type}")
        status = plugin_dependency_status(self.engine_type)
        if not status.get("available", False):
            raise RuntimeError(status.get("install_hint") or status.get("reason") or "ASR plugin unavailable")
        self.kind = self.engine_type
        self.model_id = str(self.spec.get("model") or self.spec.get("model_id") or self.plugin_id)
        self.sample_rate = int(self.spec.get("sample_rate") or config.audio.sample_rate)
        self.language = str(self.spec.get("language") or self.spec.get("default_language") or "zh")
        self.timeout_sec = int(self.spec.get("timeout_sec") or 300)
        self.device = _resolve_device()
        self.initialized = True

    def is_silent(self, audio_data, threshold=0.012, use_vad=True):
        return rms_is_silent(audio_data, threshold=threshold)

    def evaluate_audio_quality(self, audio_data: np.ndarray) -> dict:
        return evaluate_audio_quality(audio_data)

    async def run_asr(self, audio_data, use_preprocessing=True):
        if audio_data is None or len(audio_data) < 1600:
            return empty_asr_result()
        if self.is_silent(audio_data, use_vad=False):
            return empty_asr_result()
        return await asyncio.to_thread(self._run_sync, audio_data)

    def _run_sync(self, audio_data) -> dict[str, Any]:
        import soundfile as sf

        audio = np.asarray(audio_data, dtype=np.float32)
        command = _normalize_command(self.spec.get("command"))
        env = os.environ.copy()
        plugin_env = self.spec.get("env") or {}
        if isinstance(plugin_env, dict):
            env.update({str(k): str(v) for k, v in plugin_env.items()})

        with tempfile.TemporaryDirectory(prefix=f"asr_plugin_{self.plugin_id}_") as tmpdir:
            audio_path = os.path.join(tmpdir, "input.wav")
            sf.write(audio_path, audio, self.sample_rate)
            args = _render_command(
                command,
                audio_path=audio_path,
                device=self.device,
                sample_rate=self.sample_rate,
                language=self.language,
            )
            logger.info("[ASR Plugin] run %s: %s", self.engine_type, " ".join(args[:3]))
            started = time.time()
            proc = subprocess.run(
                args,
                cwd=str(_plugin_working_dir(self.spec)),
                env=env,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                timeout=self.timeout_sec,
                check=False,
            )
            elapsed = time.time() - started
            if proc.returncode != 0:
                stderr = (proc.stderr or "").strip()
                raise RuntimeError(
                    f"ASR plugin {self.engine_type} exited with {proc.returncode}: {stderr[-1200:]}"
                )

        decoded = _decode_stdout(proc.stdout)
        result = normalize_asr_result(decoded, audio_duration=len(audio) / float(self.sample_rate))
        text = filter_hallucinations(result["text"])
        result["text"] = text
        metadata = dict(result.get("provider_metadata") or {})
        metadata.update({
            "plugin_id": self.plugin_id,
            "engine_type": self.engine_type,
            "model": self.model_id,
            "elapsed_sec": round(elapsed, 3),
        })
        result["provider_metadata"] = metadata
        return result
