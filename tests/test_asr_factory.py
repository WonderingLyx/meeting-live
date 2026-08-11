"""ASR 引擎工厂测试."""
import sys
import types
import json
import asyncio
from unittest.mock import MagicMock

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def reset_asr_manager_state():
    from engine.asr.factory import ASREngineManager
    ASREngineManager.reset()
    yield
    ASREngineManager.reset()


def test_asr_config_engine_default(monkeypatch):
    from app.config import AudioConfig
    monkeypatch.delenv("ASR_ENGINE", raising=False)
    cfg = AudioConfig.from_env()
    assert cfg.asr_engine == "sensevoice_zh"


def test_asr_config_engine_from_env(monkeypatch):
    monkeypatch.setenv("ASR_ENGINE", "SenseVoice")
    from app.config import AudioConfig
    cfg = AudioConfig.from_env()
    assert cfg.asr_engine == "sensevoice"


def test_asr_factory_qwen_uses_qwen_module(monkeypatch):
    fake_asr = types.ModuleType("engine.asr_engine")
    expected = MagicMock()
    fake_asr.ASREngine = MagicMock(return_value=expected)
    monkeypatch.setitem(sys.modules, "engine.asr_engine", fake_asr)

    from engine.asr.factory import get_asr_engine
    assert get_asr_engine("qwen3") is expected
    fake_asr.ASREngine.assert_called_once()


def test_asr_factory_aliases():
    from engine.asr.factory import get_asr_engine_info
    assert get_asr_engine_info("qwen")["type"] == "qwen3"
    assert get_asr_engine_info("sensevoice-small")["type"] == "sensevoice"
    assert get_asr_engine_info("sensevoice-cn")["type"] == "sensevoice_zh"
    assert get_asr_engine_info("paraformer-punc")["type"] == "paraformer_full"
    assert get_asr_engine_info("paraformer-large")["type"] == "paraformer_large"
    assert get_asr_engine_info("paraformer-speaker")["type"] == "paraformer_spk"
    assert get_asr_engine_info("funasr-streaming")["type"] == "paraformer_streaming"


def test_get_all_asr_engines_shape(monkeypatch):
    from app.config import config
    from engine.asr.factory import ASREngineManager
    monkeypatch.setattr(config.audio, "asr_engine", "sensevoice_zh")
    ASREngineManager.reset()

    from engine.asr import get_all_asr_engines
    data = get_all_asr_engines()
    assert data["current"] == "sensevoice_zh"
    assert "qwen3" in data["engines"]
    assert "sensevoice" in data["engines"]
    assert "sensevoice_zh" in data["engines"]
    assert "paraformer" in data["engines"]
    assert "paraformer_full" in data["engines"]
    assert "paraformer_large" in data["engines"]
    assert "paraformer_spk" in data["engines"]
    assert "paraformer_streaming" in data["engines"]
    assert "sherpa_onnx" not in data["engines"]
    assert data["switching"] is False
    assert data["pending"] is None
    assert data["cached"] == []


def test_external_asr_plugin_appears_in_catalog(tmp_path, monkeypatch):
    config_path = tmp_path / "asr_plugins.json"
    config_path.write_text(json.dumps({
        "plugins": {
            "dummy-plugin": {
                "name": "Dummy Plugin",
                "model": "dummy-local",
                "command": [sys.executable, "-c", "print('hello')"],
            }
        }
    }), encoding="utf-8")
    monkeypatch.setenv("ASR_PLUGINS_FILE", str(config_path))

    from engine.asr.factory import ASREngineManager
    ASREngineManager.reset()
    from engine.asr import get_all_asr_engines

    data = get_all_asr_engines()
    assert "plugin:dummy_plugin" in data["engines"]
    assert data["engines"]["plugin:dummy_plugin"]["plugin"] is True
    assert data["engines"]["plugin:dummy_plugin"]["available"] is True


def test_external_asr_plugin_disabled_and_missing_dependencies(tmp_path, monkeypatch):
    config_path = tmp_path / "asr_plugins.json"
    config_path.write_text(json.dumps({
        "plugins": {
            "disabled": {
                "enabled": False,
                "name": "Disabled Plugin",
                "command": [sys.executable, "-c", "print('disabled')"],
            },
            "missing": {
                "name": "Missing Dependency Plugin",
                "command": [sys.executable, "-c", "print('missing')"],
                "required_python_modules": ["module_that_does_not_exist_123"],
            },
        }
    }), encoding="utf-8")
    monkeypatch.setenv("ASR_PLUGINS_FILE", str(config_path))

    from engine.asr.factory import ASREngineManager
    ASREngineManager.reset()
    from engine.asr import get_all_asr_engines

    data = get_all_asr_engines()
    assert "plugin:disabled" not in data["engines"]
    assert data["engines"]["plugin:missing"]["available"] is False
    assert "module_that_does_not_exist_123" in data["engines"]["plugin:missing"]["reason"]


def test_external_asr_plugin_command_contract(tmp_path, monkeypatch):
    config_path = tmp_path / "asr_plugins.json"
    config_path.write_text(json.dumps({
        "plugins": {
            "dummy": {
                "name": "Dummy Plugin",
                "model": "dummy-local",
                "sample_rate": 16000,
                "command": [
                    sys.executable,
                    "-c",
                    "import json; print(json.dumps({'text':'插件测试','language':'zh'}))",
                ],
            }
        }
    }), encoding="utf-8")
    monkeypatch.setenv("ASR_PLUGINS_FILE", str(config_path))

    from engine.asr.factory import ASREngineManager, get_asr_engine

    ASREngineManager.reset()
    engine = get_asr_engine("plugin:dummy")
    audio = np.ones(32000, dtype=np.float32) * 0.05
    result = asyncio.run(engine.run_asr(audio))
    assert result["text"] == "插件测试"
    assert result["language"] == "zh"
    assert result["provider_metadata"]["plugin_id"] == "dummy"


def test_asr_engine_capabilities_are_explicit():
    from engine.asr.factory import get_asr_engine_info

    qwen = get_asr_engine_info("qwen3")
    streaming = get_asr_engine_info("paraformer_streaming")
    spk = get_asr_engine_info("paraformer_spk")

    assert qwen["capabilities"]["word_timestamps"] is True
    assert qwen["capabilities"]["speaker_diarization"] is False
    assert streaming["capabilities"]["upload"] is True
    assert streaming["capabilities"]["true_streaming"] == "adapter_not_yet"
    assert streaming["capabilities"]["word_timestamps"] is False
    assert qwen["capabilities"]["result_contract"] == "asr-result-v1"
    assert qwen["capabilities"]["timestamp_granularity"] == "word_optional"
    assert streaming["capabilities"]["native_metadata"] is True
    assert streaming["capabilities"]["timestamp_granularity"] == "dynamic"
    assert spk["capabilities"]["speaker_diarization"] == "provider_chunk_labels"


def test_funasr_builtin_variants_load_expected_automodel(monkeypatch):
    calls = []

    class FakeAutoModel:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    fake_fun = types.ModuleType("funasr")
    fake_fun.AutoModel = FakeAutoModel
    monkeypatch.setitem(sys.modules, "funasr", fake_fun)
    fake_utils = types.ModuleType("funasr.utils.postprocess_utils")
    fake_utils.rich_transcription_postprocess = lambda text: text
    monkeypatch.setitem(sys.modules, "funasr.utils.postprocess_utils", fake_utils)
    monkeypatch.setattr(
        "engine.asr.funasr_engine.FunASREngine._resolve_device",
        staticmethod(lambda _requested: "cpu"),
    )

    from engine.asr.funasr_engine import FunASREngine

    FunASREngine._instances.clear()
    FunASREngine("sensevoice_zh")
    FunASREngine("paraformer_full")
    FunASREngine("paraformer_large")
    FunASREngine("paraformer_spk")

    assert calls[0]["model"] == "iic/SenseVoiceSmall"
    assert calls[1]["model"] == "paraformer-zh"
    assert calls[1]["punc_model"] == "ct-punc"
    assert calls[2]["model"] == "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
    assert calls[2]["punc_model"] == "ct-punc"
    assert calls[3]["spk_model"] == "cam++"


def test_asr_engine_capabilities_can_be_overridden(monkeypatch):
    monkeypatch.setenv(
        "ASR_CAPABILITIES_JSON",
        '{"qwen3":{"description":"Custom deployment","capabilities":{"word_timestamps":false,"notes":"本部署关闭字级时间戳"}}}',
    )
    from engine.asr.factory import get_asr_engine_info

    info = get_asr_engine_info("qwen3")
    assert info["customized"] is True
    assert info["description"] == "Custom deployment"
    assert info["capabilities"]["word_timestamps"] is False
    assert info["capabilities"]["speaker_diarization"] is False
    assert info["capabilities"]["notes"] == "本部署关闭字级时间戳"


def test_asr_manager_switch_success_after_load(monkeypatch):
    from engine.asr import factory

    factory.ASREngineManager.reset()
    monkeypatch.setenv("ASR_ENGINE", "qwen3")
    engines = {"qwen3": object(), "sensevoice": object()}

    def fake_load(engine_type=None):
        return engines[engine_type or "qwen3"]

    monkeypatch.setattr(factory, "get_asr_engine", fake_load)
    monkeypatch.setattr(factory.importlib.util, "find_spec", lambda _name: object())

    manager = factory.get_asr_manager()
    assert manager.get_engine() is engines["qwen3"]

    result = manager.switch_engine("sensevoice")
    assert result["success"] is True
    assert result["previous_type"] == "qwen3"
    assert result["engine_type"] == "sensevoice"
    assert manager.current_type == "sensevoice"
    assert manager.get_engine() is engines["sensevoice"]


def test_asr_manager_switch_failure_keeps_current(monkeypatch):
    from engine.asr import factory

    factory.ASREngineManager.reset()
    monkeypatch.setenv("ASR_ENGINE", "qwen3")
    current = object()

    def fake_load(engine_type=None):
        if engine_type == "sensevoice":
            raise RuntimeError("download failed")
        return current

    monkeypatch.setattr(factory, "get_asr_engine", fake_load)
    monkeypatch.setattr(factory.importlib.util, "find_spec", lambda _name: object())

    manager = factory.get_asr_manager()
    assert manager.get_engine() is current

    result = manager.switch_engine("sensevoice")
    assert result["success"] is False
    assert "download failed" in result["error"]
    assert manager.current_type == "qwen3"
    assert manager.get_engine() is current


def test_asr_dependency_status_marks_missing_funasr(monkeypatch):
    from engine.asr import factory

    factory.ASREngineManager.reset()

    def fake_find_spec(name):
        if name == "funasr":
            return None
        return object()

    monkeypatch.setattr(factory.importlib.util, "find_spec", fake_find_spec)

    info = factory.get_asr_engine_info("sensevoice")
    assert info["available"] is False
    assert info["dependency"] == "funasr"
    assert "funasr" in info["install_hint"]


def test_asr_manager_switch_missing_dependency_does_not_load(monkeypatch):
    from engine.asr import factory

    factory.ASREngineManager.reset()
    current = object()

    def fake_find_spec(name):
        if name == "funasr":
            return None
        return object()

    def fake_load(engine_type=None):
        return current

    monkeypatch.setattr(factory.importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setattr(factory, "get_asr_engine", fake_load)

    manager = factory.get_asr_manager()
    assert manager.get_engine() is current

    result = manager.switch_engine("sensevoice")
    assert result["success"] is False
    assert result["dependency"] == "funasr"
    assert manager.current_type == "qwen3"
    assert manager.get_engine() is current
