import json
import os


def test_public_model_settings_masks_api_keys(tmp_path, monkeypatch):
    config_path = tmp_path / "model-settings.json"
    monkeypatch.setenv("MODEL_SETTINGS_FILE", str(config_path))

    from app.services.model_config import public_model_settings, update_model_section

    update_model_section(
        "asr",
        {
            "provider": "modelscope",
            "endpoint": "https://modelscope.cn",
            "api_key": "secret-token-1234",
            "model": "sensevoice_zh",
            "device": "auto",
        },
    )

    public = public_model_settings()

    assert public["asr"]["api_key_configured"] is True
    assert public["asr"]["api_key_preview"] == "secret***1234"
    assert "api_key" not in public["asr"]
    assert public["asr"]["config_source"] == "model-settings"


def test_public_model_settings_ignores_comment_placeholder_secret(tmp_path, monkeypatch):
    config_path = tmp_path / "model-settings.json"
    monkeypatch.setenv("MODEL_SETTINGS_FILE", str(config_path))
    monkeypatch.setenv("LLM_API_KEY", "# [必填 when 公网 LLM] Bearer token")

    from app.services.model_config import public_model_settings

    public = public_model_settings()

    assert public["llm"]["api_key_configured"] is False
    assert public["llm"]["api_key_preview"] is None


def test_model_settings_apply_to_runtime(tmp_path, monkeypatch):
    config_path = tmp_path / "model-settings.json"
    monkeypatch.setenv("MODEL_SETTINGS_FILE", str(config_path))

    from app.config import config
    from app.services.model_config import apply_model_settings_to_runtime, update_model_section

    update_model_section(
        "asr",
        {
            "provider": "modelscope",
            "endpoint": "https://modelscope.cn",
            "model": "paraformer_full",
            "device": "cpu",
            "word_timestamps": False,
            "load_timeout_sec": 120,
        },
    )

    apply_model_settings_to_runtime()

    assert config.audio.asr_engine == "paraformer_full"
    assert config.audio.asr_device == "cpu"
    assert config.audio.asr_load_timeout_sec == 120


def test_model_settings_accepts_utf8_bom(tmp_path, monkeypatch):
    config_path = tmp_path / "model-settings.json"
    monkeypatch.setenv("MODEL_SETTINGS_FILE", str(config_path))
    config_path.write_bytes(
        b"\xef\xbb\xbf"
        + json.dumps({"asr": {"model": "sensevoice_zh"}}, ensure_ascii=False).encode("utf-8")
    )

    from app.services.model_config import read_raw_model_settings

    assert read_raw_model_settings()["asr"]["model"] == "sensevoice_zh"


def test_diarization_settings_apply_engine_to_runtime(tmp_path, monkeypatch):
    config_path = tmp_path / "model-settings.json"
    monkeypatch.setenv("MODEL_SETTINGS_FILE", str(config_path))

    from app.services.model_config import apply_model_settings_to_runtime, update_model_section

    update_model_section(
        "diarization",
        {
            "engine": "funasr_campplus",
            "provider": "modelscope",
            "endpoint": "https://modelscope.cn",
            "model": "paraformer-zh + fsmn-vad + ct-punc + cam++",
            "command": "",
            "device": "cpu",
        },
    )

    apply_model_settings_to_runtime()

    assert os.environ["DIARIZATION_ENGINE"] == "funasr_campplus"
    assert os.environ["DIARIZATION_MODEL"] == "paraformer-zh + fsmn-vad + ct-punc + cam++"
    assert os.environ["PYANNOTE_DEVICE"] == "cpu"


def test_face_settings_are_public_and_apply_to_runtime(tmp_path, monkeypatch):
    config_path = tmp_path / "model-settings.json"
    monkeypatch.setenv("MODEL_SETTINGS_FILE", str(config_path))

    from app.config import config
    from app.services.model_config import (
        apply_model_settings_to_runtime,
        public_model_settings,
        update_model_section,
    )

    update_model_section(
        "face",
        {
            "provider": "insightface",
            "endpoint": "file://./models/face/insightface",
            "api_key": "face-secret-1234",
            "model": "buffalo_s",
            "device": "directml",
            "enabled": True,
            "match_threshold": 0.62,
            "match_margin": 0.12,
            "frame_interval_sec": 8,
        },
    )

    public = public_model_settings()
    assert public["face"]["api_key_configured"] is True
    assert public["face"]["api_key_preview"] == "face-s***1234"
    assert public["face"]["model"] == "buffalo_s"
    assert any(provider["key"] == "insightface" for provider in public["providers"])

    apply_model_settings_to_runtime()

    assert config.face.provider == "insightface"
    assert config.face.model == "buffalo_s"
    assert config.face.device == "directml"
    assert config.face.match_threshold == 0.62
    assert config.face.match_margin == 0.12
    assert config.face.frame_interval_sec == 8


def test_update_model_section_preserves_secret_when_api_key_blank(tmp_path, monkeypatch):
    config_path = tmp_path / "model-settings.json"
    monkeypatch.setenv("MODEL_SETTINGS_FILE", str(config_path))

    from app.services.model_config import update_model_section

    update_model_section("llm", {"api_key": "sk-old", "model": "qwen2.5"})
    update_model_section("llm", {"api_key": "", "model": "qwen3"})

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    assert payload["llm"]["api_key"] == "sk-old"
    assert payload["llm"]["model"] == "qwen3"
