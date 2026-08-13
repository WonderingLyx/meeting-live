import json


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


def test_update_model_section_preserves_secret_when_api_key_blank(tmp_path, monkeypatch):
    config_path = tmp_path / "model-settings.json"
    monkeypatch.setenv("MODEL_SETTINGS_FILE", str(config_path))

    from app.services.model_config import update_model_section

    update_model_section("llm", {"api_key": "sk-old", "model": "qwen2.5"})
    update_model_section("llm", {"api_key": "", "model": "qwen3"})

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    assert payload["llm"]["api_key"] == "sk-old"
    assert payload["llm"]["model"] == "qwen3"
