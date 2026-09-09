"""配置模块测试"""

import os


def test_storage_config_defaults():
    from app.config import StorageConfig
    cfg = StorageConfig.from_env()
    assert cfg.db_path == "./data/matrix.db"


def test_llm_config_defaults():
    from app.config import LLMConfig
    cfg = LLMConfig.from_env()
    assert cfg.enabled is False
    assert cfg.endpoint == "http://127.0.0.1:11434/v1"
    assert cfg.model == "qwen2.5:1.5b"
    assert cfg.timeout_sec == 200


def test_deployment_config_defaults_to_local():
    from app.config import DeploymentConfig
    cfg = DeploymentConfig.from_env()
    assert cfg.mode == "local"


def test_audio_enhancement_config_reads_env(monkeypatch):
    from app.config import AudioConfig

    monkeypatch.setenv("AUDIO_ENHANCEMENT_ENABLED", "false")
    monkeypatch.setenv("AUDIO_ENHANCEMENT_NOISE_REDUCTION", "0.55")
    monkeypatch.setenv("AUDIO_ENHANCEMENT_TARGET_RMS", "0.06")

    cfg = AudioConfig.from_env()

    assert cfg.enhancement_enabled is False
    assert cfg.enhancement_noise_reduction == 0.55
    assert cfg.enhancement_target_rms == 0.06


def test_server_defaults_to_loopback(monkeypatch):
    from app.config import ServerConfig
    monkeypatch.delenv("HOST", raising=False)
    assert ServerConfig.from_env().host == "127.0.0.1"


def test_deployment_config_reads_known_mode(monkeypatch):
    from app.config import DeploymentConfig
    monkeypatch.setenv("DEPLOYMENT_MODE", "LAN")
    cfg = DeploymentConfig.from_env()
    assert cfg.mode == "lan"


def test_llm_config_reads_env(monkeypatch):
    from app.config import LLMConfig
    monkeypatch.setenv("LLM_ENABLED", "true")
    monkeypatch.setenv("LLM_MODEL", "llama3:8b")
    monkeypatch.setenv("LLM_TIMEOUT_SEC", "120")
    cfg = LLMConfig.from_env()
    assert cfg.enabled is True
    assert cfg.model == "llama3:8b"
    assert cfg.timeout_sec == 120


def test_llm_api_key_comment_placeholder_becomes_none(monkeypatch):
    from app.config import LLMConfig

    monkeypatch.setenv("LLM_API_KEY", "# [必填 when 公网 LLM] Bearer token")
    cfg = LLMConfig.from_env()
    assert cfg.api_key is None


def test_llm_allowed_hosts_from_env(monkeypatch):
    from app.config import LLMConfig
    monkeypatch.setenv("LLM_ALLOWED_HOSTS", "127.0.0.1,192.168.1.5,::1")
    cfg = LLMConfig.from_env()
    assert cfg.allowed_hosts == ("127.0.0.1", "192.168.1.5", "::1")


def test_appconfig_load_includes_new_blocks():
    from app.config import AppConfig, StorageConfig, LLMConfig, DeploymentConfig
    cfg = AppConfig.load()
    assert isinstance(cfg.storage, StorageConfig)
    assert isinstance(cfg.llm, LLMConfig)
    assert isinstance(cfg.deployment, DeploymentConfig)


def test_invalid_pyannote_device_falls_back_to_auto(monkeypatch):
    from app.config import SpeakerConfig

    monkeypatch.setenv("PYANNOTE_DEVICE", "metal")

    assert SpeakerConfig.from_env().diarization_device == "auto"


def test_load_project_env_accepts_windows_ansi_gbk(tmp_path, monkeypatch):
    from app.config import load_project_env

    key = "MATRIX_TEST_GBK_ENV_VALUE"
    monkeypatch.delenv(key, raising=False)
    env_path = tmp_path / ".env"
    env_path.write_bytes(f"{key}=中文\n".encode("gb18030"))

    encoding = load_project_env(env_path)

    assert encoding in {"gb18030", "cp936", "mbcs"}
    assert os.environ[key] == "中文"
