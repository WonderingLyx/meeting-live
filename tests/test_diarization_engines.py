import json
import sys
import types


def test_diarization_status_prefers_file_engine_over_stale_env(tmp_path, monkeypatch):
    config_path = tmp_path / "model-settings.json"
    config_path.write_text(
        json.dumps({
            "diarization": {
                "engine": "funasr_sensevoice_campplus",
                "provider": "modelscope",
                "endpoint": "https://modelscope.cn",
                "model": "iic/SenseVoiceSmall + fsmn-vad + cam++",
                "device": "auto",
            }
        }),
        encoding="utf-8",
    )
    monkeypatch.setenv("MODEL_SETTINGS_FILE", str(config_path))
    monkeypatch.setenv("DIARIZATION_ENGINE", "funasr_campplus")
    monkeypatch.setenv("DIARIZATION_MODEL", "paraformer-zh + fsmn-vad + ct-punc + cam++")

    from app.api.settings import _diarization_status

    status = _diarization_status()

    assert status["engine"] == "funasr_sensevoice_campplus"
    assert status["model_id"] == "iic/SenseVoiceSmall + fsmn-vad + cam++"


def test_configured_diarization_engine_prefers_file_over_stale_env(tmp_path, monkeypatch):
    config_path = tmp_path / "model-settings.json"
    config_path.write_text(
        json.dumps({
            "diarization": {
                "engine": "funasr_campplus_cn_en",
                "provider": "modelscope",
                "endpoint": "https://modelscope.cn",
                "model": "paraformer-zh + fsmn-vad + ct-punc + iic/speech_campplus_sv_zh_en_16k-common_advanced",
                "device": "auto",
            }
        }),
        encoding="utf-8",
    )
    monkeypatch.setenv("MODEL_SETTINGS_FILE", str(config_path))
    monkeypatch.setenv("DIARIZATION_ENGINE", "pyannote_community")

    from app.services.pyannote_diarization import configured_diarization_engine

    assert configured_diarization_engine() == "funasr_campplus_cn_en"


def test_parse_json_diarization_output_ms_and_seconds():
    from app.services.pyannote_diarization import parse_diarization_output

    raw = json.dumps({
        "segments": [
            {"start": 0, "end": 1200, "speaker": "0"},
            {"start_sec": 1.3, "end_sec": 2.0, "speaker": "0"},
            {"start": 2500, "end": 3000, "speaker": "alice"},
        ]
    })

    assert parse_diarization_output(raw) == [
        (0.0, 2.0, "SPEAKER_00"),
        (2.5, 3.0, "SPEAKER_alice"),
    ]


def test_parse_rttm_diarization_output():
    from app.services.pyannote_diarization import parse_diarization_output

    raw = "SPEAKER meeting 1 0.50 1.25 <NA> <NA> SPEAKER_01 <NA> <NA>"

    assert parse_diarization_output(raw) == [(0.5, 1.75, "SPEAKER_01")]


def test_diarization_engine_catalog_contains_local_backends():
    from app.services.pyannote_diarization import get_all_diarization_engines

    engines = get_all_diarization_engines()

    assert "funasr_campplus" in engines
    assert engines["funasr_campplus"]["provider"] == "modelscope"
    assert "funasr_sensevoice_campplus" in engines
    assert engines["funasr_sensevoice_campplus"]["runtime"] == "funasr"
    assert "funasr_campplus_cn_en" in engines
    assert "speech_campplus_sv_zh_en" in engines["funasr_campplus_cn_en"]["model"]
    assert "funasr_paraformer_large_campplus" in engines
    assert "funasr_eres2netv2" in engines
    assert engines["funasr_eres2netv2"]["experimental"] is True
    assert "sherpa_onnx_cli" in engines
    assert engines["sherpa_onnx_cli"]["dependency"] == "external_command"


def test_diarization_engine_aliases_cover_added_funasr_models():
    from app.services.pyannote_diarization import normalize_diarization_engine

    assert normalize_diarization_engine("sensevoice_campplus") == "funasr_sensevoice_campplus"
    assert normalize_diarization_engine("campplus-cn-en") == "funasr_campplus_cn_en"
    assert normalize_diarization_engine("paraformer-large-campplus") == "funasr_paraformer_large_campplus"
    assert normalize_diarization_engine("eres2netv2") == "funasr_eres2netv2"


def test_funasr_diarization_rocm_guard_marks_paraformer_unavailable(monkeypatch):
    from app.services import pyannote_diarization as diarization
    from engine.asr import funasr_engine

    fake_torch = types.SimpleNamespace(
        version=types.SimpleNamespace(hip="7.2.1"),
        cuda=types.SimpleNamespace(is_available=lambda: True),
        backends=types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: False)
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delenv("ASR_FUNASR_ALLOW_ROCM_PARAFORMER", raising=False)
    monkeypatch.setattr(funasr_engine, "_installed_funasr_version", lambda: "1.4.13")
    monkeypatch.setattr(diarization, "_module_available", lambda name: name == "funasr")

    info = diarization.get_diarization_engine_info("funasr_campplus")

    assert info["available"] is False
    assert info["dependency_available"] is False
    assert "0xC0000005" in info["reason"]
    assert "funasr==1.4.1" in info["install_hint"]


def test_funasr_diarization_rocm_guard_keeps_sensevoice_available(monkeypatch):
    from app.services import pyannote_diarization as diarization
    from engine.asr import funasr_engine

    fake_torch = types.SimpleNamespace(
        version=types.SimpleNamespace(hip="7.2.1"),
        cuda=types.SimpleNamespace(is_available=lambda: True),
        backends=types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: False)
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delenv("ASR_FUNASR_ALLOW_ROCM_PARAFORMER", raising=False)
    monkeypatch.setattr(funasr_engine, "_installed_funasr_version", lambda: "1.4.13")
    monkeypatch.setattr(diarization, "_module_available", lambda name: name == "funasr")

    info = diarization.get_diarization_engine_info("funasr_sensevoice_campplus")

    assert info["available"] is True


def test_funasr_diarizer_guard_stops_before_model_load(monkeypatch):
    from app.services import pyannote_diarization as diarization
    from engine.asr import funasr_engine

    fake_torch = types.SimpleNamespace(
        version=types.SimpleNamespace(hip="7.2.1"),
        cuda=types.SimpleNamespace(is_available=lambda: True),
        backends=types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: False)
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delenv("ASR_FUNASR_ALLOW_ROCM_PARAFORMER", raising=False)
    monkeypatch.setattr(funasr_engine, "_installed_funasr_version", lambda: "1.4.13")
    diarization.FunASRCampPlusDiarizer.reset()

    diarizer = diarization.FunASRCampPlusDiarizer("funasr_campplus")

    assert diarizer.enabled is False
    assert "0xC0000005" in (diarizer.last_error or "")
