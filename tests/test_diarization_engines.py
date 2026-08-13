import json


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
