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
    assert "sherpa_onnx_cli" in engines
    assert engines["sherpa_onnx_cli"]["dependency"] == "external_command"
