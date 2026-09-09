import numpy as np
import pytest
import soundfile as sf

from app.services.audio_files import (
    inspect_media_file,
    media_diagnostic_summary,
    split_audio_into_chunks,
    transcode_audio_to_wav,
    validate_audio_file,
)


def _mp4_atom(name: bytes, payload: bytes) -> bytes:
    return (len(payload) + 8).to_bytes(4, "big") + name + payload


def test_split_audio_rejects_invalid_overlap():
    with pytest.raises(ValueError):
        split_audio_into_chunks(np.zeros(16000), 16000, 1, 1)


def test_split_audio_returns_absolute_boundaries():
    chunks = split_audio_into_chunks(np.zeros(48000), 16000, 2, 0.5)
    assert [(start, end) for _, start, end in chunks] == [(0, 2), (1.5, 3)]


def test_validate_audio_rejects_mislabeled_file(tmp_path, monkeypatch):
    path = tmp_path / "fake.wav"
    path.write_text("not audio", encoding="utf-8")
    import librosa
    monkeypatch.setattr(librosa, "load", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("bad audio")))
    with pytest.raises(ValueError):
        validate_audio_file(path)


def test_validate_audio_returns_duration(tmp_path):
    path = tmp_path / "valid.wav"
    sf.write(str(path), np.zeros(16000, dtype=np.float32), 16000)
    assert validate_audio_file(path) == pytest.approx(1.0)


def test_inspect_mp4_reports_missing_moov(tmp_path):
    path = tmp_path / "broken.mp4"
    path.write_bytes(
        _mp4_atom(b"ftyp", b"isom\x00\x00\x02\x00isomiso2")
        + _mp4_atom(b"mdat", b"\x00" * 32)
    )

    info = inspect_media_file(path)

    assert info["kind"] == "mp4"
    assert info["has_ftyp"] is True
    assert info["has_mdat"] is True
    assert info["has_moov"] is False
    assert info["atoms"] == ["ftyp@0+24", "mdat@24+40"]
    assert "has_moov=False" in media_diagnostic_summary(path)


def test_transcode_missing_moov_error_includes_diagnostic(tmp_path, monkeypatch):
    from app.services import audio_files

    path = tmp_path / "broken.mp4"
    path.write_bytes(
        _mp4_atom(b"ftyp", b"isom\x00\x00\x02\x00isomiso2")
        + _mp4_atom(b"mdat", b"\x00" * 32)
    )

    class Result:
        returncode = 1
        stderr = "moov atom not found"
        stdout = ""

    monkeypatch.setattr(audio_files, "_resolve_media_tool", lambda name: "ffmpeg.exe")
    monkeypatch.setattr(audio_files.subprocess, "run", lambda *args, **kwargs: Result())

    with pytest.raises(ValueError) as excinfo:
        transcode_audio_to_wav(path, tmp_path / "out.wav")

    message = str(excinfo.value)
    assert "缺少 moov" in message
    assert "has_moov=False" in message
