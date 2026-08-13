import pytest


@pytest.mark.anyio
async def test_probe_model_source_accepts_local_existing_path(tmp_path):
    from app.api.settings import _probe_model_source

    result = await _probe_model_source(
        "diarization",
        "local",
        f"file://{tmp_path}",
        None,
    )

    assert result["ok"] is True
    assert result["provider"] == "local"


@pytest.mark.anyio
async def test_probe_model_source_rejects_bad_endpoint():
    from app.api.settings import _probe_model_source
    from fastapi import HTTPException

    with pytest.raises(HTTPException):
        await _probe_model_source("asr", "custom", "not-a-url", None)
