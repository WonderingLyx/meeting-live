from __future__ import annotations

import numpy as np


def _set_enhancement_defaults(config):
    config.audio.enhancement_enabled = True
    config.audio.enhancement_high_pass_hz = 0.0
    config.audio.enhancement_low_pass_hz = 0.0
    config.audio.enhancement_noise_reduction = 0.45
    config.audio.enhancement_noise_floor = 0.08
    config.audio.enhancement_target_rms = 0.005
    config.audio.enhancement_max_gain = 1.0
    config.audio.enhancement_max_block_seconds = 30.0


def test_enhancement_disabled_returns_sanitized_audio():
    from app.config import config
    from app.services.audio_enhancement import enhance_audio_for_models

    config.audio.enhancement_enabled = False
    audio = np.array([0.1, np.nan, np.inf, -2.0, 2.0], dtype=np.float32)

    result = enhance_audio_for_models(audio, 16000)

    assert result.dtype == np.float32
    assert np.isfinite(result).all()
    np.testing.assert_array_equal(
        result,
        np.array([0.1, 0.0, 0.0, -1.0, 1.0], dtype=np.float32),
    )


def test_enhancement_preserves_length_and_bounds():
    from app.config import config
    from app.services.audio_enhancement import enhance_audio_for_models

    _set_enhancement_defaults(config)
    rng = np.random.default_rng(42)
    audio = rng.normal(0.0, 0.03, 16000).astype(np.float32)

    result = enhance_audio_for_models(audio, 16000)

    assert result.dtype == np.float32
    assert len(result) == len(audio)
    assert np.isfinite(result).all()
    assert float(np.max(np.abs(result))) <= 1.0


def test_spectral_gate_reduces_stationary_noise_when_gain_is_capped():
    from app.config import config
    from app.services.audio_enhancement import audio_rms, enhance_audio_for_models

    _set_enhancement_defaults(config)
    config.audio.enhancement_noise_reduction = 0.7
    rng = np.random.default_rng(7)
    noise_only = rng.normal(0.0, 0.03, 16000 * 2).astype(np.float32)

    result = enhance_audio_for_models(noise_only, 16000)

    assert audio_rms(result) < audio_rms(noise_only)
