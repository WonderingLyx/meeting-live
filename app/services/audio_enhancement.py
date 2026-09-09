"""Lightweight audio enhancement before model inference.

The pipeline intentionally avoids heavyweight denoising models.  It is used in
the hot path for live meetings and upload processing, so failure must degrade
to the original waveform instead of breaking transcription.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger("Matrix_Core")


@dataclass(frozen=True)
class AudioEnhancementSettings:
    enabled: bool = True
    high_pass_hz: float = 80.0
    low_pass_hz: float = 7600.0
    noise_reduction: float = 0.35
    noise_floor: float = 0.08
    target_rms: float = 0.08
    max_gain: float = 8.0
    max_block_seconds: float = 30.0


def _clamp(value: Any, default: float, minimum: float, maximum: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    if not np.isfinite(number):
        number = default
    return float(max(minimum, min(maximum, number)))


def _settings() -> AudioEnhancementSettings:
    try:
        from app.config import config

        audio = config.audio
        return AudioEnhancementSettings(
            enabled=bool(getattr(audio, "enhancement_enabled", True)),
            high_pass_hz=_clamp(
                getattr(audio, "enhancement_high_pass_hz", 80.0), 80.0, 0.0, 1000.0
            ),
            low_pass_hz=_clamp(
                getattr(audio, "enhancement_low_pass_hz", 7600.0), 7600.0, 0.0, 24000.0
            ),
            noise_reduction=_clamp(
                getattr(audio, "enhancement_noise_reduction", 0.35), 0.35, 0.0, 0.95
            ),
            noise_floor=_clamp(
                getattr(audio, "enhancement_noise_floor", 0.08), 0.08, 0.0, 0.8
            ),
            target_rms=_clamp(
                getattr(audio, "enhancement_target_rms", getattr(audio, "target_rms", 0.08)),
                0.08,
                0.005,
                0.5,
            ),
            max_gain=_clamp(
                getattr(audio, "enhancement_max_gain", getattr(audio, "max_gain", 8.0)),
                8.0,
                1.0,
                30.0,
            ),
            max_block_seconds=_clamp(
                getattr(audio, "enhancement_max_block_seconds", 30.0), 30.0, 5.0, 120.0
            ),
        )
    except Exception:
        logger.debug("[AUDIO] enhancement config unavailable; using defaults", exc_info=True)
        return AudioEnhancementSettings()


def enhancement_enabled() -> bool:
    return _settings().enabled


def audio_rms(audio_data: np.ndarray) -> float:
    audio = _sanitize_audio(audio_data)
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


def enhance_audio_for_models(
    audio_data: np.ndarray,
    sample_rate: int,
    *,
    profile: str = "default",
) -> np.ndarray:
    """Return enhanced mono float32 audio for ASR/speaker models.

    Length is preserved so timestamps derived from sample counts remain valid.
    """
    audio = _sanitize_audio(audio_data)
    if audio.size == 0:
        return audio

    settings = _settings()
    if not settings.enabled:
        return audio

    try:
        sr = int(sample_rate)
    except (TypeError, ValueError):
        sr = 16000
    if sr <= 0:
        sr = 16000

    if audio.size < max(160, int(sr * 0.02)):
        return audio

    try:
        block_samples = int(settings.max_block_seconds * sr)
        if block_samples > 0 and audio.size > block_samples:
            enhanced = _enhance_long_audio(audio, sr, settings, block_samples)
        else:
            enhanced = _enhance_block(audio, sr, settings)
        if enhanced.shape != audio.shape:
            enhanced = enhanced[: audio.size]
            if enhanced.size < audio.size:
                enhanced = np.pad(enhanced, (0, audio.size - enhanced.size))
        return np.clip(enhanced, -1.0, 1.0).astype(np.float32, copy=False)
    except Exception as exc:
        logger.warning("[AUDIO] enhancement failed for %s, using original audio: %s", profile, exc)
        return audio


def _sanitize_audio(audio_data: np.ndarray) -> np.ndarray:
    if audio_data is None:
        return np.array([], dtype=np.float32)
    audio = np.asarray(audio_data, dtype=np.float32)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1, dtype=np.float32)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(audio, -1.0, 1.0).astype(np.float32, copy=False)


def _enhance_long_audio(
    audio: np.ndarray,
    sample_rate: int,
    settings: AudioEnhancementSettings,
    block_samples: int,
) -> np.ndarray:
    pieces: list[np.ndarray] = []
    for start in range(0, audio.size, block_samples):
        end = min(audio.size, start + block_samples)
        pieces.append(_enhance_block(audio[start:end], sample_rate, settings))
    return np.concatenate(pieces).astype(np.float32, copy=False)


def _enhance_block(
    audio: np.ndarray,
    sample_rate: int,
    settings: AudioEnhancementSettings,
) -> np.ndarray:
    work = _apply_bandpass(audio, sample_rate, settings.high_pass_hz, settings.low_pass_hz)
    work = _spectral_gate(work, sample_rate, settings.noise_reduction, settings.noise_floor)
    work = _normalize_rms(work, settings.target_rms, settings.max_gain)
    return work.astype(np.float32, copy=False)


def _apply_bandpass(audio: np.ndarray, sample_rate: int, high_hz: float, low_hz: float) -> np.ndarray:
    nyquist = sample_rate / 2.0
    work = audio.astype(np.float32, copy=True)
    try:
        from scipy import signal

        if 0 < high_hz < nyquist * 0.95:
            sos = signal.butter(3, high_hz, btype="highpass", fs=sample_rate, output="sos")
            work = _safe_sos_filter(signal, sos, work)
        if 0 < low_hz < nyquist * 0.98:
            sos = signal.butter(4, low_hz, btype="lowpass", fs=sample_rate, output="sos")
            work = _safe_sos_filter(signal, sos, work)
    except Exception:
        logger.debug("[AUDIO] scipy bandpass unavailable; skipping filters", exc_info=True)
    return work.astype(np.float32, copy=False)


def _safe_sos_filter(signal_module: Any, sos: np.ndarray, audio: np.ndarray) -> np.ndarray:
    try:
        min_len = 3 * (2 * len(sos) + 1)
        if audio.size > min_len and hasattr(signal_module, "sosfiltfilt"):
            return signal_module.sosfiltfilt(sos, audio).astype(np.float32, copy=False)
    except Exception:
        logger.debug("[AUDIO] zero-phase filter failed; falling back to sosfilt", exc_info=True)
    return signal_module.sosfilt(sos, audio).astype(np.float32, copy=False)


def _frame_size(sample_rate: int) -> int:
    if sample_rate <= 12000:
        return 512
    return 1024


def _spectral_gate(
    audio: np.ndarray,
    sample_rate: int,
    strength: float,
    floor: float,
) -> np.ndarray:
    if strength <= 0.0:
        return audio.astype(np.float32, copy=False)
    frame_size = _frame_size(sample_rate)
    if audio.size < frame_size * 3:
        return audio.astype(np.float32, copy=False)
    hop = frame_size // 4
    window = np.hanning(frame_size).astype(np.float32)
    pad = (frame_size - ((audio.size - frame_size) % hop)) % hop
    padded = np.pad(audio.astype(np.float32, copy=False), (0, pad + frame_size), mode="constant")

    specs: list[np.ndarray] = []
    frame_rms: list[float] = []
    for start in range(0, padded.size - frame_size + 1, hop):
        frame = padded[start : start + frame_size] * window
        spectrum = np.fft.rfft(frame)
        specs.append(spectrum)
        frame_rms.append(float(np.sqrt(np.mean(frame * frame))))
    if not specs:
        return audio.astype(np.float32, copy=False)

    rms_arr = np.asarray(frame_rms, dtype=np.float32)
    quiet_cut = float(np.percentile(rms_arr, 30))
    quiet_indices = np.where(rms_arr <= quiet_cut)[0]
    if quiet_indices.size == 0:
        quiet_indices = np.argsort(rms_arr)[: max(1, len(specs) // 5)]
    noise_mag = np.median(
        np.stack([np.abs(specs[int(index)]) for index in quiet_indices], axis=0),
        axis=0,
    ).astype(np.float32)

    output = np.zeros_like(padded, dtype=np.float32)
    norm = np.zeros_like(padded, dtype=np.float32)
    eps = 1e-8
    min_gain = max(0.02, min(1.0, floor))
    for index, spectrum in enumerate(specs):
        mag = np.abs(spectrum).astype(np.float32)
        gain = 1.0 - (strength * noise_mag / (mag + eps))
        gain = np.clip(gain, min_gain, 1.0)
        enhanced = np.fft.irfft(spectrum * gain, n=frame_size).astype(np.float32)
        start = index * hop
        output[start : start + frame_size] += enhanced * window
        norm[start : start + frame_size] += window * window

    valid = norm > eps
    output[valid] /= norm[valid]
    return output[: audio.size].astype(np.float32, copy=False)


def _normalize_rms(audio: np.ndarray, target_rms: float, max_gain: float) -> np.ndarray:
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) if audio.size else 0.0
    if rms <= 1e-6:
        return audio.astype(np.float32, copy=False)
    gain = min(float(max_gain), float(target_rms) / rms)
    gain = max(0.2, gain)
    normalized = audio.astype(np.float32, copy=True) * gain
    peak = float(np.max(np.abs(normalized))) if normalized.size else 0.0
    if peak > 0.98:
        normalized *= 0.98 / peak
    return np.tanh(normalized * 1.2).astype(np.float32) / 1.2
