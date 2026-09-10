"""Lightweight audio enhancement before model inference.

The pipeline intentionally avoids heavyweight denoising models.  It is used in
the hot path for live meetings and upload processing, so failure must degrade
to the original waveform instead of breaking transcription.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

logger = logging.getLogger("Matrix_Core")


@dataclass(frozen=True)
class AudioEnhancementSettings:
    enabled: bool = True
    profile: str = "meeting"
    high_pass_hz: float = 80.0
    low_pass_hz: float = 7600.0
    noise_reduction: float = 0.35
    noise_floor: float = 0.08
    target_rms: float = 0.08
    max_gain: float = 8.0
    max_block_seconds: float = 30.0
    compression_threshold: float = 0.12
    compression_ratio: float = 2.5
    overlap_detection_enabled: bool = True
    overlap_window_seconds: float = 0.8
    overlap_hop_seconds: float = 0.2
    overlap_threshold: float = 0.68
    overlap_min_duration: float = 0.45


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
            profile=str(getattr(audio, "enhancement_profile", "meeting") or "meeting"),
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
            compression_threshold=_clamp(
                getattr(audio, "enhancement_compression_threshold", 0.12),
                0.12,
                0.02,
                0.8,
            ),
            compression_ratio=_clamp(
                getattr(audio, "enhancement_compression_ratio", 2.5),
                2.5,
                1.0,
                10.0,
            ),
            overlap_detection_enabled=bool(
                getattr(audio, "overlap_detection_enabled", True)
            ),
            overlap_window_seconds=_clamp(
                getattr(audio, "overlap_window_seconds", 0.8),
                0.8,
                0.25,
                2.0,
            ),
            overlap_hop_seconds=_clamp(
                getattr(audio, "overlap_hop_seconds", 0.2),
                0.2,
                0.05,
                1.0,
            ),
            overlap_threshold=_clamp(
                getattr(audio, "overlap_threshold", 0.68),
                0.68,
                0.3,
                0.95,
            ),
            overlap_min_duration=_clamp(
                getattr(audio, "overlap_min_duration", 0.45),
                0.45,
                0.1,
                3.0,
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


def _profile_settings(settings: AudioEnhancementSettings, profile: str) -> AudioEnhancementSettings:
    requested = (profile or "default").lower().replace("-", "_")
    configured = settings.profile.lower().replace("-", "_")
    if requested.startswith("funasr:"):
        requested = "asr"
    if requested in {"default", "upload_meeting", "live_segment", "asr", "playback"}:
        requested = configured or "meeting"
    if requested in {"off", "none", "disabled"}:
        return replace(settings, enabled=False)
    if requested in {"voice_sample", "speaker", "diarization", "speaker_identity"}:
        return replace(
            settings,
            high_pass_hz=max(60.0, settings.high_pass_hz * 0.75),
            low_pass_hz=settings.low_pass_hz,
            noise_reduction=min(settings.noise_reduction, 0.22),
            noise_floor=max(settings.noise_floor, 0.16),
            target_rms=min(settings.target_rms, 0.075),
            max_gain=min(settings.max_gain, 5.0),
            compression_ratio=min(settings.compression_ratio, 1.8),
        )
    if requested in {"aggressive", "noisy"}:
        return replace(
            settings,
            high_pass_hz=max(settings.high_pass_hz, 100.0),
            low_pass_hz=min(settings.low_pass_hz, 7200.0),
            noise_reduction=max(settings.noise_reduction, 0.58),
            noise_floor=min(settings.noise_floor, 0.06),
            target_rms=max(settings.target_rms, 0.09),
            max_gain=max(settings.max_gain, 10.0),
            compression_threshold=min(settings.compression_threshold, 0.10),
            compression_ratio=max(settings.compression_ratio, 3.2),
        )
    if requested in {"light", "safe"}:
        return replace(
            settings,
            noise_reduction=min(settings.noise_reduction, 0.25),
            noise_floor=max(settings.noise_floor, 0.14),
            target_rms=min(settings.target_rms, 0.075),
            max_gain=min(settings.max_gain, 6.0),
            compression_ratio=min(settings.compression_ratio, 2.0),
        )
    return replace(
        settings,
        high_pass_hz=max(settings.high_pass_hz, 80.0),
        low_pass_hz=min(settings.low_pass_hz, 7600.0),
        noise_reduction=max(settings.noise_reduction, 0.42),
        noise_floor=min(settings.noise_floor, 0.08),
        target_rms=max(settings.target_rms, 0.08),
        max_gain=max(settings.max_gain, 8.0),
        compression_threshold=min(settings.compression_threshold, 0.12),
        compression_ratio=max(settings.compression_ratio, 2.5),
    )


def enhancement_metadata(profile: str = "default") -> dict[str, Any]:
    settings = _profile_settings(_settings(), profile)
    return {
        "enabled": settings.enabled,
        "profile": profile,
        "configured_profile": settings.profile,
        "high_pass_hz": settings.high_pass_hz,
        "low_pass_hz": settings.low_pass_hz,
        "noise_reduction": settings.noise_reduction,
        "noise_floor": settings.noise_floor,
        "target_rms": settings.target_rms,
        "max_gain": settings.max_gain,
        "compression_threshold": settings.compression_threshold,
        "compression_ratio": settings.compression_ratio,
    }


def audio_quality_report(audio_data: np.ndarray, sample_rate: int) -> dict[str, Any]:
    """Return a compact, model-agnostic quality estimate for diagnostics."""
    audio = _sanitize_audio(audio_data)
    if audio.size == 0:
        return {
            "label": "empty",
            "score": 0.0,
            "rms": 0.0,
            "peak": 0.0,
            "snr_db": None,
            "silence_ratio": 1.0,
            "clipping_ratio": 0.0,
            "dynamic_range_db": 0.0,
        }
    try:
        sr = int(sample_rate)
    except (TypeError, ValueError):
        sr = 16000
    sr = max(8000, sr)
    rms_values = _window_rms(audio, max(256, int(sr * 0.05)), max(128, int(sr * 0.025)))
    rms = audio_rms(audio)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    clipping_ratio = float(np.mean(np.abs(audio) >= 0.98))
    if rms_values.size:
        noise_floor = float(np.percentile(rms_values, 10))
        speech_level = float(np.percentile(rms_values, 80))
        silence_cut = max(0.003, noise_floor * 1.8)
        silence_ratio = float(np.mean(rms_values <= silence_cut))
        snr_db = 20.0 * np.log10((speech_level + 1e-8) / (noise_floor + 1e-8))
        low = max(float(np.percentile(rms_values, 20)), 1e-8)
        high = max(float(np.percentile(rms_values, 95)), low)
        dynamic_range_db = 20.0 * np.log10(high / low)
    else:
        silence_ratio = 1.0 if rms < 0.003 else 0.0
        snr_db = None
        dynamic_range_db = 0.0

    score = 1.0
    if snr_db is not None:
        if snr_db < 6:
            score -= 0.35
        elif snr_db < 10:
            score -= 0.18
    if clipping_ratio > 0.005:
        score -= 0.35
    elif clipping_ratio > 0.001:
        score -= 0.15
    if silence_ratio > 0.75:
        score -= 0.25
    if rms < 0.004:
        score -= 0.3
    if dynamic_range_db > 28:
        score -= 0.12
    score = round(float(max(0.0, min(1.0, score))), 4)

    if rms < 0.003 or silence_ratio > 0.9:
        label = "mostly_silent"
    elif clipping_ratio > 0.005:
        label = "clipped"
    elif snr_db is not None and snr_db < 8:
        label = "noisy"
    elif dynamic_range_db > 28:
        label = "uneven_volume"
    else:
        label = "ok"
    return {
        "label": label,
        "score": score,
        "rms": round(float(rms), 6),
        "peak": round(float(peak), 6),
        "snr_db": round(float(snr_db), 2) if snr_db is not None and np.isfinite(snr_db) else None,
        "silence_ratio": round(float(silence_ratio), 4),
        "clipping_ratio": round(float(clipping_ratio), 6),
        "dynamic_range_db": round(float(dynamic_range_db), 2),
    }


def detect_overlapped_speech(
    audio_data: np.ndarray,
    sample_rate: int,
) -> list[dict[str, float]]:
    """Heuristic overlapped-speech detector for single-channel meeting audio.

    This is intentionally conservative and dependency-light. It marks regions
    that look spectrally complex for speech, but it does not replace a trained
    overlap model such as pyannote OSD.
    """
    settings = _settings()
    if not settings.overlap_detection_enabled:
        return []
    audio = _sanitize_audio(audio_data)
    if audio.size == 0:
        return []
    try:
        sr = int(sample_rate)
    except (TypeError, ValueError):
        sr = 16000
    sr = max(8000, sr)
    window = max(512, int(sr * settings.overlap_window_seconds))
    hop = max(160, int(sr * settings.overlap_hop_seconds))
    if audio.size < window:
        if audio.size < max(512, int(sr * 0.25)):
            return []
        window = audio.size
        hop = max(160, window // 2)

    rms_values = _window_rms(audio, max(256, int(sr * 0.05)), max(128, int(sr * 0.025)))
    if rms_values.size == 0:
        return []
    speech_threshold = max(0.01, float(np.percentile(rms_values, 55)) * 1.15)
    raw_regions: list[dict[str, float]] = []
    taper = np.hanning(window).astype(np.float32)
    for start in range(0, audio.size - window + 1, hop):
        frame = audio[start : start + window]
        rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))))
        if rms < speech_threshold:
            continue
        score = _overlap_complexity_score(frame * taper, sr, speech_threshold)
        if score >= settings.overlap_threshold:
            raw_regions.append({
                "start": round(start / sr, 3),
                "end": round((start + window) / sr, 3),
                "score": round(float(score), 4),
            })
    return _merge_scored_regions(raw_regions, settings.overlap_min_duration)


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

    settings = _profile_settings(_settings(), profile)
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
    work = _compress_dynamics(
        work,
        threshold=settings.compression_threshold,
        ratio=settings.compression_ratio,
    )
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


def _compress_dynamics(audio: np.ndarray, threshold: float, ratio: float) -> np.ndarray:
    if ratio <= 1.0 or threshold <= 0.0:
        return audio.astype(np.float32, copy=False)
    work = audio.astype(np.float32, copy=True)
    magnitude = np.abs(work)
    mask = magnitude > threshold
    if not np.any(mask):
        return work
    compressed = threshold + (magnitude[mask] - threshold) / ratio
    work[mask] = np.sign(work[mask]) * compressed
    return work.astype(np.float32, copy=False)


def _window_rms(audio: np.ndarray, frame_size: int, hop: int) -> np.ndarray:
    if audio.size == 0 or frame_size <= 0 or hop <= 0:
        return np.array([], dtype=np.float32)
    if audio.size < frame_size:
        return np.asarray([audio_rms(audio)], dtype=np.float32)
    values: list[float] = []
    for start in range(0, audio.size - frame_size + 1, hop):
        frame = audio[start : start + frame_size]
        values.append(float(np.sqrt(np.mean(np.square(frame, dtype=np.float64)))))
    return np.asarray(values, dtype=np.float32)


def _overlap_complexity_score(
    frame: np.ndarray,
    sample_rate: int,
    speech_threshold: float,
) -> float:
    spectrum = np.abs(np.fft.rfft(frame)).astype(np.float64) + 1e-10
    freqs = np.fft.rfftfreq(frame.size, d=1.0 / sample_rate)
    speech_mask = (freqs >= 120) & (freqs <= min(3800, sample_rate / 2 - 1))
    pitch_mask = (freqs >= 85) & (freqs <= min(520, sample_rate / 2 - 1))
    if not np.any(speech_mask) or not np.any(pitch_mask):
        return 0.0

    speech = spectrum[speech_mask]
    pitch = spectrum[pitch_mask]
    flatness = float(np.exp(np.mean(np.log(speech))) / (np.mean(speech) + 1e-10))
    probs = speech / (float(np.sum(speech)) + 1e-10)
    entropy = float(-np.sum(probs * np.log(probs + 1e-10)) / np.log(max(2, probs.size)))

    if pitch.size >= 3:
        local_max = (pitch[1:-1] > pitch[:-2]) & (pitch[1:-1] > pitch[2:])
        peak_floor = max(float(np.percentile(pitch, 82)), float(np.max(pitch)) * 0.10)
        peak_count = int(np.sum(local_max & (pitch[1:-1] >= peak_floor)))
    else:
        peak_count = 0
    low_energy = float(np.sum(spectrum[(freqs >= 150) & (freqs < 900)]))
    high_energy = float(np.sum(spectrum[(freqs >= 900) & (freqs <= 3600)]))
    balance = min(low_energy, high_energy) / (max(low_energy, high_energy) + 1e-10)
    rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))))
    loudness = min(1.0, max(0.0, (rms / max(speech_threshold, 1e-5) - 1.0) / 3.0))

    pitch_score = min(1.0, peak_count / 4.0)
    entropy_score = min(1.0, max(0.0, (entropy - 0.58) / 0.28))
    flatness_score = min(1.0, max(0.0, (flatness - 0.035) / 0.18))
    balance_score = min(1.0, balance * 1.7)
    return float(
        0.34 * pitch_score
        + 0.27 * entropy_score
        + 0.22 * flatness_score
        + 0.12 * balance_score
        + 0.05 * loudness
    )


def _merge_scored_regions(
    regions: list[dict[str, float]],
    min_duration: float,
) -> list[dict[str, float]]:
    if not regions:
        return []
    merged: list[dict[str, float]] = []
    for region in regions:
        if not merged or region["start"] > merged[-1]["end"]:
            merged.append(dict(region))
            continue
        current = merged[-1]
        current["end"] = max(current["end"], region["end"])
        current["score"] = max(current["score"], region["score"])
    return [
        {
            "start": round(float(region["start"]), 3),
            "end": round(float(region["end"]), 3),
            "score": round(float(region["score"]), 4),
        }
        for region in merged
        if region["end"] - region["start"] >= min_duration
    ]


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
