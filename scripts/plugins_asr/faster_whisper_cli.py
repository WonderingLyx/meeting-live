"""faster-whisper command adapter for ASR plugins.

This script is optional. Install it only in the environment where the plugin is
used:

    python -m pip install faster-whisper
"""
from __future__ import annotations

import argparse
import json
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--language", default="zh")
    parser.add_argument("--compute-type", default="")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--vad-filter", action="store_true")
    parser.add_argument("--word-timestamps", action="store_true")
    args = parser.parse_args()

    try:
        from faster_whisper import WhisperModel
    except Exception as exc:
        print(json.dumps({"text": "", "provider_metadata": {"error": f"faster-whisper import failed: {exc}"}}))
        return 2

    device = args.device if args.device in {"cuda", "cpu"} else "cpu"
    compute_type = args.compute_type or ("float16" if device == "cuda" else "int8")
    model = WhisperModel(args.model, device=device, compute_type=compute_type)
    language = None if args.language in {"", "auto", "none"} else args.language
    segments_iter, info = model.transcribe(
        args.audio,
        language=language,
        beam_size=args.beam_size,
        vad_filter=args.vad_filter,
        word_timestamps=args.word_timestamps,
    )

    segments = []
    words = []
    text_parts = []
    for segment in segments_iter:
        seg_words = []
        for word in getattr(segment, "words", None) or []:
            item = {
                "text": word.word,
                "start": float(word.start),
                "end": float(word.end),
            }
            words.append(item)
            seg_words.append(item)
        text = str(segment.text or "").strip()
        if text:
            text_parts.append(text)
        segments.append({
            "text": text,
            "start": float(segment.start),
            "end": float(segment.end),
            "speaker": None,
            "words": seg_words or None,
        })

    result = {
        "text": "".join(text_parts).strip(),
        "segments": segments or None,
        "words": words or None,
        "language": getattr(info, "language", language),
        "provider_metadata": {
            "adapter": "faster-whisper",
            "model": args.model,
            "device": device,
            "compute_type": compute_type,
            "language_probability": getattr(info, "language_probability", None),
        },
        "timestamp_origin": "chunk",
        "speaker_scope": "none",
        "mode": "segmented",
        "is_final": True,
    }
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
