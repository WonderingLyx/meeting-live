"""Audio ingestion primitives shared by APIs and background processing."""
import asyncio
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import BinaryIO

import numpy as np


ALLOWED_AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".mp4",
    ".webm",
    ".flac",
    ".ogg",
    ".aac",
    ".wma",
}
UPLOAD_TRANSCODE_REQUIRED_EXTENSIONS = ALLOWED_AUDIO_EXTENSIONS - {".wav"}
UPLOAD_READ_SIZE = 1024 * 1024


class UploadTooLargeError(ValueError):
    """Raised after an upload crosses its configured byte limit."""


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_media_tool(name: str) -> str | None:
    env_keys = (
        ("FFMPEG_BINARY", "FFMPEG_PATH")
        if name == "ffmpeg"
        else ("FFPROBE_BINARY", "FFPROBE_PATH")
    )
    candidates: list[str | None] = [os.getenv(key) for key in env_keys]
    exe_name = f"{name}.exe" if os.name == "nt" else name
    root = _project_root()
    candidates.extend(
        str(path)
        for path in (
            root / ".runtime" / "ffmpeg" / "bin" / exe_name,
            root / "offline" / "ffmpeg" / "bin" / exe_name,
            root / "tools" / "ffmpeg" / "bin" / exe_name,
            root / "ffmpeg" / "bin" / exe_name,
        )
    )
    candidates.append(shutil.which(name) or shutil.which(exe_name))
    if name == "ffmpeg":
        try:
            import imageio_ffmpeg

            candidates.append(imageio_ffmpeg.get_ffmpeg_exe())
        except Exception:
            pass

    for candidate in candidates:
        if not candidate:
            continue
        try:
            path = Path(candidate)
            if path.is_file():
                return str(path)
        except OSError:
            continue
    return None


def audio_transcode_available() -> bool:
    return _resolve_media_tool("ffmpeg") is not None


def inspect_media_file(path: str | Path, *, max_atoms: int = 64) -> dict[str, object]:
    """Return lightweight container diagnostics without decoding media data."""
    source = Path(path)
    if not source.is_file():
        return {"exists": False, "path": str(source)}

    size = source.stat().st_size
    with source.open("rb") as fh:
        head = fh.read(32)
    info: dict[str, object] = {
        "exists": True,
        "path": str(source),
        "size": size,
        "head_hex": head[:16].hex(),
        "kind": "unknown",
        "atoms": [],
        "has_ftyp": False,
        "has_mdat": False,
        "has_moov": False,
        "parse_error": "",
    }
    if head.startswith(b"RIFF"):
        info["kind"] = "riff"
        return info
    if head.startswith(b"\x1aE\xdf\xa3"):
        info["kind"] = "webm"
        return info
    if len(head) < 12 or head[4:8] != b"ftyp":
        return info

    info["kind"] = "mp4"
    atoms: list[str] = []
    pos = 0
    try:
        with source.open("rb") as fh:
            for _ in range(max_atoms):
                if pos + 8 > size:
                    break
                fh.seek(pos)
                header = fh.read(16)
                if len(header) < 8:
                    break
                atom_size = int.from_bytes(header[0:4], "big")
                atom_type_raw = header[4:8]
                atom_type = atom_type_raw.decode("ascii", errors="replace")
                header_size = 8
                if atom_size == 1:
                    if len(header) < 16:
                        info["parse_error"] = f"incomplete large atom header at {pos}"
                        break
                    atom_size = int.from_bytes(header[8:16], "big")
                    header_size = 16
                elif atom_size == 0:
                    atom_size = size - pos
                if atom_size < header_size:
                    info["parse_error"] = f"invalid atom {atom_type}@{pos} size={atom_size}"
                    break
                atoms.append(f"{atom_type}@{pos}+{atom_size}")
                if atom_type == "ftyp":
                    info["has_ftyp"] = True
                elif atom_type == "mdat":
                    info["has_mdat"] = True
                elif atom_type == "moov":
                    info["has_moov"] = True
                next_pos = pos + atom_size
                if next_pos <= pos or next_pos > size:
                    if next_pos > size:
                        info["parse_error"] = (
                            f"atom {atom_type}@{pos} extends past EOF: {next_pos}>{size}"
                        )
                    break
                pos = next_pos
    except OSError as exc:
        info["parse_error"] = str(exc)

    info["atoms"] = atoms
    return info


def media_diagnostic_summary(path: str | Path) -> str:
    info = inspect_media_file(path)
    if not info.get("exists"):
        return "diagnostic=missing-file"
    atoms = ",".join((info.get("atoms") or [])[:8]) or "none"
    parse_error = str(info.get("parse_error") or "")
    if parse_error:
        parse_error = f" parse_error={parse_error}"
    return (
        f"diagnostic kind={info.get('kind')} size={info.get('size')} "
        f"head={info.get('head_hex')} has_ftyp={info.get('has_ftyp')} "
        f"has_mdat={info.get('has_mdat')} has_moov={info.get('has_moov')} "
        f"atoms={atoms}{parse_error}"
    )


def transcode_audio_to_wav(
    source: str | Path,
    target: str | Path,
    *,
    sample_rate: int = 16000,
    timeout_sec: float | None = None,
) -> Path:
    """Convert an uploaded compressed/container audio file to internal PCM WAV."""
    ffmpeg = _resolve_media_tool("ffmpeg")
    if not ffmpeg:
        raise ValueError(
            "当前环境无法解码该音频容器: 未找到 FFmpeg。请重新运行一键安装,或在当前虚拟环境安装 imageio-ffmpeg。"
        )

    source = Path(source)
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.stem}.{uuid.uuid4().hex}.tmp.wav")
    command = [
        ffmpeg,
        "-hide_banner",
        "-y",
        "-v",
        "error",
        "-nostdin",
        "-i",
        str(source),
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(tmp),
    ]
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"音频转码超时: 超过 {timeout_sec:g}s") from exc

    if result.returncode != 0:
        tmp.unlink(missing_ok=True)
        detail = (result.stderr or result.stdout or "").strip()
        if detail:
            detail = detail[:500]
        else:
            detail = f"ffmpeg exit code {result.returncode}"
        if "moov atom not found" in detail.lower():
            raise ValueError(
                "MP4 文件缺少 moov 索引，通常表示录制/导出/下载未完成或文件被截断，"
                f"无法可靠转写。{media_diagnostic_summary(source)}"
            )
        raise ValueError(
            f"音频无法用 FFmpeg 转为 WAV: {detail}。{media_diagnostic_summary(source)}"
        )
    if not tmp.is_file() or tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        raise ValueError("音频无法用 FFmpeg 转为 WAV: 输出为空")

    tmp.replace(target)
    return target


async def persist_upload(upload, target: Path, *, max_bytes: int) -> int:
    """Stream an UploadFile to disk without blocking the event loop on writes."""
    size = 0
    output: BinaryIO | None = None
    try:
        output = await asyncio.to_thread(target.open, "wb")
        while chunk := await upload.read(UPLOAD_READ_SIZE):
            size += len(chunk)
            if size > max_bytes:
                raise UploadTooLargeError
            await asyncio.to_thread(output.write, chunk)
        await asyncio.to_thread(output.flush)
        return size
    finally:
        if output is not None:
            await asyncio.to_thread(output.close)


def validate_audio_file(path: str | Path, *, max_duration_sec: float | None = None) -> float:
    """Decode a small prefix and return duration, rejecting mislabeled/non-audio files."""
    import librosa

    path = str(path)
    # 用前 1 秒探测是否可解码(快速失败);时长独立用 get_duration 精确计算。
    # 注意:get_duration(path=path) 不可用时的 fallback 不能复用只加载了 1 秒的
    # audio,否则几小时的长音频会被误判为 1 秒。fallback 重新无 duration 限制加载。
    audio, sample_rate = librosa.load(path, sr=None, mono=True, duration=1.0)
    if sample_rate <= 0 or len(audio) == 0:
        raise ValueError("音频中没有可解码的采样")
    get_duration = getattr(librosa, "get_duration", None)
    if get_duration is not None:
        duration = float(get_duration(path=path))
    else:
        # fallback(旧 librosa 无 get_duration):只解码到 max_duration+1 秒上限,
        # 避免几小时的长音频被全量解码进内存 OOM。若实际更长,下面 max 检查会拒绝。
        cap = (max_duration_sec + 1.0) if max_duration_sec else 1.0
        full_audio, _ = librosa.load(path, sr=None, mono=True, duration=cap)
        duration = len(full_audio) / float(sample_rate)
    if duration <= 0:
        raise ValueError("音频时长无效")
    if max_duration_sec and duration > max_duration_sec:
        raise ValueError(f"音频超过 {max_duration_sec:g} 秒限制")
    return duration


def split_audio_into_chunks(
    audio: np.ndarray,
    sample_rate: int,
    chunk_duration: float,
    overlap_duration: float,
) -> list[tuple[np.ndarray, float, float]]:
    if sample_rate <= 0 or chunk_duration <= 0 or overlap_duration < 0:
        raise ValueError("音频分段配置无效")
    chunk_samples = int(chunk_duration * sample_rate)
    step_samples = chunk_samples - int(overlap_duration * sample_rate)
    if chunk_samples <= 0 or step_samples <= 0:
        raise ValueError("分段重叠必须小于分段时长")
    chunks = []
    start = 0
    while start < len(audio):
        end = min(start + chunk_samples, len(audio))
        chunk = audio[start:end]
        if len(chunk) >= sample_rate * 0.5:
            chunks.append((chunk, start / sample_rate, end / sample_rate))
        start += step_samples
        if len(audio) - start < sample_rate * 0.5:
            break
    return chunks
