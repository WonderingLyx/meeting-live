"""Meeting list, detail, correction, audio, and deletion API."""
import asyncio
import hashlib
import logging
import uuid
import wave
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field, field_validator

from app.config import config
from app.repositories.meetings import SpeakerNotFoundError
from app.services.audio_files import (
    ALLOWED_AUDIO_EXTENSIONS,
    UPLOAD_TRANSCODE_REQUIRED_EXTENSIONS,
    UploadTooLargeError,
    audio_transcode_available,
    persist_upload,
    transcode_audio_to_wav,
    validate_audio_file,
)
from app.services.audio_enhancement import enhance_audio_for_models
from app.services.exporter import export_meeting as render_meeting_export


logger = logging.getLogger("Matrix_Meetings")

router = APIRouter(prefix="/v1/meetings", tags=["meetings"])


AUDIO_MEDIA_TYPES = {
    ".aac": "audio/aac",
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
    ".mp3": "audio/mpeg",
    ".mp4": "audio/mp4",
    ".ogg": "audio/ogg",
    ".wav": "audio/wav",
    ".webm": "audio/webm",
}


def _audio_media_type(path: Path) -> str:
    return AUDIO_MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream")


def _expected_upload_size(request: Request) -> int | None:
    raw = request.headers.get("x-upload-size")
    if not raw:
        return None
    try:
        size = int(raw)
    except (TypeError, ValueError):
        logger.warning("[upload] 忽略无效 X-Upload-Size=%r", raw)
        return None
    return size if size >= 0 else None


def _enhancement_cache_signature() -> str:
    audio = config.audio
    names = (
        "enhancement_enabled",
        "enhancement_high_pass_hz",
        "enhancement_low_pass_hz",
        "enhancement_noise_reduction",
        "enhancement_noise_floor",
        "enhancement_target_rms",
        "enhancement_max_gain",
        "enhancement_max_block_seconds",
    )
    return "|".join(f"{name}={getattr(audio, name, '')}" for name in names)


def _browser_playback_cache_path(source: Path, variant: str = "browser") -> Path:
    stat = source.stat()
    key_parts = [variant, str(source.resolve()), str(stat.st_size), str(stat.st_mtime_ns)]
    if variant == "enhanced":
        key_parts.append(_enhancement_cache_signature())
    key = "|".join(key_parts)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
    suffix = ".wav" if variant == "browser" else f".{variant}.wav"
    return Path(config.storage.media_dir).resolve() / "playback" / f"{digest}{suffix}"


def _is_browser_pcm_wav(path: Path) -> bool:
    if path.suffix.lower() != ".wav":
        return False
    try:
        with wave.open(str(path), "rb") as wf:
            return (
                wf.getcomptype() == "NONE"
                and wf.getnchannels() in (1, 2)
                and wf.getsampwidth() == 2
                and wf.getframerate() > 0
                and wf.getnframes() > 0
            )
    except (EOFError, OSError, wave.Error):
        return False


def _recover_pcm_wav_payload(path: Path) -> tuple[object, int]:
    """Recover PCM samples from WAV files whose headers under-report data size."""
    import numpy as np

    raw = path.read_bytes()
    if len(raw) < 44 or raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE file")

    fmt: dict[str, int] | None = None
    data_start = 0
    data_size = 0
    offset = 12
    while offset + 8 <= len(raw):
        chunk_id = raw[offset : offset + 4]
        chunk_size = int.from_bytes(raw[offset + 4 : offset + 8], "little")
        start = offset + 8
        if start > len(raw):
            break
        if chunk_id == b"fmt " and start + 16 <= len(raw):
            fmt = {
                "format": int.from_bytes(raw[start : start + 2], "little"),
                "channels": int.from_bytes(raw[start + 2 : start + 4], "little"),
                "sample_rate": int.from_bytes(raw[start + 4 : start + 8], "little"),
                "block_align": int.from_bytes(raw[start + 12 : start + 14], "little"),
                "bits": int.from_bytes(raw[start + 14 : start + 16], "little"),
            }
        elif chunk_id == b"data":
            data_start = start
            data_size = len(raw) - start
            break
        next_offset = start + chunk_size + (chunk_size % 2)
        if next_offset <= offset:
            break
        offset = next_offset

    if not fmt or not data_start or data_size <= 0:
        raise ValueError("missing WAV fmt/data payload")
    if fmt["format"] != 1 or fmt["bits"] != 16:
        raise ValueError("only PCM16 WAV payload recovery is supported")
    channels = fmt["channels"]
    sample_rate = fmt["sample_rate"] or config.audio.sample_rate
    block_align = fmt["block_align"] or channels * 2
    usable = (data_size // block_align) * block_align
    if channels <= 0 or usable <= 0:
        raise ValueError("empty WAV PCM payload")

    pcm = np.frombuffer(raw[data_start : data_start + usable], dtype="<i2")
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1)
    audio = pcm.astype(np.float32) / 32768.0
    if audio.size == 0:
        raise ValueError("empty recovered WAV audio")
    return audio, int(sample_rate)


def _write_pcm16_wav(path: Path, audio: object, sample_rate: int) -> None:
    import numpy as np

    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if samples.size == 0 or sample_rate <= 0:
        raise ValueError("decoded audio is empty")
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm.tobytes())


def _decode_audio_for_playback(path: Path) -> tuple[object, int]:
    import librosa
    import numpy as np

    if path.suffix.lower() == ".wav":
        try:
            audio, sample_rate = _recover_pcm_wav_payload(path)
            return np.asarray(audio, dtype=np.float32), int(sample_rate)
        except Exception as exc:
            logger.warning("[audio] %s PCM WAV 恢复失败，回退到常规解码: %s", path, exc)

    audio, sample_rate = librosa.load(str(path), sr=config.audio.sample_rate, mono=True)
    return np.asarray(audio, dtype=np.float32), int(sample_rate)


def _write_cached_playback_wav(target: Path, audio: object, sample_rate: int) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.stem}.{uuid.uuid4().hex}.tmp.wav")
    try:
        _write_pcm16_wav(tmp, audio, int(sample_rate))
        tmp.replace(target)
    finally:
        tmp.unlink(missing_ok=True)
    return target


def _browser_playback_audio(path: Path) -> Path:
    """Return browser-decodable PCM WAV, converting and caching if needed."""
    target = _browser_playback_cache_path(path)
    if target.is_file() and _is_browser_pcm_wav(target):
        return target
    audio, sample_rate = _decode_audio_for_playback(path)
    return _write_cached_playback_wav(target, audio, sample_rate)


def _enhanced_playback_audio(path: Path) -> Path:
    """Return browser-decodable enhanced PCM WAV for meeting playback."""
    target = _browser_playback_cache_path(path, variant="enhanced")
    if target.is_file() and _is_browser_pcm_wav(target):
        return target
    audio, sample_rate = _decode_audio_for_playback(path)
    enhanced = enhance_audio_for_models(audio, int(sample_rate), profile="playback")
    return _write_cached_playback_wav(target, enhanced, sample_rate)


def _playable_live_audio(path: Path) -> tuple[bool, str]:
    """A live WAV is not playable until wave.close() has written the header."""
    if path.suffix.lower() != ".wav":
        return (path.stat().st_size > 0, "audio file is still empty")
    try:
        with wave.open(str(path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
    except (EOFError, OSError, wave.Error) as exc:
        return False, f"audio file is not finalized yet: {exc}"
    if frames <= 0 or rate <= 0:
        return False, "audio file is not finalized yet"
    return True, ""


class MeetingUpdate(BaseModel):
    title: str = Field(min_length=1, max_length=200)

    @field_validator("title")
    @classmethod
    def _strip_title(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("title 不能为空")
        return v


class SegmentAssignment(BaseModel):
    segment_ids: list[int] = Field(min_length=1, max_length=500)
    meeting_speaker_id: str | None = None


class SpeakerConfirmation(BaseModel):
    person_id: str | None = None


class SegmentTextUpdate(BaseModel):
    text: str = Field(min_length=1, max_length=20_000)

    @field_validator("text")
    @classmethod
    def _strip_text(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("text 不能为空")
        return v


class NoteUpdate(BaseModel):
    content: str = Field(min_length=1, max_length=100_000)

    @field_validator("content")
    @classmethod
    def _strip_content(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("content 不能为空")
        return v


class SpeakerMergeRequest(BaseModel):
    target_speaker_id: str
    source_speaker_ids: list[str] = Field(min_length=1, max_length=50)


class SpeakerSplitRequest(BaseModel):
    source_speaker_id: str
    segment_ids: list[int] = Field(min_length=1, max_length=500)


def _require_mutable_meeting(request: Request, meeting_id: str) -> dict:
    meeting = request.app.state.meeting_repo.get(meeting_id)
    if meeting is None:
        raise HTTPException(status_code=404, detail="会议不存在")
    if meeting.get("status") == "processing":
        raise HTTPException(
            status_code=409,
            detail="后台精修中，草稿暂时只读；完成后再修改",
        )
    return meeting


@router.get("")
def list_meetings(
    request: Request,
    q: str | None = Query(None, max_length=100),
    status: str | None = Query(None, pattern="^(draft|processing|ready|failed)$"),
    source: str | None = Query(None, pattern="^(live|upload)$"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
):
    total, items = request.app.state.meeting_repo.list(
        q=q,
        status=status,
        source=source,
        limit=page_size,
        offset=(page - 1) * page_size,
    )
    return {"total": total, "items": items}


@router.post("/upload", status_code=202)
async def create_upload_meeting(
    request: Request,
    file: UploadFile = File(...),
    mode: str = Query("meeting", pattern="^(quick|meeting)$"),
):
    """Persist audio and immediately return a durable meeting/job pair."""
    filename = Path(file.filename or "recording.wav").name
    safe_filename = filename.replace("\r", "\\r").replace("\n", "\\n")[:200]
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_AUDIO_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_AUDIO_EXTENSIONS))
        logger.warning(
            "[upload] 拒绝上传: 不支持的音频格式 filename=%r extension=%r allowed=%s",
            safe_filename,
            extension or "<none>",
            allowed,
        )
        raise HTTPException(
            status_code=400,
            detail=f"不支持的音频格式: {extension or '无扩展名'}。支持: {allowed}",
        )
    media_dir = Path(config.storage.media_dir).resolve()
    media_dir.mkdir(parents=True, exist_ok=True)
    target = media_dir / f"{uuid.uuid4().hex}{extension}"
    meeting_id = None
    max_bytes = config.audio.upload_max_file_size
    expected_size = _expected_upload_size(request)
    try:
        try:
            size = await persist_upload(file, target, max_bytes=max_bytes)
        except UploadTooLargeError:
            limit_mb = max_bytes // (1024 * 1024)
            logger.warning(
                "[upload] 拒绝上传: 文件超过大小限制 filename=%r limit_mb=%d",
                safe_filename,
                limit_mb,
            )
            raise HTTPException(
                status_code=400, detail=f"文件超过 {limit_mb}MB 限制"
            ) from None
        if size == 0:
            logger.warning("[upload] 拒绝上传: 上传文件为空 filename=%r", safe_filename)
            raise HTTPException(status_code=400, detail="上传文件为空")
        if expected_size is not None and expected_size != size:
            logger.warning(
                "[upload] 拒绝上传: 文件大小不一致 filename=%r browser_size=%d saved_size=%d",
                safe_filename,
                expected_size,
                size,
            )
            raise HTTPException(
                status_code=400,
                detail=f"文件上传不完整: 浏览器报告 {expected_size} 字节,后端收到 {size} 字节",
            )
        if (
            extension != ".wav"
            and (extension in UPLOAD_TRANSCODE_REQUIRED_EXTENSIONS or audio_transcode_available())
        ):
            wav_target = target.with_suffix(".wav")
            try:
                await asyncio.to_thread(
                    transcode_audio_to_wav,
                    target,
                    wav_target,
                    sample_rate=config.audio.sample_rate,
                    timeout_sec=max(600, config.audio.upload_max_duration * 2),
                )
            except ValueError as exc:
                reason = str(exc) or "音频转码失败"
                logger.warning(
                    "[upload] 音频转码失败 filename=%r size=%d path=%s: %s",
                    safe_filename,
                    size,
                    target,
                    reason,
                )
                raise HTTPException(status_code=400, detail=reason) from None
            target.unlink(missing_ok=True)
            target = wav_target
            logger.info(
                "[upload] 已转为内部 WAV filename=%r source_ext=%s wav_path=%s",
                safe_filename,
                extension,
                target,
            )
        try:
            await asyncio.to_thread(
                validate_audio_file,
                target,
                max_duration_sec=config.audio.upload_max_duration,
            )
        except ValueError as exc:
            reason = str(exc) or "音频校验失败"
            logger.warning(
                "[upload] 音频校验失败 filename=%r size=%d path=%s: %s",
                safe_filename,
                size,
                target,
                reason,
            )
            raise HTTPException(status_code=400, detail=reason) from None
        except Exception as exc:
            logger.warning(
                "[upload] 音频解码失败 filename=%r size=%d path=%s: %s",
                safe_filename,
                size,
                target,
                exc,
            )
            allowed = ", ".join(sorted(ALLOWED_AUDIO_EXTENSIONS))
            raise HTTPException(
                status_code=400,
                detail=f"音频无法解码,请确认已安装 FFmpeg,并使用支持的音频类型: {allowed}",
            ) from None
        meeting_id, job_id = request.app.state.meeting_repo.create_with_job(
            source="upload",
            title=Path(filename).stem or "未命名会议",
            original_filename=filename,
            audio_path=str(target),
            processing_mode=mode,
            status="processing",
        )
        runner = getattr(request.app.state, "job_runner", None)
        if runner is not None:
            runner.notify()
        return {
            "meeting_id": meeting_id,
            "job_id": job_id,
            "status": "queued",
        }
    except Exception:
        if meeting_id is not None:
            request.app.state.meeting_repo.delete(meeting_id, delete_audio=False)
        target.unlink(missing_ok=True)
        raise
    finally:
        await file.close()


@router.get("/search")
def search_meetings(
    request: Request,
    q: str = Query(..., min_length=1, max_length=100),
    limit: int = Query(50, ge=1, le=200),
):
    hits = request.app.state.meeting_repo.search(q, limit=limit)
    return {"query": q, "total": len(hits), "hits": hits}


@router.get("/{meeting_id}")
def get_meeting(meeting_id: str, request: Request):
    detail = request.app.state.meeting_repo.detail(meeting_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="会议不存在")
    detail["processing_job"] = request.app.state.job_repo.latest_for_meeting(
        meeting_id
    )
    return detail


@router.patch("/{meeting_id}")
def update_meeting(meeting_id: str, body: MeetingUpdate, request: Request):
    _require_mutable_meeting(request, meeting_id)
    if not request.app.state.meeting_repo.update(meeting_id, title=body.title.strip()):
        raise HTTPException(status_code=404, detail="会议不存在")
    return request.app.state.meeting_repo.get(meeting_id)


@router.delete("/{meeting_id}")
def delete_meeting(meeting_id: str, request: Request):
    result = request.app.state.meeting_repo.delete_if_inactive(meeting_id)
    if result == "active":
        raise HTTPException(status_code=409, detail="会议仍在处理中，请先取消任务")
    if result == "not_found":
        raise HTTPException(status_code=404, detail="会议不存在")
    return {"message": "会议及本地音频已删除"}


@router.post("/{meeting_id}/reprocess", status_code=202)
def reprocess_meeting(meeting_id: str, request: Request):
    meeting = request.app.state.meeting_repo.get(meeting_id)
    if meeting is None:
        raise HTTPException(status_code=404, detail="会议不存在")
    if meeting.get("status") == "processing":
        raise HTTPException(
            status_code=409, detail="会议正在实时录制或处理中,无法启动精修"
        )
    path = Path(meeting.get("audio_path") or "")
    if not path.is_file():
        raise HTTPException(status_code=409, detail="会议没有可重新处理的本地音频")
    # capped 会话:WAV 被写盘上限截断(只含前 upload_max_duration),但转写段
    # 覆盖完整时长。reprocess 会用截断 WAV 重转写 + replace_generated_transcript
    # 删掉完整转写 → 数据丢失。检测 truncated 标记拒绝。
    manifest = meeting.get("processing_manifest")
    if isinstance(manifest, dict) and manifest.get("truncated"):
        raise HTTPException(
            status_code=409,
            detail="实时录音超过时长上限已截断,重新精修会丢失超限转写,已拒绝",
        )
    # 双失败兜底:truncated manifest 写失败时 audio_path 已被置空;但若
    # 置空也失败(recover 前仍指向截断 WAV),这里按 WAV 实际时长与
    # duration_sec 不一致判定为截断,拒绝 reprocess。用 wave 读头(轻量,不依赖 librosa)。
    if meeting.get("source") == "live" and meeting.get("duration_sec"):
        try:
            import wave
            with wave.open(str(path), "rb") as _wf:
                wav_dur = _wf.getnframes() / float(_wf.getframerate() or 16000)
            if wav_dur + 1.0 < float(meeting["duration_sec"]):
                raise HTTPException(
                    status_code=409,
                    detail="实时音频文件已截断,重新精修会丢失转写,已拒绝",
                )
        except HTTPException:
            raise
        except Exception:
            safe_meeting_id = meeting_id.replace("\r", r"\r").replace("\n", r"\n")[:128]
            logger.debug("[reprocess] %s WAV 时长探测失败,跳过截断校验", safe_meeting_id)
    try:
        job_id, created = request.app.state.job_repo.enqueue_refinement(meeting_id)
    except FileNotFoundError:
        raise HTTPException(status_code=409, detail="会议音频不可用") from None
    except RuntimeError as exc:
        safe_meeting_id = meeting_id.replace("\r", r"\r").replace("\n", r"\n")[:128]
        safe_error = str(exc).replace("\r", r"\r").replace("\n", r"\n")[:500]
        logger.warning("[reprocess] %s 拒绝精修: %s", safe_meeting_id, safe_error)
        raise HTTPException(
            status_code=409, detail="会议正在实时录制,无法启动精修"
        ) from exc
    if not created:
        raise HTTPException(status_code=409, detail="会议已经在处理中")
    runner = getattr(request.app.state, "job_runner", None)
    if runner is not None:
        runner.notify()
    return {"meeting_id": meeting_id, "job_id": job_id, "status": "queued"}


@router.get("/{meeting_id}/audio")
def meeting_audio(
    meeting_id: str,
    request: Request,
    playback: str = Query("original", pattern="^(original|browser|enhanced)$"),
):
    meeting = request.app.state.meeting_repo.get(meeting_id)
    if meeting is None:
        raise HTTPException(status_code=404, detail="会议不存在")
    path = Path(meeting.get("audio_path") or "")
    if not path.is_file():
        raise HTTPException(status_code=404, detail="会议音频不可用")
    if meeting.get("source") == "live":
        if meeting.get("status") == "processing":
            raise HTTPException(
                status_code=409,
                detail="live recording audio is still being finalized",
            )
        playable, reason = _playable_live_audio(path)
        if not playable:
            raise HTTPException(status_code=409, detail=reason)
    serve_path = path
    filename = meeting.get("original_filename") or path.name
    media_type = _audio_media_type(path)
    if playback in {"browser", "enhanced"}:
        try:
            if playback == "enhanced":
                serve_path = _enhanced_playback_audio(path)
                filename = f"{path.stem}.enhanced.wav"
            else:
                serve_path = _browser_playback_audio(path)
                filename = f"{path.stem}.playback.wav"
            media_type = "audio/wav"
        except Exception as exc:
            logger.warning("[audio] %s 转换 %s 播放音频失败: %s", meeting_id, playback, exc)
            if playback == "enhanced":
                detail = "会议音频无法生成降噪播放版本，请确认 FFmpeg/libsndfile 可用"
            else:
                detail = "会议音频无法转换成浏览器可播放格式，请确认 FFmpeg/libsndfile 可用"
            raise HTTPException(
                status_code=409,
                detail=detail,
            ) from None
    return FileResponse(
        serve_path,
        filename=filename,
        media_type=media_type,
        content_disposition_type="inline",
        headers={"Cache-Control": "no-store", "Accept-Ranges": "bytes"},
    )


@router.get("/{meeting_id}/export")
def export_meeting(
    meeting_id: str,
    request: Request,
    format: str = Query("markdown", pattern="^(markdown|json|srt|vtt)$"),
):
    detail = request.app.state.meeting_repo.detail(meeting_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="会议不存在")
    title = detail["meeting"]["title"]
    safe_name = "".join(c for c in title if c.isalnum() or c in "-_ ").strip() or meeting_id
    content = render_meeting_export(detail, format)
    media_type, suffix = {
        "json": ("application/json", "json"),
        "srt": ("application/x-subrip", "srt"),
        "vtt": ("text/vtt", "vtt"),
        "markdown": ("text/markdown", "md"),
    }[format]
    return Response(
        content=content,
        media_type=media_type,
        headers={
            "Content-Disposition": (
                f'attachment; filename="meeting.{suffix}"; '
                f"filename*=UTF-8''{quote(f'{safe_name}.{suffix}')}"
            )
        },
    )


@router.post("/{meeting_id}/notes/{note_type}")
async def generate_meeting_note(
    meeting_id: str,
    note_type: str,
    request: Request,
):
    if note_type not in {"summary", "minutes", "actions"}:
        raise HTTPException(status_code=400, detail="不支持的会议产出类型")
    _require_mutable_meeting(request, meeting_id)
    detail = request.app.state.meeting_repo.detail(meeting_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="会议不存在")
    segments = detail["segments"]
    if not segments:
        raise HTTPException(status_code=409, detail="会议还没有可用文字")
    from app.api.llm import _get_gateway

    gateway = _get_gateway(request)
    operation = {
        "summary": "summarize",
        "actions": "action_items",
        "minutes": "minutes",
    }[note_type]
    # summarize:不显式传 max_words,由 _generate 按会议总时长自适应
    # (短~120/中~200/长~300/超长~400 字)。模板用 {max_words} 占位符渲染。
    content, source = await gateway._generate(operation, segments)
    note = request.app.state.meeting_repo.save_note(
        meeting_id, note_type, content, source
    )
    return note


@router.put("/{meeting_id}/notes/{note_type}")
def update_meeting_note(
    meeting_id: str, note_type: str, body: NoteUpdate, request: Request
):
    if note_type not in {"summary", "minutes", "actions"}:
        raise HTTPException(status_code=400, detail="不支持的会议产出类型")
    _require_mutable_meeting(request, meeting_id)
    return request.app.state.meeting_repo.save_note(
        meeting_id, note_type, body.content.strip(), "manual"
    )


@router.patch("/{meeting_id}/segments/speaker")
def assign_segment_speaker(
    meeting_id: str,
    body: SegmentAssignment,
    request: Request,
):
    _require_mutable_meeting(request, meeting_id)
    try:
        updated = request.app.state.meeting_repo.assign_segments(
            meeting_id, body.segment_ids, body.meeting_speaker_id
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"updated": updated}


@router.patch("/{meeting_id}/segments/{segment_id}")
def update_segment_text(
    meeting_id: str, segment_id: int, body: SegmentTextUpdate, request: Request
):
    _require_mutable_meeting(request, meeting_id)
    if not request.app.state.meeting_repo.update_segment_text(
        meeting_id, segment_id, body.text.strip()
    ):
        raise HTTPException(status_code=404, detail="文字片段不存在")
    return {"message": "文字已保存"}


@router.patch("/{meeting_id}/speakers/{speaker_id}/person")
def confirm_meeting_speaker(
    meeting_id: str,
    speaker_id: str,
    body: SpeakerConfirmation,
    request: Request,
):
    _require_mutable_meeting(request, meeting_id)
    try:
        updated = request.app.state.meeting_repo.confirm_speaker(
            meeting_id, speaker_id, body.person_id
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="人物不存在") from exc
    if not updated:
        raise HTTPException(status_code=404, detail="会议说话人不存在")
    return {"message": "说话人身份已确认"}


@router.post("/{meeting_id}/speakers/merge")
def merge_meeting_speakers(
    meeting_id: str,
    body: SpeakerMergeRequest,
    request: Request,
):
    """合并:把多个 source speaker 的 segments 全部改指到 target,然后删 source。"""
    _require_mutable_meeting(request, meeting_id)
    try:
        migrated = request.app.state.meeting_repo.merge_speakers(
            meeting_id, body.target_speaker_id, body.source_speaker_ids
        )
    except SpeakerNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "target_speaker_id": body.target_speaker_id,
        "merged_source_ids": body.source_speaker_ids,
        "segments_updated": migrated,
    }


@router.post("/{meeting_id}/speakers/split")
def split_meeting_speaker(
    meeting_id: str,
    body: SpeakerSplitRequest,
    request: Request,
):
    """拆分:给选中的 segments 创建新的匿名 speaker,脱离 source。"""
    _require_mutable_meeting(request, meeting_id)
    try:
        new_speaker_id = request.app.state.meeting_repo.split_speaker(
            meeting_id, body.source_speaker_id, body.segment_ids
        )
    except SpeakerNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "new_speaker_id": new_speaker_id,
        "segments_updated": len(body.segment_ids),
    }
