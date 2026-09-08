"""Face library and meeting attendance API."""
from __future__ import annotations

import asyncio
import hashlib
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.config import config
from app.repositories.people import DuplicateFaceSampleError
from app.services.face_attendance import (
    face_status,
    get_face_service,
    normalize_face_model,
    reset_face_service,
)
from app.services.model_config import apply_model_settings_to_runtime, update_model_section
from app.api.settings import _write_env_values


router = APIRouter(tags=["face"])

ALLOWED_FACE_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
FACE_IMAGE_MEDIA_TYPES = {
    ".bmp": "image/bmp",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


class FaceSettingsRequest(BaseModel):
    provider: str = Field("insightface", min_length=1, max_length=40)
    endpoint: str | None = Field(None, max_length=500)
    api_key: str | None = Field(None, max_length=500)
    model: str = Field("buffalo_l", min_length=1, max_length=80)
    device: str = Field("auto", pattern="^(auto|cpu|cuda|directml)$")
    enabled: bool = True
    match_threshold: float = Field(0.55, ge=0.0, le=1.0)
    match_margin: float = Field(0.08, ge=0.0, le=1.0)
    frame_interval_sec: int = Field(5, ge=2, le=60)


class ManualAttendanceRequest(BaseModel):
    person_id: str
    present: bool = True


async def _read_upload(file: UploadFile, *, max_bytes: int) -> bytes:
    size = 0
    chunks: list[bytes] = []
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        size += len(chunk)
        if size > max_bytes:
            limit_mb = max_bytes // (1024 * 1024)
            raise HTTPException(status_code=413, detail=f"图片超过 {limit_mb}MB 限制")
        chunks.append(chunk)
    if size == 0:
        raise HTTPException(status_code=400, detail="图片为空")
    return b"".join(chunks)


def _canonical_uuid(value: str, *, label: str) -> str:
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError):
        raise HTTPException(status_code=404, detail=f"{label}不存在") from None
    canonical = str(parsed)
    if canonical != value:
        raise HTTPException(status_code=404, detail=f"{label}不存在")
    return canonical


def _face_image_media_type(path: Path) -> str:
    return FACE_IMAGE_MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream")


@router.get("/v1/face/settings")
async def get_face_settings():
    return face_status()


@router.put("/v1/face/settings")
async def update_face_settings(body: FaceSettingsRequest):
    try:
        update_model_section(
            "face",
            {
                "provider": body.provider,
                "endpoint": body.endpoint,
                "api_key": body.api_key,
                "model": normalize_face_model(body.model),
                "device": body.device,
                "enabled": body.enabled,
                "match_threshold": body.match_threshold,
                "match_margin": body.match_margin,
                "frame_interval_sec": body.frame_interval_sec,
            },
        )
        _write_env_values(
            {
                "FACE_RECOGNITION_ENABLED": str(body.enabled).lower(),
                "FACE_PROVIDER": body.provider,
                "FACE_ENDPOINT": body.endpoint or "",
                "FACE_MODEL": normalize_face_model(body.model),
                "FACE_DEVICE": body.device,
                "FACE_MATCH_THRESHOLD": str(body.match_threshold),
                "FACE_MATCH_MARGIN": str(body.match_margin),
                "FACE_FRAME_INTERVAL_SEC": str(body.frame_interval_sec),
            }
        )
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"无法保存人脸模型配置: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    apply_model_settings_to_runtime()
    reset_face_service()
    return face_status()


@router.post("/v1/face/test")
async def test_face_settings():
    if not config.face.enabled:
        raise HTTPException(status_code=400, detail="人脸识别未启用")
    reset_face_service()
    await asyncio.to_thread(get_face_service().load)
    return face_status(load=False)


@router.post("/v1/people/{person_id}/face-samples", status_code=201)
async def add_face_sample(
    person_id: str,
    request: Request,
    file: UploadFile = File(...),
):
    if not config.face.enabled:
        raise HTTPException(status_code=503, detail="人脸识别未启用")
    canonical_person_id = _canonical_uuid(person_id, label="人物")
    if request.app.state.people_repo.get(canonical_person_id) is None:
        raise HTTPException(status_code=404, detail="人物不存在")
    extension = Path(file.filename or "face.jpg").suffix.lower()
    if extension not in ALLOWED_FACE_IMAGE_EXTENSIONS:
        raise HTTPException(status_code=400, detail="不支持的人脸照片格式")

    data = await _read_upload(file, max_bytes=config.face.upload_max_file_size)
    image_sha256 = hashlib.sha256(data).hexdigest()
    if request.app.state.people_repo.has_face_sample_hash(canonical_person_id, image_sha256):
        raise HTTPException(status_code=409, detail="该人物已注册相同的人脸照片")

    face_root = (Path(config.storage.media_dir).resolve() / "faces").resolve()
    face_dir = (face_root / uuid.UUID(canonical_person_id).hex).resolve()
    try:
        face_dir.relative_to(face_root)
    except ValueError:
        raise HTTPException(status_code=400, detail="人脸样本路径无效") from None
    face_dir.mkdir(parents=True, exist_ok=True)
    target = face_dir / f"{uuid.uuid4().hex}{extension}"
    try:
        enrollment = await asyncio.to_thread(get_face_service().enroll_image_bytes, data)
        target.write_bytes(data)
        try:
            sample_id = request.app.state.people_repo.add_face_sample(
                canonical_person_id,
                image_path=str(target),
                embedding=enrollment.embedding.astype("<f4", copy=False).tobytes(),
                embedding_dim=enrollment.embedding_dim,
                model_name=enrollment.model_name,
                detection_score=enrollment.detection_score,
                image_sha256=image_sha256,
            )
        except DuplicateFaceSampleError:
            raise HTTPException(status_code=409, detail="该人物已注册相同的人脸照片") from None
        return {
            "id": sample_id,
            "person_id": canonical_person_id,
            "embedding_dim": enrollment.embedding_dim,
            "model_name": enrollment.model_name,
            "detection_score": enrollment.detection_score,
            "face_count": enrollment.face_count,
        }
    except HTTPException:
        target.unlink(missing_ok=True)
        raise
    except ValueError as exc:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"人脸样本注册失败: {exc}") from exc
    finally:
        await file.close()


@router.get("/v1/people/{person_id}/face-samples/{sample_id}/image")
def get_face_sample_image(person_id: str, sample_id: str, request: Request):
    sample = request.app.state.people_repo.get_face_sample(person_id, sample_id)
    if sample is None:
        raise HTTPException(status_code=404, detail="人脸样本不存在")
    path = Path(sample["image_path"])
    if not path.is_file():
        raise HTTPException(status_code=404, detail="人脸样本文件不存在")
    return FileResponse(
        path,
        media_type=_face_image_media_type(path),
        filename=f"face-sample{path.suffix.lower()}",
        content_disposition_type="inline",
    )


@router.delete("/v1/people/{person_id}/face-samples/{sample_id}")
def delete_face_sample(person_id: str, sample_id: str, request: Request):
    if not request.app.state.people_repo.delete_face_sample(person_id, sample_id):
        raise HTTPException(status_code=404, detail="人脸样本不存在")
    return {"message": "人脸样本已删除"}


@router.get("/v1/meetings/{meeting_id}/attendance")
def list_meeting_attendance(meeting_id: str, request: Request):
    if request.app.state.meeting_repo.get(meeting_id) is None:
        raise HTTPException(status_code=404, detail="会议不存在")
    items = request.app.state.meeting_repo.list_attendance(meeting_id)
    return {"total": len(items), "items": items}


@router.post("/v1/meetings/{meeting_id}/attendance/manual")
def set_manual_attendance(
    meeting_id: str,
    body: ManualAttendanceRequest,
    request: Request,
):
    if request.app.state.meeting_repo.get(meeting_id) is None:
        raise HTTPException(status_code=404, detail="会议不存在")
    if request.app.state.people_repo.get(body.person_id) is None:
        raise HTTPException(status_code=404, detail="人物不存在")
    if not body.present:
        request.app.state.meeting_repo.delete_attendance(meeting_id, body.person_id)
        return {"items": request.app.state.meeting_repo.list_attendance(meeting_id)}
    item = request.app.state.meeting_repo.upsert_attendance(
        meeting_id,
        body.person_id,
        source="manual",
        confidence=1.0,
    )
    return {"item": item, "items": request.app.state.meeting_repo.list_attendance(meeting_id)}


@router.post("/v1/meetings/{meeting_id}/attendance/face-frame")
async def recognize_attendance_frame(
    meeting_id: str,
    request: Request,
    timestamp_sec: float | None = Form(None),
    file: UploadFile = File(...),
):
    if not config.face.enabled:
        raise HTTPException(status_code=503, detail="人脸识别未启用")
    if request.app.state.meeting_repo.get(meeting_id) is None:
        raise HTTPException(status_code=404, detail="会议不存在")
    data = await _read_upload(file, max_bytes=config.face.upload_max_file_size)
    try:
        result = await asyncio.to_thread(
            get_face_service().recognize_frame_bytes,
            data,
            request.app.state.people_repo,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"人脸识别失败: {exc}") from exc
    finally:
        await file.close()

    checked_in: list[dict[str, Any]] = []
    seen_people: set[str] = set()
    for detection in result["detections"]:
        person_id = detection.get("person_id")
        if not detection.get("matched") or not person_id or person_id in seen_people:
            continue
        seen_people.add(person_id)
        checked_in.append(
            request.app.state.meeting_repo.upsert_attendance(
                meeting_id,
                person_id,
                source="face",
                confidence=detection.get("confidence"),
                timestamp_sec=timestamp_sec,
            )
        )

    return {
        **result,
        "checked_in": checked_in,
        "attendance": request.app.state.meeting_repo.list_attendance(meeting_id),
    }
