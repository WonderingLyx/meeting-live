"""Local face recognition for meeting attendance.

The engine is intentionally lazy: importing the app should not import
InsightFace or ONNXRuntime, because those packages are optional during tests
and can download model packs on first load.
"""
from __future__ import annotations

import importlib.metadata
import logging
import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.config import config


logger = logging.getLogger("Matrix_Face")


SUPPORTED_FACE_MODELS: dict[str, dict[str, Any]] = {
    "buffalo_l": {
        "type": "buffalo_l",
        "name": "InsightFace Buffalo-L",
        "provider": "insightface",
        "model": "buffalo_l",
        "embedding_dim": 512,
        "description": "InsightFace 官方通用人脸检测 + ArcFace 识别模型包，精度优先。",
        "description_en": "InsightFace detection + ArcFace recognition pack, accuracy first.",
        "recommended_for": ["attendance", "photo-library", "accuracy"],
    },
    "buffalo_s": {
        "type": "buffalo_s",
        "name": "InsightFace Buffalo-S",
        "provider": "insightface",
        "model": "buffalo_s",
        "embedding_dim": 512,
        "description": "更轻量的 InsightFace 模型包，适合 CPU 或低功耗机器快速试用。",
        "description_en": "A smaller InsightFace pack for CPU or low-power machines.",
        "recommended_for": ["cpu", "quick-test"],
    },
    "buffalo_m": {
        "type": "buffalo_m",
        "name": "InsightFace Buffalo-M",
        "provider": "insightface",
        "model": "buffalo_m",
        "embedding_dim": 512,
        "description": "体积和速度介于 Buffalo-S 与 Buffalo-L 之间。",
        "description_en": "A middle option between Buffalo-S and Buffalo-L.",
        "recommended_for": ["balanced"],
    },
}


def normalize_face_model(value: str | None) -> str:
    raw = (value or "buffalo_l").strip().lower()
    if raw.startswith("insightface:"):
        raw = raw.split(":", 1)[1]
    if raw.startswith("insightface_"):
        raw = raw.removeprefix("insightface_")
    return raw or "buffalo_l"


def face_model_id(model: str | None = None) -> str:
    return f"insightface:{normalize_face_model(model or config.face.model)}"


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _available_ort_providers() -> list[str]:
    try:
        import onnxruntime as ort

        return list(ort.get_available_providers())
    except Exception:
        return []


def _opencv_status() -> tuple[str | None, str | None]:
    try:
        import cv2

        version = (
            _package_version("opencv-python-headless")
            or _package_version("opencv-python")
            or getattr(cv2, "__version__", None)
        )
        return version, None
    except Exception as exc:
        return None, str(exc)


def face_dependency_status() -> dict[str, Any]:
    ort_providers = _available_ort_providers()
    opencv_version, opencv_error = _opencv_status()
    return {
        "insightface_version": _package_version("insightface"),
        "onnxruntime_version": (
            _package_version("onnxruntime")
            or _package_version("onnxruntime-gpu")
            or _package_version("onnxruntime-directml")
        ),
        "onnxruntime_gpu_version": _package_version("onnxruntime-gpu"),
        "onnxruntime_directml_version": _package_version("onnxruntime-directml"),
        "opencv_version": opencv_version,
        "opencv_error": opencv_error,
        "available_providers": ort_providers,
        "dependency_available": bool(
            _package_version("insightface") and ort_providers and opencv_version
        ),
    }


@dataclass(frozen=True)
class FaceEnrollment:
    embedding: np.ndarray
    embedding_dim: int
    model_name: str
    detection_score: float
    face_count: int


class InsightFaceAttendanceService:
    """Thin wrapper around ``insightface.app.FaceAnalysis``."""

    def __init__(self) -> None:
        self.model = normalize_face_model(config.face.model)
        self.device = (config.face.device or "auto").strip().lower()
        self.match_threshold = float(config.face.match_threshold)
        self.match_margin = float(config.face.match_margin)
        self.root = (Path(config.models.models_dir).resolve() / "face" / "insightface")
        self._lock = threading.RLock()
        self._app = None
        self._loaded_provider = "unloaded"
        self._last_error: str | None = None

    @property
    def loaded(self) -> bool:
        return self._app is not None

    @property
    def loaded_provider(self) -> str:
        return self._loaded_provider

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def _resolve_providers(self) -> list[str]:
        available = _available_ort_providers()
        if not available:
            raise RuntimeError("ONNXRuntime 不可用，请先安装 onnxruntime/onnxruntime-gpu/onnxruntime-directml")

        preferred_by_device = {
            "cpu": ["CPUExecutionProvider"],
            "cuda": ["CUDAExecutionProvider"],
            "directml": ["DmlExecutionProvider"],
            "auto": ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"],
        }
        preferred = preferred_by_device.get(self.device)
        if preferred is None:
            raise RuntimeError("FACE_DEVICE 只支持 auto / cpu / cuda / directml")
        selected = [provider for provider in preferred if provider in available]
        if not selected:
            raise RuntimeError(
                f"当前 ONNXRuntime providers={available}，无法使用 FACE_DEVICE={self.device}"
            )
        if "CPUExecutionProvider" in available and "CPUExecutionProvider" not in selected:
            selected.append("CPUExecutionProvider")
        return selected

    def load(self) -> None:
        if self._app is not None:
            return
        with self._lock:
            if self._app is not None:
                return
            try:
                from insightface.app import FaceAnalysis

                providers = self._resolve_providers()
                self.root.mkdir(parents=True, exist_ok=True)
                kwargs = {
                    "name": self.model,
                    "root": str(self.root),
                    "providers": providers,
                    "allowed_modules": ["detection", "recognition"],
                }
                try:
                    app = FaceAnalysis(**kwargs)
                except TypeError:
                    kwargs.pop("allowed_modules", None)
                    app = FaceAnalysis(**kwargs)
                ctx_id = -1 if providers[0] == "CPUExecutionProvider" else 0
                app.prepare(ctx_id=ctx_id, det_size=(640, 640))
                self._app = app
                self._loaded_provider = providers[0]
                self._last_error = None
                logger.info(
                    "[FACE] InsightFace loaded model=%s provider=%s root=%s",
                    self.model,
                    self._loaded_provider,
                    self.root,
                )
            except Exception as exc:
                self._last_error = str(exc)
                self._loaded_provider = "unavailable"
                logger.error("[FACE] InsightFace 加载失败: %s", exc)
                raise

    @staticmethod
    def _decode_image(data: bytes):
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "OpenCV(cv2) 未安装。请重新运行一键安装脚本，或在当前项目虚拟环境中安装 opencv-python-headless。"
            ) from exc

        if not data:
            raise ValueError("图片为空")
        array = np.frombuffer(data, dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise ValueError("图片无法解码，请使用 JPG/PNG/WebP 等常见图片格式")
        return image

    @staticmethod
    def _normalized_embedding(face) -> np.ndarray:
        embedding = getattr(face, "normed_embedding", None)
        if embedding is None:
            embedding = getattr(face, "embedding", None)
        if embedding is None:
            raise ValueError("模型未返回人脸向量")
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if not vector.size or not np.isfinite(vector).all():
            raise ValueError("模型返回了无效人脸向量")
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-8 or not math.isfinite(norm):
            raise ValueError("模型返回了空人脸向量")
        return vector / norm

    @staticmethod
    def _face_box(face) -> list[float]:
        bbox = np.asarray(getattr(face, "bbox", []), dtype=np.float32).reshape(-1)
        return [round(float(v), 2) for v in bbox[:4]]

    def _detect(self, image) -> list:
        self.load()
        faces = self._app.get(image) if self._app is not None else []
        return list(faces or [])

    def enroll_image_bytes(self, data: bytes) -> FaceEnrollment:
        image = self._decode_image(data)
        faces = self._detect(image)
        if not faces:
            raise ValueError("未检测到人脸，请换一张清晰正脸照片")
        if len(faces) > 1:
            raise ValueError("照片中检测到多张人脸，请为每个人上传单人照片")
        face = faces[0]
        embedding = self._normalized_embedding(face)
        return FaceEnrollment(
            embedding=embedding,
            embedding_dim=int(embedding.size),
            model_name=face_model_id(self.model),
            detection_score=float(getattr(face, "det_score", 0.0) or 0.0),
            face_count=1,
        )

    def recognize_frame_bytes(self, data: bytes, people_repo) -> dict[str, Any]:
        image = self._decode_image(data)
        faces = self._detect(image)
        detections: list[dict[str, Any]] = []
        if not faces:
            return {
                "model_name": face_model_id(self.model),
                "provider": self._loaded_provider,
                "detections": detections,
                "registered_sample_count": 0,
            }
        first_embedding = self._normalized_embedding(faces[0])
        samples = people_repo.matching_face_samples(
            face_model_id(self.model),
            int(first_embedding.size),
        )
        sample_vectors = []
        for row in samples:
            vector = np.frombuffer(row["embedding"], dtype=np.float32).reshape(-1)
            if vector.size != first_embedding.size:
                continue
            norm = float(np.linalg.norm(vector))
            if norm <= 1e-8 or not math.isfinite(norm):
                continue
            sample_vectors.append((row, vector / norm))

        for index, face in enumerate(faces):
            embedding = self._normalized_embedding(face)
            ranked_by_person: dict[str, dict[str, Any]] = {}
            for row, sample_vector in sample_vectors:
                score = float(np.dot(embedding, sample_vector))
                person_id = str(row["person_id"])
                current = ranked_by_person.get(person_id)
                if current is None or score > current["confidence"]:
                    ranked_by_person[person_id] = {
                        "person_id": person_id,
                        "person_name": row.get("name"),
                        "confidence": score,
                    }
            ranked = sorted(
                ranked_by_person.values(),
                key=lambda item: item["confidence"],
                reverse=True,
            )
            best = ranked[0] if ranked else None
            runner_up = ranked[1] if len(ranked) > 1 else None
            margin = (
                best["confidence"] - runner_up["confidence"]
                if best and runner_up
                else 1.0
            )
            matched = bool(
                best
                and best["confidence"] >= self.match_threshold
                and margin >= self.match_margin
            )
            detections.append(
                {
                    "face_index": index,
                    "bbox": self._face_box(face),
                    "detection_score": float(getattr(face, "det_score", 0.0) or 0.0),
                    "matched": matched,
                    "person_id": best["person_id"] if matched and best else None,
                    "person_name": best["person_name"] if matched and best else None,
                    "confidence": best["confidence"] if best else None,
                    "runner_up_confidence": runner_up["confidence"] if runner_up else None,
                    "threshold": self.match_threshold,
                    "margin": margin,
                }
            )
        return {
            "model_name": face_model_id(self.model),
            "provider": self._loaded_provider,
            "detections": detections,
            "registered_sample_count": len(samples),
        }


_service: InsightFaceAttendanceService | None = None
_service_lock = threading.Lock()


def get_face_service() -> InsightFaceAttendanceService:
    global _service
    with _service_lock:
        if _service is None:
            _service = InsightFaceAttendanceService()
        return _service


def reset_face_service() -> None:
    global _service
    with _service_lock:
        _service = None


def get_all_face_models() -> dict[str, dict[str, Any]]:
    deps = face_dependency_status()
    models: dict[str, dict[str, Any]] = {}
    for key, item in SUPPORTED_FACE_MODELS.items():
        models[key] = {
            **item,
            "available": bool(deps["dependency_available"]),
            "dependency_available": bool(deps["dependency_available"]),
            "install_hint": (
                None
                if deps["dependency_available"]
                else "请安装 insightface 与 ONNXRuntime 后重试"
            ),
        }
    return models


def face_status(*, load: bool = False) -> dict[str, Any]:
    from app.services.model_config import public_model_settings

    service = get_face_service()
    load_error = None
    if load:
        try:
            service.load()
        except Exception as exc:
            load_error = str(exc)
    deps = face_dependency_status()
    public = public_model_settings().get("face", {})
    return {
        **public,
        "enabled": bool(config.face.enabled),
        "provider": public.get("provider") or config.face.provider,
        "model": normalize_face_model(public.get("model") or config.face.model),
        "device": public.get("device") or config.face.device,
        "loaded": service.loaded,
        "loaded_provider": service.loaded_provider,
        "model_id": face_model_id(service.model),
        "root": str(service.root),
        "match_threshold": config.face.match_threshold,
        "match_margin": config.face.match_margin,
        "frame_interval_sec": config.face.frame_interval_sec,
        "last_error": load_error or service.last_error,
        "dependencies": deps,
        "models": get_all_face_models(),
    }
