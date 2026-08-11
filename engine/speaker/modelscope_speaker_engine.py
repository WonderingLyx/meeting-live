"""Generic ModelScope speaker embedding engine.

This wrapper is used for extra 3D-Speaker / ModelScope speaker verification
models that expose the same ``Model.from_pretrained`` inference contract as the
existing CamPlus, ERes2Net and Wespeaker engines.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Tuple

import chromadb
import numpy as np
import torch
from chromadb.config import Settings
from modelscope.models import Model

from engine.speaker.base_engine import BaseSpeakerEngine
from engine.speaker.speaker_factory import ENGINE_CONFIG


logger = logging.getLogger("Matrix_Speaker")


class ModelScopeSpeakerEngine(BaseSpeakerEngine):
    """Config-driven speaker embedding engine for ModelScope models."""

    def __init__(self, engine_type: str):
        if engine_type not in ENGINE_CONFIG:
            raise ValueError(f"Unknown ModelScope speaker engine: {engine_type}")
        self.engine_type = engine_type
        self.info = ENGINE_CONFIG[engine_type]
        self.device = "cpu"
        self.THRESHOLD_PROFILE = tuple(
            self.info.get("threshold_profile") or (0.42, 0.52, 0.52, 0.62)
        )

        model_id = str(self.info["model"])
        cache_name = str(self.info.get("cache_name") or engine_type)
        logger.info("[%s] Loading model...", self._model_name)
        from app.services.model_resolver import resolve_modelscope

        local = resolve_modelscope(
            model_id,
            "speaker",
            cache_name,
            revision=self.info.get("model_revision"),
        )
        logger.info("[%s] Loading from local path: %s", self._model_name, local)
        self.model = Model.from_pretrained(local, device="cpu")
        self.model.eval()

        self.chroma_client = chromadb.EphemeralClient(
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name=f"speaker_fingerprints_{engine_type}",
            metadata={"hnsw:space": "cosine"},
        )
        self.emb_buffer = defaultdict(list)
        self.EMB_BUFFER_SIZE = 5
        self.match_history = defaultdict(list)
        self.HISTORY_SIZE = 3
        self.ENGINE_NAME = self._model_name
        logger.info("[%s] Initialized", self._model_name)

    @property
    def _model_name(self) -> str:
        return str(self.info.get("name") or self.engine_type)

    def extract_feat(self, audio_data: np.ndarray) -> Tuple[np.ndarray, float]:
        """Extract a normalized speaker embedding from 16 kHz mono PCM."""
        try:
            audio_duration = len(audio_data) / 16000.0
            tensor = torch.FloatTensor(audio_data).unsqueeze(0)
            with torch.no_grad():
                outputs = self.model(tensor)
                final_emb = self._extract_embedding_array(outputs)
                final_emb = final_emb / (np.linalg.norm(final_emb) + 1e-6)
                return final_emb, audio_duration
        except Exception as e:
            logger.error("[%s] Extraction failed: %s", self._model_name, e)
            return None, 0

    @staticmethod
    def _extract_embedding_array(outputs) -> np.ndarray:
        if isinstance(outputs, dict):
            for key in (
                "spk_embedding",
                "speaker_embedding",
                "embedding",
                "emb",
                "xvector",
            ):
                if key in outputs:
                    outputs = outputs[key]
                    break
            else:
                outputs = next(iter(outputs.values()))
        if isinstance(outputs, (list, tuple)):
            if not outputs:
                raise ValueError("empty speaker model output")
            outputs = outputs[0]
        if hasattr(outputs, "detach"):
            arr = outputs.detach().cpu().numpy()
        elif hasattr(outputs, "cpu") and hasattr(outputs.cpu(), "numpy"):
            arr = outputs.cpu().numpy()
        else:
            arr = np.asarray(outputs)
        arr = np.asarray(arr, dtype=np.float32).reshape(-1)
        if not arr.size or not np.isfinite(arr).all():
            raise ValueError("invalid speaker embedding")
        return arr
