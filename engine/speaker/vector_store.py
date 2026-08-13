"""Process-local vector store for live anonymous speaker clusters.

ChromaDB is a fast optional backend, but its hnsw dependency often requires
Microsoft C++ Build Tools on fresh Windows machines.  The in-memory fallback
implements the small subset of the Chroma collection API used by the speaker
engines, which keeps local deployment usable when Chroma cannot be installed.
"""
from __future__ import annotations

import logging
import math
from copy import deepcopy
from typing import Any

import numpy as np


logger = logging.getLogger("Matrix_Speaker")


class InMemoryVectorCollection:
    """Minimal Chroma-compatible collection for cosine nearest-neighbor search."""

    def __init__(self, name: str):
        self.name = name
        self._rows: dict[str, dict[str, Any]] = {}

    def add(self, *, ids, embeddings, metadatas=None, **_kwargs) -> None:
        for index, row_id in enumerate(ids):
            metadata = metadatas[index] if metadatas and index < len(metadatas) else {}
            self._rows[str(row_id)] = {
                "embedding": self._normalize(embeddings[index]),
                "metadata": dict(metadata or {}),
            }

    def query(self, *, query_embeddings, n_results: int = 3, where=None, **_kwargs) -> dict:
        where = dict(where or {})
        output_ids: list[list[str]] = []
        output_distances: list[list[float]] = []
        output_metadatas: list[list[dict[str, Any]]] = []
        for query_embedding in query_embeddings:
            query = self._normalize(query_embedding)
            scored: list[tuple[float, str, dict[str, Any]]] = []
            for row_id, row in self._rows.items():
                metadata = row["metadata"]
                if not self._metadata_matches(metadata, where):
                    continue
                distance = self._cosine_distance(query, row["embedding"])
                scored.append((distance, row_id, deepcopy(metadata)))
            scored.sort(key=lambda item: item[0])
            chosen = scored[: max(0, int(n_results))]
            output_distances.append([float(item[0]) for item in chosen])
            output_ids.append([item[1] for item in chosen])
            output_metadatas.append([item[2] for item in chosen])
        return {
            "ids": output_ids,
            "distances": output_distances,
            "metadatas": output_metadatas,
        }

    def get(self, *, ids, include=None, **_kwargs) -> dict:
        embeddings = []
        metadatas = []
        found_ids = []
        for row_id in ids:
            row = self._rows.get(str(row_id))
            if row is None:
                continue
            found_ids.append(str(row_id))
            embeddings.append(row["embedding"].tolist())
            metadatas.append(deepcopy(row["metadata"]))
        result: dict[str, Any] = {"ids": found_ids}
        include = set(include or ())
        if "embeddings" in include:
            result["embeddings"] = embeddings
        if "metadatas" in include:
            result["metadatas"] = metadatas
        return result

    def update(self, *, ids, embeddings=None, metadatas=None, **_kwargs) -> None:
        for index, row_id in enumerate(ids):
            row = self._rows.get(str(row_id))
            if row is None:
                continue
            if embeddings is not None and index < len(embeddings):
                row["embedding"] = self._normalize(embeddings[index])
            if metadatas is not None and index < len(metadatas):
                row["metadata"] = dict(metadatas[index] or {})

    def delete(self, *, where=None, ids=None, **_kwargs) -> None:
        if ids is not None:
            for row_id in ids:
                self._rows.pop(str(row_id), None)
            return
        where = dict(where or {})
        if not where:
            self._rows.clear()
            return
        for row_id in [
            row_id
            for row_id, row in self._rows.items()
            if self._metadata_matches(row["metadata"], where)
        ]:
            self._rows.pop(row_id, None)

    def count(self) -> int:
        return len(self._rows)

    @staticmethod
    def _normalize(value) -> np.ndarray:
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if not math.isfinite(norm) or norm <= 1e-12:
            return vector
        return vector / norm

    @staticmethod
    def _metadata_matches(metadata: dict[str, Any], where: dict[str, Any]) -> bool:
        return all(metadata.get(key) == expected for key, expected in where.items())

    @staticmethod
    def _cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
        if left.size != right.size:
            return 1.0
        denom = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denom <= 1e-12:
            return 1.0
        similarity = float(np.dot(left, right) / denom)
        return max(0.0, min(2.0, 1.0 - similarity))


class InMemoryVectorClient:
    def get_or_create_collection(self, *, name: str, metadata=None, **_kwargs) -> InMemoryVectorCollection:
        return InMemoryVectorCollection(name)


def create_runtime_vector_collection(name: str):
    """Create a process-local collection, preferring Chroma when available."""
    force_memory = str(__import__("os").environ.get("SPEAKER_VECTOR_STORE", "")).lower()
    if force_memory in {"memory", "inmemory", "numpy"}:
        logger.info("[SpeakerVectorStore] using in-memory collection: %s", name)
        client = InMemoryVectorClient()
        return client, client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})
    try:
        import chromadb
        from chromadb.config import Settings

        client = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))
        collection = client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )
        return client, collection
    except Exception as exc:
        logger.warning(
            "[SpeakerVectorStore] ChromaDB unavailable, using in-memory fallback: %s",
            exc,
        )
        client = InMemoryVectorClient()
        return client, client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})
