from __future__ import annotations

from pathlib import Path
from typing import Optional

from functools import lru_cache

from qdrant_client import QdrantClient

from app.Logger import get_logger

logger = get_logger()


@lru_cache(maxsize=8)
def _shared_qdrant_client(path: str) -> QdrantClient:
    norm = str(Path(path).resolve())
    return QdrantClient(path=norm)


class CodeEmbeddingsStore:

    def __init__(
        self,
        collection_name: str = "code_embeddings",
        path: str | Path = "./.qdrant_code_embeddings",
        distance: str = "cosine",
        default_vector_name: Optional[str] = None,
    ) -> None:
        self.collection_name: str = str(collection_name)
        self.path = str(Path(path).resolve())
        self.distance: str = (distance or "cosine").lower()
        self.default_vector_name: Optional[str] = default_vector_name
        self._client: Optional[QdrantClient] = None

    def close(self) -> None:
        self._client = None

    def _ensure_client(self) -> QdrantClient:
        if self._client is None:
            self._client = _shared_qdrant_client(self.path)
        return self._client

    def _collection_exists(self, client: QdrantClient) -> bool:
        try:
            if hasattr(client, "collection_exists"):
                return bool(client.collection_exists(self.collection_name))
            client.get_collection(self.collection_name)
            return True
        except Exception:
            return False

    def _detect_existing_vector_names(self, client: QdrantClient) -> list[str]:
        try:
            info = client.get_collection(self.collection_name)
            vectors = getattr(getattr(getattr(info, "config", None), "params", None), "vectors", None)
            if isinstance(vectors, dict):
                return [str(k) for k in vectors.keys()]
        except Exception:
            pass
        return []

    def ensure_default_vector_name(self) -> Optional[str]:
        if self.default_vector_name:
            return self.default_vector_name

        client = self._ensure_client()
        if not self._collection_exists(client):
            return self.default_vector_name

        names = self._detect_existing_vector_names(client)
        if names:
            self.default_vector_name = names[0]
            logger.info(f"Auto-detected Qdrant named vector: default_vector_name='{self.default_vector_name}'")
        return self.default_vector_name

    def clear(self) -> None:
        client = self._ensure_client()
        try:
            client.delete_collection(collection_name=self.collection_name)
            logger.info(f"Qdrant collection deleted: {self.collection_name}")
        except Exception as e:
            logger.warning(f"Failed to delete Qdrant collection {self.collection_name}: {e}")
