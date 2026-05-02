import threading
from typing import Dict, Iterable, List, Optional, Sequence

from app.backend.schemas.retrieval import RetrievalResult
from app.backend.services.paths import QDRANT_DIR


COLLECTION_NAME = "market1501_gallery"


class VectorStoreError(RuntimeError):
    pass


class QdrantService:
    def __init__(self, collection_name: str = COLLECTION_NAME) -> None:
        self.collection_name = collection_name
        self._client = None
        self._client_lock = threading.Lock()

    @property
    def client(self):
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    try:
                        from qdrant_client import QdrantClient
                    except ImportError as exc:
                        raise VectorStoreError("qdrant-client is not installed. Run uv sync first.") from exc
                    try:
                        QDRANT_DIR.mkdir(parents=True, exist_ok=True)
                        self._client = QdrantClient(path=str(QDRANT_DIR))
                    except Exception as exc:
                        raise VectorStoreError(f"Qdrant client initialization failed: {exc}") from exc
        return self._client

    def ensure_collection(self, vector_size: int, recreate: bool = False) -> None:
        try:
            from qdrant_client.models import Distance, VectorParams
        except ImportError as exc:
            raise VectorStoreError("qdrant-client is not installed. Run uv sync first.") from exc

        existing = self._collection_exists()
        if existing and recreate:
            self.client.delete_collection(self.collection_name)
            existing = False

        if not existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

    def upsert(self, points: Iterable[Dict]) -> None:
        try:
            from qdrant_client.models import PointStruct
        except ImportError as exc:
            raise VectorStoreError("qdrant-client is not installed. Run uv sync first.") from exc

        point_structs = [
            PointStruct(id=point["id"], vector=point["vector"], payload=point["payload"])
            for point in points
        ]
        if point_structs:
            self.client.upsert(collection_name=self.collection_name, points=point_structs)

    def search(self, vector: Sequence[float], top_k: int) -> List[RetrievalResult]:
        if not self._collection_exists():
            raise VectorStoreError(
                "Qdrant collection is missing. Run app/backend/scripts/build_market1501_vector_db.py first."
            )

        try:
            hits = self.client.search(
                collection_name=self.collection_name,
                query_vector=list(vector),
                limit=top_k,
                with_payload=True,
            )
        except Exception as exc:
            raise VectorStoreError(f"Qdrant search failed: {exc}") from exc

        results = []
        for index, hit in enumerate(hits, start=1):
            payload = hit.payload or {}
            results.append(
                RetrievalResult(
                    rank=index,
                    image_name=str(payload.get("image_name", "")),
                    image_url=str(payload.get("image_url", "")),
                    score=round(float(hit.score), 6),
                )
            )
        return results

    def warm_up(self, probe_vector: Optional[Sequence[float]] = None) -> None:
        _ = self.client
        if not self._collection_exists():
            raise VectorStoreError(
                "Qdrant collection is missing. Run app/backend/scripts/build_market1501_vector_db.py first."
            )
        if probe_vector:
            self.search(probe_vector, top_k=1)

    def _collection_exists(self) -> bool:
        try:
            return bool(self.client.collection_exists(self.collection_name))
        except AttributeError:
            try:
                collections = self.client.get_collections().collections
                return any(collection.name == self.collection_name for collection in collections)
            except Exception as exc:
                raise VectorStoreError(f"Qdrant collection check failed: {exc}") from exc
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError(f"Qdrant collection check failed: {exc}") from exc
