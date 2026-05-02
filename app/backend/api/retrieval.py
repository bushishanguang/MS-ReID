import logging
from io import BytesIO

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from app.backend.schemas.retrieval import RetrievalSearchResponse
from app.backend.services.qdrant_service import QdrantService, VectorStoreError
from app.backend.services.reid_service import ReIDService, ReIDServiceError


router = APIRouter()
logger = logging.getLogger(__name__)
reid_service = ReIDService()
qdrant_service = QdrantService()


def warmup_services() -> None:
    probe_vector = None
    try:
        probe_vector = reid_service.warm_up()
    except ReIDServiceError:
        logger.exception("Failed to warm up ReID service")

    try:
        qdrant_service.warm_up(probe_vector=probe_vector)
    except VectorStoreError:
        logger.exception("Failed to warm up Qdrant service")


@router.post("/search", response_model=RetrievalSearchResponse)
async def search(file: UploadFile = File(...), top_k: int = Form(5)):
    if top_k not in {1, 5, 10}:
        raise HTTPException(status_code=400, detail="top_k only allows 1, 5, or 10")
    file_name = file.filename or ""

    try:
        image = Image.open(BytesIO(await file.read())).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="uploaded file is not a valid image")

    try:
        vector = reid_service.extract_feature(image)
        results = qdrant_service.search(vector, top_k=top_k)
    except ReIDServiceError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except VectorStoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    return RetrievalSearchResponse(query=file_name or "query.jpg", top_k=top_k, results=results)
