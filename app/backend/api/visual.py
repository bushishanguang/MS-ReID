from io import BytesIO

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from PIL import Image, UnidentifiedImageError

from app.backend.schemas.visual import ImageListResponse, VisualResultResponse
from app.backend.services.reid_service import ReIDServiceError
from app.backend.services.visual_service import VisualService


router = APIRouter()
service = VisualService()


@router.get("/images", response_model=ImageListResponse)
def list_images(limit: int = Query(80, ge=1, le=500)):
    return ImageListResponse(images=service.list_images(limit=limit))


@router.get("/result", response_model=VisualResultResponse)
def get_visual_result(image_name: str = Query(..., min_length=1)):
    result = service.get_result(image_name)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Image not found: {image_name}")
    return result


@router.post("/result", response_model=VisualResultResponse)
async def compute_visual_result(file: UploadFile = File(...)):
    try:
        image = Image.open(BytesIO(await file.read())).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="uploaded file is not a valid image")

    try:
        return service.compute_result(image, file.filename or "uploaded.jpg")
    except ReIDServiceError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
