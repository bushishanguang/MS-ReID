from typing import List, Optional

from pydantic import BaseModel


class ImageListResponse(BaseModel):
    images: List[str]


class AttentionWeights(BaseModel):
    branch_1x1: float
    branch_3x3: float
    branch_5x5: float


class VisualResultResponse(BaseModel):
    image_name: str
    original: Optional[str] = None
    LH: Optional[str] = None
    HL: Optional[str] = None
    HH: Optional[str] = None
    attention_overlay: Optional[str] = None
    attention_weights: Optional[AttentionWeights] = None
    message: Optional[str] = None
