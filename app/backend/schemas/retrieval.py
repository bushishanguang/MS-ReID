from typing import List

from pydantic import BaseModel


class RetrievalResult(BaseModel):
    rank: int
    image_name: str
    image_url: str
    score: float


class RetrievalSearchResponse(BaseModel):
    query: str
    top_k: int
    results: List[RetrievalResult]

