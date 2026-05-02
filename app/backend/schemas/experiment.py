from typing import List

from pydantic import BaseModel


class ExperimentResult(BaseModel):
    experiment: str
    rank1: float
    map: float
    rank1_delta: float
    map_delta: float
    is_best_rank1: bool
    is_best_map: bool


class ExperimentResultsResponse(BaseModel):
    dataset: str
    results: List[ExperimentResult]


class ExperimentCurve(BaseModel):
    experiment: str
    epochs: List[int]
    accuracy: List[float]
    loss: List[float]


class ExperimentCurvesResponse(BaseModel):
    dataset: str
    curves: List[ExperimentCurve]

