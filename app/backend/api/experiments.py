from fastapi import APIRouter, HTTPException, Query

from app.backend.schemas.experiment import ExperimentCurvesResponse, ExperimentResultsResponse
from app.backend.services.experiment_service import ExperimentService


router = APIRouter()
service = ExperimentService()


@router.get("/results", response_model=ExperimentResultsResponse)
def get_results(dataset: str = Query("market1501")):
    normalized = dataset.lower()
    if normalized not in {"market1501", "dukemtmc"}:
        raise HTTPException(status_code=400, detail="dataset must be market1501 or dukemtmc")
    return ExperimentResultsResponse(dataset=normalized, results=service.get_results(normalized))


@router.get("/curves", response_model=ExperimentCurvesResponse)
def get_curves(dataset: str = Query("market1501")):
    normalized = dataset.lower()
    if normalized != "market1501":
        raise HTTPException(status_code=400, detail="curves currently support market1501 only")
    return ExperimentCurvesResponse(dataset=normalized, curves=service.get_curves(normalized))

