from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.backend.api import experiments, retrieval, visual


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"


def create_app() -> FastAPI:
    app = FastAPI(
        title="MS-ReID Web API",
        description="FastAPI prototype for experiment display and person retrieval.",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ],
        allow_origin_regex=r"^http://(localhost|127\.0\.0\.1):\d+$",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    app.include_router(visual.router, prefix="/api/visual", tags=["visual"])
    app.include_router(experiments.router, prefix="/api/experiments", tags=["experiments"])
    app.include_router(retrieval.router, prefix="/api/retrieval", tags=["retrieval"])
    app.add_event_handler("startup", retrieval.warmup_services)

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    return app


app = create_app()
