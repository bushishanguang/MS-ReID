from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
APP_DIR = BACKEND_DIR.parent
REPO_ROOT = BACKEND_DIR.parents[1]
CORE_DIR = REPO_ROOT / "core"

STATIC_DIR = BACKEND_DIR / "static"
VISUAL_STATIC_DIR = STATIC_DIR / "visual"
VISUAL_GENERATED_DIR = VISUAL_STATIC_DIR / "generated"
GALLERY_STATIC_DIR = STATIC_DIR / "retrieval" / "gallery"
QDRANT_DIR = BACKEND_DIR / "vector_store" / "qdrant"

MARKET1501_DIR = CORE_DIR / "storage" / "datasets" / "market1501"
MARKET1501_GALLERY_DIR = MARKET1501_DIR / "bounding_box_test"
MARKET1501_QUERY_DIR = MARKET1501_DIR / "query"
OUTPUTS_DIR = CORE_DIR / "storage" / "outputs"

