import argparse
import shutil
import sys
from pathlib import Path
from typing import Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.backend.services.paths import GALLERY_STATIC_DIR, MARKET1501_GALLERY_DIR
from app.backend.services.qdrant_service import QdrantService
from app.backend.services.reid_service import ReIDService


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def iter_images(root: Path, limit: int = 0) -> Iterable[Path]:
    count = 0
    for path in sorted(root.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path
            count += 1
            if limit and count >= limit:
                return


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a local Qdrant gallery DB for Market1501.")
    parser.add_argument("--gallery-root", type=Path, default=MARKET1501_GALLERY_DIR)
    parser.add_argument("--weight", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--collection", default="market1501_gallery")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0, help="Optional debug limit. 0 means all images.")
    parser.add_argument("--recreate", action="store_true")
    args = parser.parse_args()

    if not args.gallery_root.exists():
        raise FileNotFoundError(f"gallery root not found: {args.gallery_root}")

    GALLERY_STATIC_DIR.mkdir(parents=True, exist_ok=True)
    reid = ReIDService(weight_path=args.weight, config_file=args.config)
    qdrant = QdrantService(collection_name=args.collection)

    batch: List[dict] = []
    first_vector_size = None
    total = 0

    for image_path in iter_images(args.gallery_root, limit=args.limit):
        vector = list(reid.extract_feature(image_path))
        if first_vector_size is None:
            first_vector_size = len(vector)
            qdrant.ensure_collection(vector_size=first_vector_size, recreate=args.recreate)

        target = GALLERY_STATIC_DIR / image_path.name
        if not target.exists():
            shutil.copyfile(image_path, target)

        batch.append(
            {
                "id": total + 1,
                "vector": vector,
                "payload": {
                    "image_name": image_path.name,
                    "image_url": f"/static/retrieval/gallery/{image_path.name}",
                    "source_path": str(image_path),
                },
            }
        )
        total += 1

        if len(batch) >= args.batch_size:
            qdrant.upsert(batch)
            batch.clear()
            print(f"Indexed {total} images")

    if batch:
        qdrant.upsert(batch)

    print(f"Finished. Indexed {total} gallery images into collection '{args.collection}'.")


if __name__ == "__main__":
    main()
