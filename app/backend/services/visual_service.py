import base64
from io import BytesIO
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image, ImageOps

from app.backend.schemas.visual import AttentionWeights, VisualResultResponse
from app.backend.services.paths import (
    CORE_DIR,
    MARKET1501_GALLERY_DIR,
    MARKET1501_QUERY_DIR,
    OUTPUTS_DIR,
    VISUAL_GENERATED_DIR,
    VISUAL_STATIC_DIR,
)
from app.backend.services.reid_service import ReIDService, ReIDServiceError


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


class VisualService:
    def __init__(self) -> None:
        VISUAL_STATIC_DIR.mkdir(parents=True, exist_ok=True)
        VISUAL_GENERATED_DIR.mkdir(parents=True, exist_ok=True)
        self._attention_reid = ReIDService(
            weight_path=OUTPUTS_DIR
            / "multiscale_attention"
            / "train"
            / "market1501"
            / "resnet50_model_60.pth",
            config_file=CORE_DIR / "configs" / "exp6_multiscale_attention.yml",
        )

    def list_images(self, limit: int = 80) -> List[str]:
        names = []
        for source_dir in [VISUAL_STATIC_DIR / "original", MARKET1501_QUERY_DIR, MARKET1501_GALLERY_DIR]:
            if not source_dir.exists():
                continue
            for path in sorted(source_dir.iterdir()):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    names.append(path.name)
                if len(names) >= limit:
                    return names
        return names

    def get_result(self, image_name: str) -> Optional[VisualResultResponse]:
        source = self._find_image(image_name)
        if source is None:
            return None

        try:
            urls = self._ensure_visuals(source)
        except (OSError, ValueError) as exc:
            return VisualResultResponse(image_name=image_name, message=str(exc))

        return VisualResultResponse(image_name=image_name, **urls)

    def compute_result(self, image: Image.Image, image_name: str) -> VisualResultResponse:
        rgb = image.convert("RGB")
        lh, hl, hh = self._haar_high_frequency(rgb)
        overlay, weights = self._compute_attention_overlay(rgb)
        return VisualResultResponse(
            image_name=image_name,
            original=self._data_url(rgb),
            LH=self._data_url(lh),
            HL=self._data_url(hl),
            HH=self._data_url(hh),
            attention_overlay=self._data_url(overlay),
            attention_weights=weights,
        )

    def _find_image(self, image_name: str) -> Optional[Path]:
        clean_name = Path(image_name).name
        candidates = [
            VISUAL_STATIC_DIR / "original" / clean_name,
            MARKET1501_QUERY_DIR / clean_name,
            MARKET1501_GALLERY_DIR / clean_name,
        ]
        for candidate in candidates:
            if candidate.exists() and candidate.suffix.lower() in IMAGE_EXTENSIONS:
                return candidate
        return None

    def _ensure_visuals(self, source: Path) -> Dict[str, str]:
        stem = source.stem
        ext = ".jpg"
        original_path = VISUAL_GENERATED_DIR / f"{stem}_original{ext}"
        lh_path = VISUAL_GENERATED_DIR / f"{stem}_LH{ext}"
        hl_path = VISUAL_GENERATED_DIR / f"{stem}_HL{ext}"
        hh_path = VISUAL_GENERATED_DIR / f"{stem}_HH{ext}"
        overlay_path = VISUAL_GENERATED_DIR / f"{stem}_attention_overlay{ext}"

        if not original_path.exists():
            shutil.copyfile(source, original_path)

        missing = [p for p in [lh_path, hl_path, hh_path, overlay_path] if not p.exists()]
        if missing:
            image = Image.open(source).convert("RGB")
            lh, hl, hh = self._haar_high_frequency(image)
            lh.save(lh_path, quality=92)
            hl.save(hl_path, quality=92)
            hh.save(hh_path, quality=92)
            self._attention_overlay(image, hh).save(overlay_path, quality=92)

        return {
            "original": self._url(original_path),
            "LH": self._url(lh_path),
            "HL": self._url(hl_path),
            "HH": self._url(hh_path),
            "attention_overlay": self._url(overlay_path),
        }

    @staticmethod
    def _haar_high_frequency(image: Image.Image):
        arr = np.asarray(image).astype("float32")
        height, width = arr.shape[:2]
        if height % 2:
            arr = np.pad(arr, ((0, 1), (0, 0), (0, 0)), mode="edge")
        if width % 2:
            arr = np.pad(arr, ((0, 0), (0, 1), (0, 0)), mode="edge")

        tl = arr[0::2, 0::2]
        tr = arr[0::2, 1::2]
        bl = arr[1::2, 0::2]
        br = arr[1::2, 1::2]

        lh = (tl - tr + bl - br) * 0.5
        hl = (tl + tr - bl - br) * 0.5
        hh = (tl - tr - bl + br) * 0.5

        return tuple(VisualService._detail_to_image(detail, (width, height)) for detail in [lh, hl, hh])

    @staticmethod
    def _detail_to_image(detail: np.ndarray, size):
        magnitude = np.sqrt(np.mean(np.square(detail), axis=2))
        max_value = float(np.max(magnitude))
        if max_value <= 1e-6:
            scaled = np.zeros_like(magnitude, dtype="uint8")
        else:
            scaled = np.clip(magnitude / max_value * 255, 0, 255).astype("uint8")
        return Image.fromarray(scaled, mode="L").resize(size, Image.BILINEAR).convert("RGB")

    @staticmethod
    def _attention_overlay(image: Image.Image, detail: Image.Image) -> Image.Image:
        heat = ImageOps.colorize(detail.convert("L"), black="#0F172A", white="#F97316")
        return Image.blend(image, heat, alpha=0.36)

    def _compute_attention_overlay(self, image: Image.Image):
        output = self._attention_reid.extract_visuals(image)
        fusion = output.get("visuals", {}).get("fusion")
        if not fusion:
            raise ReIDServiceError("MultiScale attention fusion visuals were not returned by the model.")

        branch_weights = fusion.get("branch_weights")
        activation_heatmap = fusion.get("activation_heatmap")
        if branch_weights is None or activation_heatmap is None:
            raise ReIDServiceError("Missing branch_weights or activation_heatmap in fusion visuals.")

        weights = branch_weights.detach().cpu().view(-1).numpy()
        if weights.shape[0] != 3:
            raise ReIDServiceError(f"Expected 3 attention branch weights, got {weights.shape[0]}.")

        heatmap = activation_heatmap.detach().cpu().squeeze().numpy()
        overlay = self._overlay_heatmap(image, heatmap)
        return overlay, AttentionWeights(
            branch_1x1=round(float(weights[0]), 6),
            branch_3x3=round(float(weights[1]), 6),
            branch_5x5=round(float(weights[2]), 6),
        )

    @staticmethod
    def _overlay_heatmap(image: Image.Image, heatmap: np.ndarray) -> Image.Image:
        response = np.maximum(heatmap.astype("float32"), 0)
        if float(response.max()) <= 1e-8:
            response = heatmap.astype("float32")

        low, high = np.percentile(response, [2, 97])
        if high <= low:
            low = float(response.min())
            high = float(response.max())
        if high <= low:
            normalized = np.zeros_like(response, dtype="float32")
        else:
            normalized = np.clip((response - low) / (high - low), 0, 1)

        normalized = np.power(normalized, 0.55)
        mask = Image.fromarray((normalized * 255).astype("uint8"), mode="L").resize(image.size, Image.BILINEAR)
        heat = ImageOps.colorize(mask, black="#1D4ED8", mid="#FACC15", white="#DC2626").convert("RGBA")
        alpha = np.asarray(mask).astype("float32") / 255.0
        alpha = np.clip(np.power(alpha, 0.8) * 230, 0, 230).astype("uint8")
        heat.putalpha(Image.fromarray(alpha, mode="L"))
        base = image.convert("RGBA")
        return Image.alpha_composite(base, heat).convert("RGB")

    @staticmethod
    def _url(path: Path) -> str:
        relative = path.relative_to(VISUAL_STATIC_DIR.parent).as_posix()
        return f"/static/{relative}"

    @staticmethod
    def _data_url(image: Image.Image) -> str:
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=92)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"
