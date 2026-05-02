import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import torch
import torch.nn.functional as F
from PIL import Image

from app.backend.services.paths import CORE_DIR, OUTPUTS_DIR, REPO_ROOT


class ReIDServiceError(RuntimeError):
    pass


class ReIDService:
    def __init__(
        self,
        weight_path: Optional[Union[str, Path]] = None,
        config_file: Optional[Union[str, Path]] = None,
        num_classes: int = 751,
    ) -> None:
        self.weight_path = Path(weight_path) if weight_path else self._default_weight()
        self.config_file = Path(config_file) if config_file else self._default_config(self.weight_path)
        self.num_classes = num_classes
        self._model = None
        self._transform = None
        self._device = None
        self._input_size = None
        self._warmup_vector = None
        self._load_lock = threading.Lock()

    def extract_feature(self, image: Union[str, Path, Image.Image]) -> Sequence[float]:
        model, transform, device = self._ensure_loaded()
        pil_image = self._to_pil(image)
        tensor = transform(pil_image).unsqueeze(0).to(device)
        with torch.no_grad():
            feature = model(tensor)
            if isinstance(feature, dict):
                feature = feature["embedding"]
            feature = F.normalize(feature, p=2, dim=1)
        return feature.squeeze(0).detach().cpu().tolist()

    def extract_visuals(self, image: Union[str, Path, Image.Image]) -> Dict[str, Any]:
        model, transform, device = self._ensure_loaded()
        pil_image = self._to_pil(image)
        tensor = transform(pil_image).unsqueeze(0).to(device)
        with torch.no_grad():
            if hasattr(model, "extract_with_visuals"):
                output = model.extract_with_visuals(tensor)
            else:
                output = model(tensor, return_visuals=True)
        if not isinstance(output, dict) or "visuals" not in output:
            raise ReIDServiceError("The selected ReID model did not return visualization tensors.")
        return output

    def warm_up(self) -> Sequence[float]:
        model, _, device = self._ensure_loaded()
        if self._warmup_vector is not None:
            return self._warmup_vector

        with self._load_lock:
            if self._warmup_vector is not None:
                return self._warmup_vector

            height, width = self._input_size or (256, 128)
            dummy = torch.zeros((1, 3, int(height), int(width)), device=device)
            with torch.no_grad():
                feature = model(dummy, return_visuals=False)
                if isinstance(feature, dict):
                    feature = feature["embedding"]
                feature = F.normalize(feature, p=2, dim=1)
            if device.type == "cuda":
                torch.cuda.synchronize(device)

            self._warmup_vector = feature.squeeze(0).detach().cpu().tolist()
            return self._warmup_vector

    def _ensure_loaded(self):
        if self._model is not None and self._transform is not None and self._device is not None:
            return self._model, self._transform, self._device

        with self._load_lock:
            if self._model is not None and self._transform is not None and self._device is not None:
                return self._model, self._transform, self._device

            if not self.weight_path or not self.weight_path.exists():
                raise ReIDServiceError(
                    "ReID model weight was not found. Set MS_REID_WEIGHT or pass --weight to the build script."
                )

            try:
                self._add_repo_to_path()
                from core.config import cfg
                from core.data.transforms.build import build_transforms
                from core.modeling import build_model
            except Exception as exc:
                raise ReIDServiceError(f"Failed to import core model code: {exc}") from exc

            local_cfg = cfg.clone()
            if self.config_file and self.config_file.exists():
                local_cfg.merge_from_file(str(self.config_file))
            local_cfg.MODEL.PRETRAIN_CHOICE = "self"
            local_cfg.TEST.WEIGHT = str(self.weight_path)

            device_name = "cuda" if torch.cuda.is_available() and local_cfg.MODEL.DEVICE == "cuda" else "cpu"
            device = torch.device(device_name)
            if device.type == "cuda":
                torch.backends.cudnn.benchmark = True

            try:
                model = build_model(local_cfg, self.num_classes)
                self._load_weights(model, self.weight_path, device)
                model.to(device)
                model.eval()
                transform = build_transforms(local_cfg, is_train=False)
            except Exception as exc:
                raise ReIDServiceError(f"Failed to initialize ReID model: {exc}") from exc

            self._model = model
            self._transform = transform
            self._device = device
            self._input_size = tuple(local_cfg.INPUT.SIZE_TEST)
            return model, transform, device

    @staticmethod
    def _to_pil(image: Union[str, Path, Image.Image]) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        return Image.open(image).convert("RGB")

    @staticmethod
    def _load_weights(model, weight_path: Path, device) -> None:
        checkpoint = torch.load(str(weight_path), map_location=device)
        if hasattr(checkpoint, "state_dict"):
            state_dict = checkpoint.state_dict()
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
        else:
            raise ReIDServiceError("Unsupported checkpoint format")

        target = model.state_dict()
        compatible = {}
        for name, value in state_dict.items():
            clean_name = name.replace("module.", "")
            if "classifier" in clean_name:
                continue
            if clean_name in target and target[clean_name].shape == value.shape:
                compatible[clean_name] = value
        target.update(compatible)
        model.load_state_dict(target)

    @staticmethod
    def _add_repo_to_path() -> None:
        repo = str(REPO_ROOT)
        core = str(CORE_DIR)
        for path in [repo, core]:
            if path not in sys.path:
                sys.path.insert(0, path)

    @staticmethod
    def _default_weight() -> Optional[Path]:
        env_weight = os.getenv("MS_REID_WEIGHT")
        if env_weight:
            return Path(env_weight)

        candidates = [
            OUTPUTS_DIR / "multiscale" / "train" / "market1501" / "resnet50_model_60.pth",
            OUTPUTS_DIR / "multiscale_attention" / "train" / "market1501" / "resnet50_model_60.pth",
            OUTPUTS_DIR / "full_msreid" / "train" / "market1501" / "resnet50_model_60.pth",
            OUTPUTS_DIR / "baseline" / "train" / "market1501" / "resnet50_model_60.pth",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    @staticmethod
    def _default_config(weight_path: Optional[Path]) -> Optional[Path]:
        if weight_path is None:
            return None

        experiment = None
        parts = list(weight_path.parts)
        if "outputs" in parts:
            index = parts.index("outputs")
            if index + 1 < len(parts):
                experiment = parts[index + 1]

        config_map = {
            "baseline": CORE_DIR / "configs" / "exp1_baseline.yml",
            "wavelet": CORE_DIR / "configs" / "exp2_wavelet.yml",
            "multiscale": CORE_DIR / "configs" / "exp3_multiscale.yml",
            "wavelet_multiscale": CORE_DIR / "configs" / "exp4_wavelet_multiscale.yml",
            "full_msreid": CORE_DIR / "configs" / "exp5_full_msreid.yml",
            "multiscale_attention": CORE_DIR / "configs" / "exp6_multiscale_attention.yml",
        }
        config = config_map.get(experiment or "")
        return config if config and config.exists() else None
