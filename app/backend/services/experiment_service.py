import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from app.backend.schemas.experiment import ExperimentCurve, ExperimentResult
from app.backend.services.paths import OUTPUTS_DIR


EXPERIMENTS = [
    ("baseline", "Baseline"),
    ("wavelet", "Wavelet Only"),
    ("multiscale", "MultiScale Only"),
    ("wavelet_multiscale", "Wavelet + MultiScale"),
    ("multiscale_attention", "MultiScale + Attention"),
    ("full_msreid", "Wavelet + MultiScale + Attention(full)"),
]


class ExperimentService:
    _rank_pattern = re.compile(r"Rank-1\s*:?\s*([0-9.]+)%")
    _map_pattern = re.compile(r"mAP:\s*([0-9.]+)%")
    _train_pattern = re.compile(
        r"Epoch\[(?P<epoch>\d+)\].*?Loss:\s*(?P<loss>[0-9.]+),\s*Acc:\s*(?P<acc>[0-9.]+)"
    )

    def get_results(self, dataset: str) -> List[ExperimentResult]:
        parsed: List[Tuple[str, float, float]] = []
        for key, label in EXPERIMENTS:
            metrics = self._read_test_metrics(key, dataset)
            if metrics is None:
                continue
            parsed.append((label, metrics[0], metrics[1]))

        if not parsed:
            return []

        baseline_rank1, baseline_map = parsed[0][1], parsed[0][2]
        best_rank1 = max(row[1] for row in parsed)
        best_map = max(row[2] for row in parsed)

        return [
            ExperimentResult(
                experiment=label,
                rank1=rank1,
                map=map_value,
                rank1_delta=round(rank1 - baseline_rank1, 2),
                map_delta=round(map_value - baseline_map, 2),
                is_best_rank1=rank1 == best_rank1,
                is_best_map=map_value == best_map,
            )
            for label, rank1, map_value in parsed
        ]

    def get_curves(self, dataset: str) -> List[ExperimentCurve]:
        curves: List[ExperimentCurve] = []
        for key, label in EXPERIMENTS:
            curve = self._read_train_curve(key, dataset)
            if curve is None:
                continue
            epochs, losses, accuracies = curve
            curves.append(
                ExperimentCurve(
                    experiment=label,
                    epochs=epochs,
                    accuracy=accuracies,
                    loss=losses,
                )
            )
        return curves

    def _read_test_metrics(self, experiment_key: str, dataset: str) -> Optional[Tuple[float, float]]:
        log_path = OUTPUTS_DIR / experiment_key / "test" / dataset / f"{experiment_key}.log"
        text = self._read_text(log_path)
        if not text:
            return None

        maps = self._map_pattern.findall(text)
        ranks = self._rank_pattern.findall(text)
        if not maps or not ranks:
            return None
        return round(float(ranks[-1]), 2), round(float(maps[-1]), 2)

    def _read_train_curve(
        self, experiment_key: str, dataset: str
    ) -> Optional[Tuple[List[int], List[float], List[float]]]:
        log_path = OUTPUTS_DIR / experiment_key / "train" / dataset / f"{experiment_key}.log"
        text = self._read_text(log_path)
        if not text:
            return None

        by_epoch: Dict[int, Tuple[float, float]] = {}
        for match in self._train_pattern.finditer(text):
            epoch = int(match.group("epoch"))
            loss = round(float(match.group("loss")), 4)
            acc = round(float(match.group("acc")), 4)
            by_epoch[epoch] = (loss, acc)

        if not by_epoch:
            return None

        epochs = sorted(by_epoch)
        losses = [by_epoch[epoch][0] for epoch in epochs]
        accuracies = [by_epoch[epoch][1] for epoch in epochs]
        return epochs, losses, accuracies

    @staticmethod
    def _read_text(path: Path) -> str:
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8", errors="ignore")
