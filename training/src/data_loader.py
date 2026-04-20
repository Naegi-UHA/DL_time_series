from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw"


@dataclass
class ECGDataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    label_to_id: Dict[float, int]
    id_to_label: Dict[int, float]
    input_length: int


def _read_tsv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {path}")

    df = pd.read_csv(path, sep="\t", header=None)
    if df.shape[1] < 2:
        raise ValueError(f"Fichier invalide (au moins 2 colonnes attendues): {path}")

    y = df.iloc[:, 0].to_numpy()
    X = df.iloc[:, 1:].to_numpy(dtype=np.float32)
    return X, y


def load_ecg200(data_dir: str | Path | None = None) -> ECGDataset:
    data_dir = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR

    train_path = data_dir / "ECG200_TRAIN.tsv"
    test_path = data_dir / "ECG200_TEST.tsv"

    X_train, y_train_raw = _read_tsv(train_path)
    X_test, y_test_raw = _read_tsv(test_path)

    unique_labels = sorted(set(np.unique(y_train_raw)).union(set(np.unique(y_test_raw))))
    label_to_id = {float(label): idx for idx, label in enumerate(unique_labels)}
    id_to_label = {idx: float(label) for label, idx in label_to_id.items()}

    y_train = np.array([label_to_id[float(label)] for label in y_train_raw], dtype=np.int32)
    y_test = np.array([label_to_id[float(label)] for label in y_test_raw], dtype=np.int32)

    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(
            "Les longueurs des signaux train et test ne correspondent pas : "
            f"{X_train.shape[1]} vs {X_test.shape[1]}"
        )

    return ECGDataset(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        label_to_id=label_to_id,
        id_to_label=id_to_label,
        input_length=X_train.shape[1],
    )