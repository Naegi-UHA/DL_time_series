from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .paths import DATA_RAW_DIR


def load_tsv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {path}")

    df = pd.read_csv(path, sep="\t", header=None)
    y = df.iloc[:, 0].to_numpy()
    X = df.iloc[:, 1:].to_numpy(dtype=np.float32)
    return X, y


def load_ecg200(data_dir: str | Path | None = None) -> dict:
    data_dir = Path(data_dir) if data_dir is not None else DATA_RAW_DIR

    train_path = data_dir / "ECG200_TRAIN.tsv"
    test_path = data_dir / "ECG200_TEST.tsv"

    X_train, y_train_raw = load_tsv(train_path)
    X_test, y_test_raw = load_tsv(test_path)

    unique_labels = sorted(set(np.unique(y_train_raw)).union(set(np.unique(y_test_raw))))
    label_to_id = {float(label): idx for idx, label in enumerate(unique_labels)}
    id_to_label = {idx: float(label) for label, idx in label_to_id.items()}

    y_train = np.array([label_to_id[float(y)] for y in y_train_raw], dtype=np.int32)
    y_test = np.array([label_to_id[float(y)] for y in y_test_raw], dtype=np.int32)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "label_to_id": label_to_id,
        "id_to_label": id_to_label,
        "input_length": X_train.shape[1],
        "num_classes": len(unique_labels),
    }


def normalize_per_signal(X: np.ndarray) -> np.ndarray:
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    return ((X - mean) / std).astype(np.float32)


def prepare_data(dataset: dict, model_type: str, val_size: float, seed: int) -> dict:
    X_train_full = normalize_per_signal(dataset["X_train"])
    X_test = normalize_per_signal(dataset["X_test"])
    y_train_full = dataset["y_train"]
    y_test = dataset["y_test"]

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=val_size,
        stratify=y_train_full,
        random_state=seed,
    )

    if model_type in {"cnn", "rnn"}:
        X_train = X_train[..., np.newaxis]
        X_val = X_val[..., np.newaxis]
        X_test = X_test[..., np.newaxis]

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "input_shape": tuple(X_train.shape[1:]),
    }
