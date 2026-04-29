"""Data loading and preparation helpers for ECG200.

The raw labels stay readable as -1 and 1. When a model needs safe class ids,
this file also creates the 0/1 labels used during training.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from paths import DATA_DIR


def load_tsv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {path}")

    df = pd.read_csv(path, sep="\t", header=None)
    if df.shape[1] < 2:
        raise ValueError(f"Fichier invalide: {path}")

    labels = df.iloc[:, 0].to_numpy()
    signals = df.iloc[:, 1:].to_numpy(dtype=np.float32)
    return signals, labels


def clean_labels(labels: np.ndarray) -> np.ndarray:
    labels = labels.astype(float)

    if not np.all(labels == np.round(labels)):
        raise ValueError("Les labels doivent être des nombres entiers, par exemple -1.0 et 1.0")

    return labels.astype(np.int32)


def make_label_maps(labels: np.ndarray) -> tuple[list[int], dict[int, int], dict[int, int]]:
    class_labels = sorted(int(label) for label in np.unique(labels))
    label_to_model_id = {label: index for index, label in enumerate(class_labels)}
    model_id_to_label = {index: label for label, index in label_to_model_id.items()}
    return class_labels, label_to_model_id, model_id_to_label


def encode_labels(labels: np.ndarray, label_to_model_id: dict[int, int]) -> np.ndarray:
    return np.array([label_to_model_id[int(label)] for label in labels], dtype=np.int32)


def decode_labels(model_ids: np.ndarray, model_id_to_label: dict[int, int]) -> np.ndarray:
    return np.array([model_id_to_label[int(model_id)] for model_id in model_ids], dtype=np.int32)


def load_ecg200(data_dir: str | Path | None = None) -> dict:
    data_dir = Path(data_dir) if data_dir is not None else DATA_DIR

    X_train, y_train_raw = load_tsv(data_dir / "ECG200_TRAIN.tsv")
    X_test, y_test_raw = load_tsv(data_dir / "ECG200_TEST.tsv")

    y_train = clean_labels(y_train_raw)
    y_test = clean_labels(y_test_raw)

    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("Train et test n'ont pas la même longueur de signal")

    all_labels = np.concatenate([y_train, y_test])
    class_labels, label_to_model_id, model_id_to_label = make_label_maps(all_labels)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "class_labels": class_labels,
        "label_to_model_id": label_to_model_id,
        "model_id_to_label": model_id_to_label,
        "input_length": X_train.shape[1],
        "num_classes": len(class_labels),
    }


def normalize_per_signal(X: np.ndarray) -> np.ndarray:
    """Normalize each ECG signal alone, so every signal is centered around 0."""
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    return ((X - mean) / std).astype(np.float32)


def add_channel_axis_if_needed(X: np.ndarray, model_type: str) -> np.ndarray:
    """CNN and RNN need a shape like (samples, time, 1). MLP does not."""
    if model_type in {"cnn", "rnn"}:
        return X[..., np.newaxis]
    return X


def prepare_data(dataset: dict, model_type: str, val_size: float, seed: int) -> dict:
    """Prepare data for training while keeping the original labels for the final report."""
    X_train_full = normalize_per_signal(dataset["X_train"])
    X_test = normalize_per_signal(dataset["X_test"])

    X_train, X_val, y_train_original, y_val_original = train_test_split(
        X_train_full,
        dataset["y_train"],
        test_size=val_size,
        stratify=dataset["y_train"],
        random_state=seed,
    )

    X_train = add_channel_axis_if_needed(X_train, model_type)
    X_val = add_channel_axis_if_needed(X_val, model_type)
    X_test = add_channel_axis_if_needed(X_test, model_type)

    label_to_model_id = dataset["label_to_model_id"]

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": encode_labels(y_train_original, label_to_model_id),
        "y_val": encode_labels(y_val_original, label_to_model_id),
        "y_test": encode_labels(dataset["y_test"], label_to_model_id),
        "y_train_original": y_train_original,
        "y_val_original": y_val_original,
        "y_test_original": dataset["y_test"],
        "input_shape": tuple(X_train.shape[1:]),
    }
