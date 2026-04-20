"""Prétraitement et mise en forme des signaux ECG."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.model_selection import train_test_split

ModelType = Literal["mlp", "cnn", "rnn"]
NormalizationType = Literal["none", "zscore_per_signal"]


@dataclass
class PreparedData:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    input_shape: tuple[int, ...]


def zscore_per_signal(X: np.ndarray) -> np.ndarray:
    means = X.mean(axis=1, keepdims=True)
    stds = X.std(axis=1, keepdims=True)
    stds = np.where(stds == 0, 1.0, stds)
    return ((X - means) / stds).astype(np.float32)


def apply_normalization(X: np.ndarray, normalization: NormalizationType) -> np.ndarray:
    if normalization == "none":
        return X.astype(np.float32)
    if normalization == "zscore_per_signal":
        return zscore_per_signal(X)
    raise ValueError(f"Normalisation inconnue: {normalization}")


def reshape_inputs(X: np.ndarray, model_type: ModelType) -> np.ndarray:
    if model_type == "mlp":
        return X.astype(np.float32)
    if model_type in {"cnn", "rnn"}:
        return X[..., np.newaxis].astype(np.float32)
    raise ValueError(f"Type de modèle inconnu: {model_type}")


def prepare_data(
    X_train_full: np.ndarray,
    y_train_full: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_type: ModelType,
    normalization: NormalizationType = "zscore_per_signal",
    validation_size: float = 0.2,
    random_state: int = 42,
) -> PreparedData:
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=validation_size,
        stratify=y_train_full,
        random_state=random_state,
    )

    X_train = apply_normalization(X_train, normalization)
    X_val = apply_normalization(X_val, normalization)
    X_test = apply_normalization(X_test, normalization)

    X_train = reshape_inputs(X_train, model_type)
    X_val = reshape_inputs(X_val, model_type)
    X_test = reshape_inputs(X_test, model_type)

    return PreparedData(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train.astype(np.int32),
        y_val=y_val.astype(np.int32),
        y_test=y_test.astype(np.int32),
        input_shape=tuple(X_train.shape[1:]),
    )
