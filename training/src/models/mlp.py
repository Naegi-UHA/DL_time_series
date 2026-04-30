import os
import random

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}

    return bool(value)


def _configure_reproducibility(seed: int, deterministic: bool) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

    if deterministic:
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass


def build_model(input_shape: tuple[int, ...], num_classes: int, config: dict) -> keras.Model:
    seed = int(config.get("seed", 42))
    deterministic = _as_bool(config.get("deterministic", True))

    _configure_reproducibility(seed, deterministic)


    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),

            # Normalisation des entrées
            layers.BatchNormalization(),

            # couche 1
            layers.Dense(config.get("hidden_1", 64), activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(config.get("dropout_1", 0.3)),

            # couche 2
            layers.Dense(config.get("hidden_2", 32), activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(config.get("dropout_2", 0.3)),

            # couche 3
            layers.Dense(config.get("hidden_3", 16), activation="relu"),

            # Sortie
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="mlp",
    )

    return model