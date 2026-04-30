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


            # Bloc Convolutif 1
            layers.Conv1D(config.get("filters_1", 16), config.get("kernel_1", 5), activation="relu", padding="same"),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(config.get("dropout_rate", 0.3)),

            # Bloc Convolutif 2
            layers.Conv1D(config.get("filters_2", 32), config.get("kernel_2", 3), activation="relu", padding="same"),
            layers.GlobalAveragePooling1D(),
            layers.Dropout(config.get("dropout_rate", 0.3)),

            # Bloc de Classification
            layers.Dense(config.get("dense", 32), activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="cnn",
    )
    return model