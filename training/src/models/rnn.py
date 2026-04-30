from __future__ import annotations

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

    units = int(config.get("units", 32))
    dense_units = int(config.get("dense", 16))

    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),

            layers.SimpleRNN(
                units,
                kernel_initializer=keras.initializers.GlorotUniform(seed=seed),
                recurrent_initializer=keras.initializers.Orthogonal(seed=seed + 1),
                bias_initializer=keras.initializers.Zeros(),
            ),

            layers.Dense(
                dense_units,
                activation="relu",
                kernel_initializer=keras.initializers.GlorotUniform(seed=seed + 2),
                bias_initializer=keras.initializers.Zeros(),
            ),

            layers.Dense(
                num_classes,
                activation="softmax",
                kernel_initializer=keras.initializers.GlorotUniform(seed=seed + 3),
                bias_initializer=keras.initializers.Zeros(),
            ),
        ],
        name="rnn",
    )
    return model
