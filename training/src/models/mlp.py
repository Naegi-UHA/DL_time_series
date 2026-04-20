from tensorflow import keras
from tensorflow.keras import layers


def build_model(input_shape: tuple[int, ...], num_classes: int, config: dict) -> keras.Model:
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Dense(config.get("hidden_1", 64), activation="relu"),
            layers.Dropout(config.get("dropout", 0.2)),
            layers.Dense(config.get("hidden_2", 32), activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="mlp",
    )
    return model
