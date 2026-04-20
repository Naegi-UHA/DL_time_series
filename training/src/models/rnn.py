from tensorflow import keras
from tensorflow.keras import layers


def build_model(input_shape: tuple[int, ...], num_classes: int, config: dict) -> keras.Model:
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.SimpleRNN(config.get("units", 32)),
            layers.Dense(config.get("dense", 16), activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="rnn",
    )
    return model
