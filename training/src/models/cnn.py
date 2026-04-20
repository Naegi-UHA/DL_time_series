from tensorflow import keras
from tensorflow.keras import layers


def build_model(input_shape: tuple[int, ...], num_classes: int, config: dict) -> keras.Model:
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv1D(config.get("filters_1", 16), config.get("kernel_1", 5), activation="relu", padding="same"),
            layers.MaxPooling1D(pool_size=2),
            layers.Conv1D(config.get("filters_2", 32), config.get("kernel_2", 3), activation="relu", padding="same"),
            layers.GlobalAveragePooling1D(),
            layers.Dense(config.get("dense", 32), activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="cnn",
    )
    return model
