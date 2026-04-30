from tensorflow import keras
from tensorflow.keras import layers


def build_model(input_shape: tuple[int, ...], num_classes: int, config: dict) -> keras.Model:
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