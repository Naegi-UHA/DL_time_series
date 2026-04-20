import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

MODELS_DIR = Path("./models")
MODEL_PATH = MODELS_DIR / "best_model.keras"
PREPROCESSING_PATH = MODELS_DIR / "preprocessing.json"
LABEL_MAP_PATH = MODELS_DIR / "label_map.json"
METADATA_PATH = MODELS_DIR / "metadata.json"

model = None
preprocessing = {}
label_to_id = {}
id_to_label = {}
metadata = {}


def patch_keras_loading_compatibility():
    original_layer_init = tf.keras.layers.Layer.__init__

    def patched_layer_init(self, *args, **kwargs):
        kwargs.pop("quantization_config", None)
        return original_layer_init(self, *args, **kwargs)

    tf.keras.layers.Layer.__init__ = patched_layer_init


def load_artifacts():
    global model, preprocessing, label_to_id, id_to_label, metadata

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modèle introuvable: {MODEL_PATH}")

    patch_keras_loading_compatibility()
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    if PREPROCESSING_PATH.exists():
        with open(PREPROCESSING_PATH, "r", encoding="utf-8") as f:
            preprocessing = json.load(f)

    if LABEL_MAP_PATH.exists():
        with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
            label_to_id = json.load(f)
        id_to_label = {int(v): k for k, v in label_to_id.items()}
    else:
        label_to_id = {}
        id_to_label = {}

    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)


def expected_length() -> int:
    return int(preprocessing.get("input_length", 0))


def parse_signal_text(text: str) -> np.ndarray:
    if text is None or not text.strip():
        raise ValueError("Le signal texte est vide")

    cleaned = text.replace(";", ",").replace("\n", ",").replace("\t", ",")
    parts = [item.strip() for item in cleaned.split(",") if item.strip()]
    if not parts:
        raise ValueError("Impossible de lire le signal")

    values = np.array([float(item) for item in parts], dtype=np.float32)
    if expected_length() and len(values) != expected_length():
        raise ValueError(
            f"Longueur invalide: {len(values)} valeurs reçues, {expected_length()} attendues"
        )
    return values


def parse_signal_file(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {path}")

    df = pd.read_csv(path, sep=r"[,;\s]+", engine="python", header=None)
    if df.empty:
        raise ValueError("Fichier vide")

    exp_len = expected_length()

    # Cas 1 : une seule ligne de 96 valeurs
    if df.shape[0] == 1 and df.shape[1] == exp_len:
        return df.iloc[0].to_numpy(dtype=np.float32)

    # Cas 2 : une seule ligne avec label + 96 valeurs
    if df.shape[0] == 1 and df.shape[1] == exp_len + 1:
        return df.iloc[0, 1:].to_numpy(dtype=np.float32)

    # Cas 3 : une seule colonne de 96 valeurs
    if df.shape[1] == 1 and df.shape[0] == exp_len:
        return df.iloc[:, 0].to_numpy(dtype=np.float32)

    # Cas 4 : une seule colonne avec label + 96 valeurs
    if df.shape[1] == 1 and df.shape[0] == exp_len + 1:
        return df.iloc[1:, 0].to_numpy(dtype=np.float32)

    # Cas 5 : fichier de dataset entier ECG200 -> on prend la première ligne
    if df.shape[1] == exp_len + 1 and df.shape[0] > 1:
        return df.iloc[0, 1:].to_numpy(dtype=np.float32)

    if df.shape[1] == exp_len and df.shape[0] > 1:
        return df.iloc[0].to_numpy(dtype=np.float32)

    raise ValueError(
        f"Format de fichier non reconnu. Reçu: {df.shape[0]} lignes x {df.shape[1]} colonnes. "
        f"Attendu: un seul signal de {exp_len} valeurs, éventuellement précédé d'un label."
    )


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    signal = signal.astype(np.float32)
    mean = np.mean(signal)
    std = np.std(signal)
    if std == 0:
        std = 1.0
    return (signal - mean) / std


def preprocess_signal(signal: np.ndarray) -> np.ndarray:
    exp_len = expected_length()
    if exp_len and len(signal) != exp_len:
        raise ValueError(
            f"Longueur invalide: {len(signal)} valeurs reçues, {exp_len} attendues"
        )

    if preprocessing.get("normalization") == "zscore_per_signal":
        signal = normalize_signal(signal)

    input_shape = preprocessing.get("input_shape", [])

    if len(input_shape) == 1:
        signal = signal.reshape(1, input_shape[0])
    elif len(input_shape) == 2:
        signal = signal.reshape(1, input_shape[0], input_shape[1])
    else:
        signal = signal.reshape(1, -1)

    return signal.astype(np.float32)


def classify_signal(signal: np.ndarray) -> dict:
    if model is None:
        load_artifacts()

    batch = preprocess_signal(signal)
    probabilities = model.predict(batch, verbose=0)[0]
    predicted_id = int(np.argmax(probabilities))
    predicted_original_label = id_to_label.get(predicted_id, str(predicted_id))

    probs_by_class = {
        id_to_label.get(i, str(i)): float(probabilities[i])
        for i in range(len(probabilities))
    }

    return {
        "predicted_class_id": predicted_id,
        "predicted_original_label": predicted_original_label,
        "probabilities": probs_by_class,
        "input_length": expected_length(),
        "model_name": preprocessing.get("model_name", metadata.get("selected_model", "unknown")),
    }
