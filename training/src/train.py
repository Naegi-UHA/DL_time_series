from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tensorflow import keras

from .data_utils import load_ecg200, prepare_data
from .paths import MODELS_DIR, resolve_config_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Chemin vers le YAML du modèle")
    return parser.parse_args()


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def plot_history(history: dict, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 4))
    plt.plot(history.get("loss", []), label="train_loss")
    plt.plot(history.get("val_loss", []), label="val_loss")
    plt.legend()
    plt.title("Loss")
    plt.tight_layout()
    fig.savefig(output_dir / "loss.png")
    plt.close(fig)

    fig = plt.figure(figsize=(8, 4))
    plt.plot(history.get("accuracy", []), label="train_accuracy")
    plt.plot(history.get("val_accuracy", []), label="val_accuracy")
    plt.legend()
    plt.title("Accuracy")
    plt.tight_layout()
    fig.savefig(output_dir / "accuracy.png")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config_path = resolve_config_path(args.config)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    seed = int(config.get("seed", 42))
    np.random.seed(seed)
    tf.random.set_seed(seed)

    model_name = config["model_name"]
    dataset = load_ecg200()
    data = prepare_data(
        dataset=dataset,
        model_type=model_name,
        val_size=float(config.get("val_size", 0.2)),
        seed=seed,
    )

    model_module = importlib.import_module(f"training.src.models.{model_name}")
    model = model_module.build_model(data["input_shape"], dataset["num_classes"], config)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=float(config.get("learning_rate", 1e-3))),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    output_dir = MODELS_DIR / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=int(config.get("patience", 10)),
            restore_best_weights=True,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "best_model.keras"),
            save_best_only=True,
            monitor="val_loss",
        ),
    ]

    start = time.perf_counter()
    history = model.fit(
        data["X_train"],
        data["y_train"],
        validation_data=(data["X_val"], data["y_val"]),
        epochs=int(config.get("epochs", 50)),
        batch_size=int(config.get("batch_size", 16)),
        verbose=1,
        callbacks=callbacks,
    )
    train_time = time.perf_counter() - start

    best_model = keras.models.load_model(output_dir / "best_model.keras")

    val_pred = np.argmax(best_model.predict(data["X_val"], verbose=0), axis=1)
    test_start = time.perf_counter()
    test_pred = np.argmax(best_model.predict(data["X_test"], verbose=0), axis=1)
    inference_time = time.perf_counter() - test_start

    summary = {
        "model_name": model_name,
        "config_file": str(config_path),
        "input_shape": list(data["input_shape"]),
        "num_parameters": int(best_model.count_params()),
        "train_time_seconds": train_time,
        "test_inference_seconds": inference_time,
        "test_inference_per_sample": inference_time / len(data["X_test"]),
        "validation_metrics": compute_metrics(data["y_val"], val_pred),
        "test_metrics": compute_metrics(data["y_test"], test_pred),
        "label_to_id": {str(k): v for k, v in dataset["label_to_id"].items()},
        "input_length": dataset["input_length"],
    }

    save_json(output_dir / "summary.json", summary)
    save_json(output_dir / "history.json", history.history)
    save_json(
        output_dir / "preprocessing.json",
        {
            "normalization": "zscore_per_signal",
            "model_name": model_name,
            "input_length": dataset["input_length"],
            "input_shape": list(data["input_shape"]),
        },
    )
    plot_history(history.history, output_dir)

    with open(output_dir / "model_summary.txt", "w", encoding="utf-8") as f:
        best_model.summary(print_fn=lambda line: f.write(line + "\n"))

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nRésultats enregistrés dans : {output_dir}")


if __name__ == "__main__":
    main()
