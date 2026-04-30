"""Trains one ECG model from a YAML config.

The script loads the data, prepares labels for the model, trains the network,
evaluates the best version, and saves the training outputs.
"""

from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml
from tensorflow import keras

from .data_utils import decode_labels, load_ecg200, prepare_data
from .metrics import compute_metrics
from .paths import MODELS_DIR, resolve_config_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to the model YAML config")
    return parser.parse_args()


def read_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def setup_seed(seed: int) -> None:
    # Use the same random seed each time, so results are easier to compare.
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_model(
    model_name: str,
    input_shape: tuple[int, ...],
    num_classes: int,
    config: dict,
) -> keras.Model:
    # Load the selected model file from training/src/models/.
    model_file = importlib.import_module(f"training.src.models.{model_name}")
    return model_file.build_model(input_shape, num_classes, config)


def compile_model(model: keras.Model, learning_rate: float) -> None:
    # The model uses labels 0 and 1 internally.
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def make_callbacks(output_dir: Path, patience: int) -> list[keras.callbacks.Callback]:
    # EarlyStopping stops training when validation loss stops improving.
    # ModelCheckpoint keeps only the best model.
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "best_model.keras"),
            save_best_only=True,
            monitor="val_loss",
        ),
    ]


def train_model(
    model: keras.Model,
    data: dict,
    config: dict,
    output_dir: Path,
) -> tuple[keras.callbacks.History, float]:
    callbacks = make_callbacks(output_dir, patience=int(config.get("patience", 10)))

    start_time = time.perf_counter()

    history = model.fit(
        data["X_train"],
        data["y_train"],
        validation_data=(data["X_val"], data["y_val"]),
        epochs=int(config.get("epochs", 50)),
        batch_size=int(config.get("batch_size", 16)),
        verbose=1,
        callbacks=callbacks,
    )

    train_time = time.perf_counter() - start_time
    return history, train_time


def load_best_model(output_dir: Path) -> keras.Model:
    return keras.models.load_model(output_dir / "best_model.keras")


def predict_original_labels(
    model: keras.Model,
    X: np.ndarray,
    model_id_to_label: dict[int, int],
) -> np.ndarray:
    # The model predicts 0 or 1.
    # This converts predictions back to the real ECG labels: -1 or 1.
    predictions = model.predict(X, verbose=0)
    model_ids = np.argmax(predictions, axis=1)

    return decode_labels(model_ids, model_id_to_label)


def measure_test_predictions(
    model: keras.Model,
    X_test: np.ndarray,
    model_id_to_label: dict[int, int],
) -> tuple[np.ndarray, float]:
    start_time = time.perf_counter()

    predictions = predict_original_labels(model, X_test, model_id_to_label)

    test_time = time.perf_counter() - start_time
    return predictions, test_time


def choose_positive_label(class_labels: list[int]) -> int:
    # In ECG200, the positive class is 1.
    if 1 in class_labels:
        return 1

    return class_labels[-1]


def make_summary(
    model_name: str,
    config_path: Path,
    model: keras.Model,
    dataset: dict,
    data: dict,
    train_time: float,
    test_time: float,
    val_pred: np.ndarray,
    test_pred: np.ndarray,
) -> dict:
    positive_label = choose_positive_label(dataset["class_labels"])

    return {
        "model_name": model_name,
        "config_file": str(config_path),
        "input_shape": list(data["input_shape"]),
        "input_length": dataset["input_length"],
        "num_parameters": int(model.count_params()),
        "train_time_seconds": train_time,
        "test_inference_seconds": test_time,
        "test_inference_per_sample": test_time / len(data["X_test"]),
        "class_labels": dataset["class_labels"],
        "positive_label": positive_label,
        "label_to_model_id": {
            str(label): model_id
            for label, model_id in dataset["label_to_model_id"].items()
        },
        "model_id_to_label": {
            str(model_id): label
            for model_id, label in dataset["model_id_to_label"].items()
        },
        "validation_metrics": compute_metrics(
            data["y_val_original"],
            val_pred,
            positive_label,
        ),
        "test_metrics": compute_metrics(
            data["y_test_original"],
            test_pred,
            positive_label,
        ),
    }


def make_preprocessing_info(
    model_name: str,
    dataset: dict,
    data: dict,
    summary: dict,
) -> dict:
    return {
        "normalization": "zscore_per_signal",
        "model_name": model_name,
        "input_length": dataset["input_length"],
        "input_shape": list(data["input_shape"]),
        "class_labels": dataset["class_labels"],
        "label_to_model_id": summary["label_to_model_id"],
        "model_id_to_label": summary["model_id_to_label"],
    }


def plot_history(history_data: dict, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 4))
    plt.plot(history_data.get("loss", []), label="train_loss")
    plt.plot(history_data.get("val_loss", []), label="val_loss")
    plt.legend()
    plt.title("Loss")
    plt.tight_layout()
    fig.savefig(output_dir / "loss.png")
    plt.close(fig)

    fig = plt.figure(figsize=(8, 4))
    plt.plot(history_data.get("accuracy", []), label="train_accuracy")
    plt.plot(history_data.get("val_accuracy", []), label="val_accuracy")
    plt.legend()
    plt.title("Accuracy")
    plt.tight_layout()
    fig.savefig(output_dir / "accuracy.png")
    plt.close(fig)


def save_model_summary(model: keras.Model, output_dir: Path) -> None:
    with open(output_dir / "model_summary.txt", "w", encoding="utf-8") as file:
        model.summary(print_fn=lambda line: file.write(line + "\n"))


def save_training_outputs(
    output_dir: Path,
    model: keras.Model,
    summary: dict,
    history_data: dict,
    preprocessing: dict,
) -> None:
    save_json(output_dir / "summary.json", summary)
    save_json(output_dir / "history.json", history_data)
    save_json(output_dir / "preprocessing.json", preprocessing)

    plot_history(history_data, output_dir)
    save_model_summary(model, output_dir)


def format_seconds(value: float) -> str:
    if value < 1:
        return f"{value * 1000:.2f} ms"

    return f"{value:.2f} s"


def format_metric(value: float) -> str:
    return f"{value * 100:.2f} %"


def make_table(headers: list[str], rows: list[list[str]]) -> str:
    columns = list(zip(headers, *rows))
    widths = [max(len(str(value)) for value in column) for column in columns]

    separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"

    header_line = "| " + " | ".join(
        header.ljust(width)
        for header, width in zip(headers, widths)
    ) + " |"

    row_lines = []
    for row in rows:
        row_lines.append(
            "| " + " | ".join(
                str(value).ljust(width)
                for value, width in zip(row, widths)
            ) + " |"
        )

    return "\n".join([separator, header_line, separator, *row_lines, separator])


def print_title(title: str) -> None:
    line = "=" * len(title)
    print()
    print(line)
    print(title)
    print(line)


def print_training_report(summary: dict, output_dir: Path) -> None:
    validation = summary["validation_metrics"]
    test = summary["test_metrics"]

    print_title(f"Training finished: {summary['model_name']}")

    general_rows = [
        ["Model", summary["model_name"]],
        ["Input shape", str(summary["input_shape"])],
        ["Classes", str(summary["class_labels"])],
        ["Positive label", str(summary["positive_label"])],
        ["Parameters", f"{summary['num_parameters']:,}".replace(",", " ")],
        ["Training time", format_seconds(summary["train_time_seconds"])],
        ["Test inference", format_seconds(summary["test_inference_seconds"])],
        [
            "Inference / sample",
            format_seconds(summary["test_inference_per_sample"]),
        ],
    ]

    print()
    print("Model info")
    print(make_table(["Item", "Value"], general_rows))

    metric_rows = [
        [
            "Validation",
            format_metric(validation["accuracy"]),
            format_metric(validation["precision"]),
            format_metric(validation["recall"]),
            format_metric(validation["f1"]),
        ],
        [
            "Test",
            format_metric(test["accuracy"]),
            format_metric(test["precision"]),
            format_metric(test["recall"]),
            format_metric(test["f1"]),
        ],
    ]

    print()
    print("Metrics")
    print(make_table(
        ["Dataset", "Accuracy", "Precision", "Recall", "F1-score"],
        metric_rows,
    ))

def main() -> None:
    # Read the YAML config.
    args = parse_args()
    config_path = resolve_config_path(args.config)
    config = read_yaml(config_path)

    # Prepare randomness, paths, dataset and labels.
    seed = int(config.get("seed", 42))
    setup_seed(seed)

    model_name = config["model_name"]
    output_dir = MODELS_DIR / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_ecg200()
    data = prepare_data(
        dataset=dataset,
        model_type=model_name,
        val_size=float(config.get("val_size", 0.2)),
        seed=seed,
    )

    # Create and compile the model.
    model = build_model(
        model_name=model_name,
        input_shape=data["input_shape"],
        num_classes=dataset["num_classes"],
        config=config,
    )
    compile_model(
        model=model,
        learning_rate=float(config.get("learning_rate", 0.001)),
    )

    # Train the model and reload the best saved version.
    history, train_time = train_model(model, data, config, output_dir)
    best_model = load_best_model(output_dir)

    # Predict using the real labels, so results show -1 and 1.
    val_pred = predict_original_labels(
        model=best_model,
        X=data["X_val"],
        model_id_to_label=dataset["model_id_to_label"],
    )
    test_pred, test_time = measure_test_predictions(
        model=best_model,
        X_test=data["X_test"],
        model_id_to_label=dataset["model_id_to_label"],
    )

    # Save JSON files, plots and model summary.
    summary = make_summary(
        model_name=model_name,
        config_path=config_path,
        model=best_model,
        dataset=dataset,
        data=data,
        train_time=train_time,
        test_time=test_time,
        val_pred=val_pred,
        test_pred=test_pred,
    )
    preprocessing = make_preprocessing_info(
        model_name=model_name,
        dataset=dataset,
        data=data,
        summary=summary,
    )

    save_training_outputs(
        output_dir=output_dir,
        model=best_model,
        summary=summary,
        history_data=history.history,
        preprocessing=preprocessing,
    )

    # Print a readable terminal report instead of the full JSON.
    print_training_report(summary, output_dir)


if __name__ == "__main__":
    main()