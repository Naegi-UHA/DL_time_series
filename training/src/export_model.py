"""Exports one trained model for deployment.

It copies the saved model and preprocessing files, then creates the JSON files
needed by the web app to read predictions as the real ECG labels.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .paths import DEPLOYMENT_MODELS_DIR, MODELS_DIR


FILES_TO_COPY = [
    "best_model.keras",
    "summary.json",
    "preprocessing.json",
]

FILES_TO_CREATE = [
    "label_map.json",
    "metadata.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["mlp", "cnn", "rnn"])
    return parser.parse_args()


def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def copy_existing_files(source_dir: Path, target_dir: Path) -> None:
    for filename in FILES_TO_COPY:
        source_file = source_dir / filename

        if not source_file.exists():
            raise FileNotFoundError(f"Fichier manquant : {source_file}")

        shutil.copy2(source_file, target_dir / filename)


def create_export_files(model_name: str, source_dir: Path, target_dir: Path) -> None:
    summary = read_json(source_dir / "summary.json")

    label_map = summary.get("model_id_to_label", {})

    metadata = {
        "selected_model": model_name,
        "task": "ecg_binary_classification",
        "class_labels": summary.get("class_labels", []),
        "label_map_direction": "model_id_to_original_label",
    }

    save_json(target_dir / "label_map.json", label_map)
    save_json(target_dir / "metadata.json", metadata)


def copy_model_files(model_name: str, source_root: Path, target_dir: Path) -> None:
    source_dir = source_root / model_name

    if not source_dir.exists():
        raise FileNotFoundError(f"Dossier source introuvable : {source_dir}")

    target_dir.mkdir(parents=True, exist_ok=True)

    copy_existing_files(source_dir, target_dir)
    create_export_files(model_name, source_dir, target_dir)


def main() -> None:
    args = parse_args()

    copy_model_files(args.model, MODELS_DIR, DEPLOYMENT_MODELS_DIR)

    print(f"Modèle exporté vers : {DEPLOYMENT_MODELS_DIR}")
    print("Fichiers exportés :")

    for filename in FILES_TO_COPY + FILES_TO_CREATE:
        print(f"- {filename}")


if __name__ == "__main__":
    main()
