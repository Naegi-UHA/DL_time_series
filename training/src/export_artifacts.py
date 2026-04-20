"""Exporte les artefacts du modèle retenu vers deployment/model-store/."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


FILES_TO_COPY = [
    "best_model.keras",
    "preprocessing.json",
    "summary.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exporter un modèle vers deployment/model-store")
    parser.add_argument("--model-name", required=True, choices=["mlp", "cnn", "rnn"])
    parser.add_argument("--source-dir", default="outputs/models")
    parser.add_argument("--target-dir", default="../deployment/model-store")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir) / args.model_name
    target_dir = Path(args.target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    if not source_dir.exists():
        raise FileNotFoundError(f"Dossier source introuvable: {source_dir}")

    for filename in FILES_TO_COPY:
        src = source_dir / filename
        if not src.exists():
            raise FileNotFoundError(f"Fichier manquant: {src}")
        shutil.copy2(src, target_dir / filename)

    summary_path = source_dir / "summary.json"
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    label_map = summary.get("label_to_id", {})
    with open(target_dir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)

    metadata = {
        "selected_model": args.model_name,
        "task": "ecg_binary_classification",
        "classes_count": len(label_map),
    }
    with open(target_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Artefacts exportés vers: {target_dir.resolve()}")


if __name__ == "__main__":
    main()
