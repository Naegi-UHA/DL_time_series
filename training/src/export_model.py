from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .paths import DEPLOYMENT_MODELS_DIR, MODELS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["mlp", "cnn", "rnn"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = MODELS_DIR / args.model
    target_dir = DEPLOYMENT_MODELS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    needed = ["best_model.keras", "summary.json", "preprocessing.json"]
    for name in needed:
        src = source_dir / name
        if not src.exists():
            raise FileNotFoundError(f"Fichier manquant : {src}")
        shutil.copy2(src, target_dir / name)

    with open(source_dir / "summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)

    with open(target_dir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(summary.get("label_to_id", {}), f, indent=2, ensure_ascii=False)

    with open(target_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "selected_model": args.model,
                "task": "ecg_binary_classification",
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Modèle exporté vers : {target_dir}")


if __name__ == "__main__":
    main()
