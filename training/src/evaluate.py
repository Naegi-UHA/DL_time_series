"""Creates a comparison file for the trained models.

Each model saves a summary after training. This script reads those summaries
and writes one CSV file to compare the main results more easily.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from paths import METRICS_DIR, MODELS_DIR


def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for summary_path in sorted(MODELS_DIR.glob("*/summary.json")):
        summary = read_json(summary_path)

        rows.append(
            {
                "model": summary["model_name"],
                "classes": str(summary.get("class_labels", [])),
                "params": summary["num_parameters"],
                "train_time_s": summary["train_time_seconds"],
                "infer_test_s": summary["test_inference_seconds"],
                "val_accuracy": summary["validation_metrics"]["accuracy"],
                "val_f1": summary["validation_metrics"]["f1"],
                "test_accuracy": summary["test_metrics"]["accuracy"],
                "test_precision": summary["test_metrics"]["precision"],
                "test_recall": summary["test_metrics"]["recall"],
                "test_f1": summary["test_metrics"]["f1"],
            }
        )

    if not rows:
        raise FileNotFoundError("Aucun summary.json trouvé dans training/outputs/models/")

    df = pd.DataFrame(rows).sort_values(by="test_f1", ascending=False)
    df.to_csv(METRICS_DIR / "comparison.csv", index=False)
    df.to_json(METRICS_DIR / "comparison.json", orient="records", indent=2)

    print(df.to_string(index=False))
    print(f"\nComparaison enregistrée dans : {METRICS_DIR}")


if __name__ == "__main__":
    main()
