"""Creates a comparison file for the trained models.

Each model saves a summary after training. This script reads those summaries,
prints the comparison table, and shows which model performed best.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .paths import METRICS_DIR, MODELS_DIR


def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def collect_model_results() -> list[dict]:
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

    return rows


def sort_results(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(
        by=["test_f1", "test_accuracy", "infer_test_s"],
        ascending=[False, False, True],
    )


def save_comparison(df: pd.DataFrame) -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(METRICS_DIR / "comparison.csv", index=False)
    df.to_json(METRICS_DIR / "comparison.json", orient="records", indent=2)


def print_best_model(df: pd.DataFrame) -> None:
    best = df.iloc[0]

    print()
    print("Meilleur modèle")
    print("---------------")
    print(f"Modèle : {best['model']}")
    print(f"Critère principal : test_f1 = {best['test_f1']:.4f}")
    print(f"Accuracy test : {best['test_accuracy']:.4f}")
    print(f"Precision test : {best['test_precision']:.4f}")
    print(f"Recall test : {best['test_recall']:.4f}")
    print(f"Temps d'inférence test : {best['infer_test_s']:.4f} s")


def main() -> None:
    rows = collect_model_results()

    if not rows:
        raise FileNotFoundError("Aucun summary.json trouvé dans training/outputs/models/")

    df = pd.DataFrame(rows)
    df = sort_results(df)

    save_comparison(df)

    print(df.to_string(index=False))
    print_best_model(df)

    print(f"\nComparaison enregistrée dans : {METRICS_DIR}")


if __name__ == "__main__":
    main()
