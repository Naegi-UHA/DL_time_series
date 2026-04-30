"""Quick visual check for the ECG200 dataset.

It prints basic dataset information and saves simple plots to check the signal
examples and the class distribution before training.
"""

from __future__ import annotations

from collections import Counter

import matplotlib.pyplot as plt

from .data_utils import load_ecg200
from .paths import FIGURES_DIR


def main() -> None:
    dataset = load_ecg200()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    train_labels = [int(label) for label in dataset["y_train"]]
    test_labels = [int(label) for label in dataset["y_test"]]
    train_counts = Counter(train_labels)
    test_counts = Counter(test_labels)

    print("=== ECG200 ===")
    print(f"Train shape: {dataset['X_train'].shape}")
    print(f"Test shape:  {dataset['X_test'].shape}")
    print(f"Input length: {dataset['input_length']}")
    print(f"Classes in TSV: {dataset['class_labels']}")
    print(f"Model encoding: {dataset['label_to_model_id']}")
    print(f"Train classes: {train_counts}")
    print(f"Test classes:  {test_counts}")

    fig = plt.figure(figsize=(10, 5))
    already_shown = set()

    for signal, label in zip(dataset["X_train"], train_labels):
        if label not in already_shown:
            plt.plot(signal, label=f"classe {label}")
            already_shown.add(label)

        if len(already_shown) == dataset["num_classes"]:
            break

    plt.title("Exemples de signaux ECG")
    plt.xlabel("Temps")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "ecg_examples.png")
    plt.close(fig)

    fig = plt.figure(figsize=(6, 4))
    labels = dataset["class_labels"]
    values = [train_counts[label] for label in labels]

    plt.bar([str(label) for label in labels], values)
    plt.title("Répartition des classes (train)")
    plt.xlabel("Classe")
    plt.ylabel("Nombre")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "class_distribution_train.png")
    plt.close(fig)

    print(f"Figures générées dans : {FIGURES_DIR}")


if __name__ == "__main__":
    main()
