from __future__ import annotations

from collections import Counter

import matplotlib.pyplot as plt

from .data_utils import load_ecg200
from .paths import FIGURES_DIR


def main() -> None:
    dataset = load_ecg200()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=== ECG200 ===")
    print(f"Train shape: {dataset['X_train'].shape}")
    print(f"Test shape:  {dataset['X_test'].shape}")
    print(f"Input length: {dataset['input_length']}")
    print(f"Label mapping: {dataset['label_to_id']}")
    print(f"Train classes: {Counter(dataset['y_train'])}")
    print(f"Test classes:  {Counter(dataset['y_test'])}")

    fig = plt.figure(figsize=(10, 5))
    shown = set()
    for x, y in zip(dataset["X_train"], dataset["y_train"]):
        if y not in shown:
            plt.plot(x, label=f"classe {y}")
            shown.add(y)
        if len(shown) == dataset["num_classes"]:
            break
    plt.title("Exemples de signaux ECG")
    plt.xlabel("Temps")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "ecg_examples.png")
    plt.close(fig)

    fig = plt.figure(figsize=(6, 4))
    counts = Counter(dataset["y_train"])
    plt.bar([str(k) for k in counts.keys()], list(counts.values()))
    plt.title("Répartition des classes (train)")
    plt.xlabel("Classe")
    plt.ylabel("Nombre")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "class_distribution_train.png")
    plt.close(fig)

    print(f"Figures générées dans : {FIGURES_DIR}")


if __name__ == "__main__":
    main()
