from __future__ import annotations

from pathlib import Path

import numpy as np


def save_confusion_matrix(matrix: np.ndarray, class_names: list[str], output_path: str | Path, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised only without deps
        raise RuntimeError("matplotlib is required. Install dependencies with `pip install -r requirements.txt`.") from exc

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    threshold = matrix.max() / 2.0 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(
                col,
                row,
                int(matrix[row, col]),
                ha="center",
                va="center",
                color="white" if matrix[row, col] > threshold else "black",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_training_history(history: list[dict], output_path: str | Path, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised only without deps
        raise RuntimeError("matplotlib is required. Install dependencies with `pip install -r requirements.txt`.") from exc

    if not history:
        raise ValueError("Training history must contain at least one epoch.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [int(row["epoch"]) for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [float(row["train_loss"]) for row in history], label="Train")
    axes[0].plot(epochs, [float(row["val_loss"]) for row in history], label="Validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, [float(row["train_f1_macro"]) for row in history], label="Train Macro F1")
    axes[1].plot(epochs, [float(row["val_f1_macro"]) for row in history], label="Validation Macro F1")
    axes[1].plot(epochs, [float(row["val_accuracy"]) for row in history], label="Validation Accuracy")
    axes[1].set_title("Classification metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend(fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
