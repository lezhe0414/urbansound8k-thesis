from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data import UrbanSound8KMelDataset
from src.models import build_model
from src.train import _class_names, _device
from src.utils.metrics import classification_metrics, confusion_matrix_array
from src.utils.plotting import save_confusion_matrix


def evaluate_run(run_dir: Path, checkpoint_name: str = "best_model.pt") -> dict[str, float]:
    config_path = run_dir / "config_resolved.json"
    checkpoint_path = run_dir / checkpoint_name
    if not config_path.exists():
        raise FileNotFoundError(f"Missing resolved config: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    resolved = json.loads(config_path.read_text(encoding="utf-8"))
    config = resolved["config"]
    fold = int(resolved["fold"])
    val_fold = int(resolved["val_fold"])
    processed_dir = Path(config["data"]["processed_dir"])
    labels = list(range(int(config["data"].get("num_classes", 10))))
    class_names = _class_names(processed_dir)
    device = _device(str(config["training"].get("device", "auto")))

    test_set = UrbanSound8KMelDataset(processed_dir, split="test", test_fold=fold, val_fold=val_fold)
    test_loader = DataLoader(
        test_set,
        batch_size=int(config["training"].get("batch_size", 32)),
        shuffle=False,
        num_workers=int(config["training"].get("num_workers", 0)),
    )

    model = build_model(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            loss = criterion(logits, targets)
            total_loss += float(loss.item()) * inputs.size(0)
            y_true.extend(targets.cpu().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())

    metrics = classification_metrics(y_true, y_pred, labels)
    metrics["test_loss"] = total_loss / max(len(test_set), 1)
    (run_dir / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    matrix = confusion_matrix_array(y_true, y_pred, labels)
    figures_dir = Path(config.get("outputs", {}).get("figures_dir", "figures"))
    matrix_path = figures_dir / f"{run_dir.name}_evaluation_confusion_matrix.png"
    save_confusion_matrix(matrix, class_names, matrix_path, title=f"{run_dir.name} evaluation")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"Wrote confusion matrix to {matrix_path}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained UrbanSound8K run.")
    parser.add_argument("--run-dir", required=True, help="Run directory under results/.")
    parser.add_argument("--checkpoint", default="best_model.pt", help="Checkpoint file name inside the run directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    evaluate_run(Path(args.run_dir), checkpoint_name=args.checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
