from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data import UrbanSound8KWaveformDataset
from src.models.pretrained_efficientat import PretrainedEfficientATClassifier
from src.pretrained_cnn_inference import predict_loader, tta_offsets_samples
from src.utils.metrics import classification_metrics, confusion_matrix_array
from src.utils.plotting import save_confusion_matrix


def _load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # pragma: no cover - PyTorch 2.2 compatibility
        return torch.load(path, map_location=device)


def evaluate_checkpoint_group(
    checkpoint_paths: list[Path],
    split: str,
    test_fold: int,
    val_fold: int,
    output_dir: Path,
    tta_config: dict | None = None,
) -> dict:
    if not checkpoint_paths:
        raise ValueError("At least one checkpoint is required.")
    if split not in {"val", "test"}:
        raise ValueError("split must be 'val' or 'test'.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payloads = [_load_checkpoint(path, device) for path in checkpoint_paths]
    configs = [payload["config"] for payload in payloads]
    reference = configs[0]
    data_config = reference["data"]
    training_config = reference["training"]
    labels = list(range(int(data_config.get("num_classes", 10))))
    dataset = UrbanSound8KWaveformDataset(
        raw_dir=data_config["raw_dir"],
        split=split,
        test_fold=int(test_fold),
        val_fold=int(val_fold),
        sample_rate=int(data_config.get("sample_rate", 32_000)),
        clip_duration_seconds=float(data_config.get("clip_duration_seconds", 5.0)),
        waveform_cache_dir=data_config.get("waveform_cache_dir"),
        require_cache=bool(data_config.get("require_waveform_cache", False)),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(training_config.get("batch_size", 32)),
        shuffle=False,
        num_workers=int(training_config.get("num_workers", 0)),
        pin_memory=bool(training_config.get("pin_memory", True)) and torch.cuda.is_available(),
    )
    offsets = tta_offsets_samples(tta_config, int(data_config.get("sample_rate", 32_000)))
    probability_sets: list[np.ndarray] = []
    expected_targets: np.ndarray | None = None
    variants: list[str] = []
    seeds: list[int] = []
    for path, payload, config in zip(checkpoint_paths, payloads, configs):
        model_config = dict(config["model"])
        model_name = model_config.pop("name")
        if model_name != "pretrained_efficientat":
            raise ValueError(f"Unsupported checkpoint model: {model_name}")
        model = PretrainedEfficientATClassifier(**model_config).to(device)
        model.load_state_dict(payload["model_state"])
        _, _, targets, probabilities = predict_loader(
            model,
            loader,
            device,
            labels,
            mixed_precision=bool(training_config.get("mixed_precision", True)),
            offsets_samples=offsets,
            description=f"{split}:{path.parent.name}",
        )
        if expected_targets is None:
            expected_targets = targets
        elif not np.array_equal(expected_targets, targets):
            raise ValueError("Checkpoint predictions do not use identical target ordering.")
        probability_sets.append(probabilities)
        variants.append(model.variant)
        seeds.append(int(config.get("seed", 42)))

    if len(set(variants)) != 1:
        raise ValueError("Probability ensembles require the same EfficientAT variant.")
    mean_probabilities = np.stack(probability_sets, axis=0).mean(axis=0)
    predictions = mean_probabilities.argmax(axis=1)
    assert expected_targets is not None
    metrics = classification_metrics(expected_targets.tolist(), predictions.tolist(), labels)
    metrics.update(
        {
            "split": split,
            "test_fold": int(test_fold),
            "validation_fold": int(val_fold),
            "checkpoint_count": len(checkpoint_paths),
            "checkpoints": [str(path) for path in checkpoint_paths],
            "seeds": seeds,
            "model_variant": variants[0],
            "tta_offsets_samples": offsets,
            "selected_by": "development_validation_macro_f1" if split == "val" else "locked_formal_protocol",
        }
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        output_dir / "predictions.npz",
        targets=expected_targets,
        probabilities=mean_probabilities,
        predictions=predictions,
    )
    matrix = confusion_matrix_array(expected_targets.tolist(), predictions.tolist(), labels)
    save_confusion_matrix(matrix, [str(label) for label in labels], output_dir / "confusion_matrix.png")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate EfficientAT checkpoints with optional TTA/ensembling.")
    parser.add_argument("--checkpoints", nargs="+", required=True, type=Path)
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--test-fold", required=True, type=int)
    parser.add_argument("--val-fold", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--tta-offsets-seconds", nargs="+", type=float)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tta_config = {
        "enabled": args.tta_offsets_seconds is not None,
        "offsets_seconds": args.tta_offsets_seconds or [0.0],
    }
    evaluate_checkpoint_group(
        checkpoint_paths=args.checkpoints,
        split=args.split,
        test_fold=args.test_fold,
        val_fold=args.val_fold,
        output_dir=args.output_dir,
        tta_config=tta_config,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
