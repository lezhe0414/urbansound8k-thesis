from __future__ import annotations

import ast
import tempfile
import unittest
import wave
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PretrainedCNNConfigTests(unittest.TestCase):
    def test_checkpoint_group_evaluation_names_confusion_matrix(self) -> None:
        tree = ast.parse((ROOT / "src" / "evaluate_pretrained_cnn.py").read_text(encoding="utf-8"))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "save_confusion_matrix"
        ]
        self.assertEqual(len(calls), 1)
        self.assertIn("title", {keyword.arg for keyword in calls[0].keywords})

    def test_linear_probe_protocol_seals_fold_10(self) -> None:
        try:
            import yaml
        except ImportError as exc:
            self.skipTest(f"PyYAML unavailable: {exc}")
        config = yaml.safe_load((ROOT / "configs" / "pretrained_cnn_linear_probe.yaml").read_text(encoding="utf-8"))
        self.assertEqual(config["data"]["development_folds"], [1, 4, 7])
        self.assertEqual(config["data"]["sealed_test_fold"], 10)
        self.assertFalse(config["evaluation"]["locked_for_test"])
        self.assertEqual(config["data"]["sample_rate"], 32_000)
        self.assertNotIn("processed_dir", config["data"])

    def test_partial_finetune_uses_differential_learning_rates(self) -> None:
        try:
            import yaml
        except ImportError as exc:
            self.skipTest(f"PyYAML unavailable: {exc}")
        config = yaml.safe_load((ROOT / "configs" / "pretrained_cnn_partial_finetune.yaml").read_text(encoding="utf-8"))
        self.assertEqual(config["model"]["stage"], "partial_finetune")
        self.assertAlmostEqual(config["training"]["encoder_learning_rate"], 1e-5)
        self.assertAlmostEqual(config["training"]["head_learning_rate"], 3e-4)
        self.assertIn("{val_fold}", config["training"]["initial_checkpoint_template"])

    def test_v2_candidates_are_development_only_and_keep_common_control(self) -> None:
        try:
            import yaml
        except ImportError as exc:
            self.skipTest(f"PyYAML unavailable: {exc}")
        names = [
            "pretrained_cnn_v2_epochs8.yaml",
            "pretrained_cnn_v2_gradual.yaml",
            "pretrained_cnn_v2_masking.yaml",
            "pretrained_cnn_v2_mixup.yaml",
        ]
        configs = [yaml.safe_load((ROOT / "configs" / name).read_text(encoding="utf-8")) for name in names]
        for config in configs:
            self.assertEqual(config["data"]["development_folds"], [1, 4, 7])
            self.assertEqual(config["data"]["sealed_test_fold"], 10)
            self.assertFalse(config["evaluation"]["locked_for_test"])
            self.assertEqual(config["training"]["epochs"], 8)
            self.assertAlmostEqual(config["training"]["encoder_learning_rate"], 2e-5)
            self.assertAlmostEqual(config["training"]["head_learning_rate"], 3e-4)
            self.assertEqual(config["model"]["partial_last_blocks"], 2)

        self.assertTrue(configs[1]["training"]["gradual_unfreezing"]["enabled"])
        self.assertEqual(configs[1]["training"]["gradual_unfreezing"]["head_only_epochs"], 2)
        self.assertTrue(configs[2]["model"]["frontend_augmentation"])
        self.assertEqual(configs[2]["model"]["frequency_mask_param"], 8)
        self.assertEqual(configs[2]["model"]["time_mask_param"], 24)
        self.assertTrue(configs[3]["training"]["mixup"]["enabled"])
        self.assertAlmostEqual(configs[3]["training"]["mixup"]["alpha"], 0.15)

    def test_recommended_candidates_preserve_development_protocol(self) -> None:
        try:
            import yaml
        except ImportError as exc:
            self.skipTest(f"PyYAML unavailable: {exc}")
        names = [
            "pretrained_cnn_recommended_shift_gain.yaml",
            "pretrained_cnn_recommended_noise.yaml",
            "pretrained_cnn_recommended_combo.yaml",
        ]
        configs = [yaml.safe_load((ROOT / "configs" / name).read_text(encoding="utf-8")) for name in names]
        for config in configs:
            self.assertEqual(config["data"]["development_folds"], [1, 4, 7])
            self.assertEqual(config["data"]["sealed_test_fold"], 10)
            self.assertFalse(config["evaluation"]["locked_for_test"])
            self.assertEqual(config["training"]["epochs"], 8)
            self.assertEqual(config["model"]["variant"], "mn10_as")
            self.assertTrue(config["training"]["waveform_augmentation"]["enabled"])

    def test_mn20_configs_use_official_audio_set_checkpoint(self) -> None:
        try:
            import yaml
        except ImportError as exc:
            self.skipTest(f"PyYAML unavailable: {exc}")
        linear = yaml.safe_load((ROOT / "configs" / "pretrained_cnn_mn20_linear.yaml").read_text())
        partial = yaml.safe_load((ROOT / "configs" / "pretrained_cnn_mn20_partial.yaml").read_text())
        for config in (linear, partial):
            self.assertEqual(config["model"]["variant"], "mn20_as")
            self.assertIn("mn20_as_mAP_478.pt", config["model"]["checkpoint_url"])
            self.assertFalse(config["evaluation"]["locked_for_test"])
        self.assertEqual(linear["model"]["stage"], "linear_probe")
        self.assertEqual(partial["model"]["stage"], "partial_finetune")

    def test_mn20_neighbor_changes_only_unfreeze_depth_and_run_name(self) -> None:
        try:
            import yaml
        except ImportError as exc:
            self.skipTest(f"PyYAML unavailable: {exc}")
        reference = yaml.safe_load((ROOT / "configs" / "pretrained_cnn_mn20_partial.yaml").read_text())
        neighbor = yaml.safe_load(
            (ROOT / "configs" / "pretrained_cnn_mn20_partial_last2.yaml").read_text()
        )
        self.assertEqual(reference["model"]["partial_last_blocks"], 3)
        self.assertEqual(neighbor["model"]["partial_last_blocks"], 2)
        reference["run_name"] = neighbor["run_name"]
        reference["model"]["partial_last_blocks"] = 2
        self.assertEqual(reference, neighbor)

    def test_locked_mn20_config_matches_selected_development_method(self) -> None:
        try:
            import yaml
        except ImportError as exc:
            self.skipTest(f"PyYAML unavailable: {exc}")
        selected = yaml.safe_load(
            (ROOT / "configs" / "pretrained_cnn_mn20_partial_last2.yaml").read_text()
        )
        locked = yaml.safe_load(
            (ROOT / "configs" / "pretrained_cnn_mn20_locked_last2.yaml").read_text()
        )
        self.assertEqual(locked["model"]["variant"], "mn20_as")
        self.assertEqual(locked["model"]["partial_last_blocks"], 2)
        self.assertEqual(locked["seed"], 42)
        self.assertTrue(locked["evaluation"]["locked_for_test"])
        self.assertEqual(
            locked["evaluation"]["tta"],
            {"enabled": False, "offsets_seconds": [0.0]},
        )
        self.assertFalse(locked["training"]["waveform_augmentation"]["enabled"])

        selected["run_name"] = locked["run_name"]
        selected["evaluation"] = locked["evaluation"]
        self.assertEqual(selected, locked)


try:
    import numpy as np
    import pandas as pd
    import torch

    from src.data import UrbanSound8KWaveformDataset
    from src.models.pretrained_efficientat import PretrainedEfficientATClassifier
    from src.train_pretrained_cnn import _add_unfrozen_encoder_group
except Exception as exc:  # pragma: no cover - dependency availability controls skip
    DEPENDENCY_ERROR = exc
else:
    DEPENDENCY_ERROR = None


@unittest.skipIf(DEPENDENCY_ERROR is not None, f"Audio ML dependencies unavailable: {DEPENDENCY_ERROR}")
class PretrainedCNNTransferTests(unittest.TestCase):
    @staticmethod
    def _write_wav(path: Path, sample_rate: int = 16_000, samples: int = 1600) -> None:
        values = (np.sin(np.linspace(0, 20, samples)) * 8000).astype(np.int16)
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes(values.tobytes())

    def test_waveform_dataset_excludes_validation_and_test_folds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = Path(tmp)
            rows = []
            for fold in range(1, 11):
                fold_dir = raw_dir / "audio" / f"fold{fold}"
                fold_dir.mkdir(parents=True)
                filename = f"item-{fold}.wav"
                self._write_wav(fold_dir / filename)
                rows.append(
                    {
                        "slice_file_name": filename,
                        "fold": fold,
                        "classID": fold % 10,
                        "class": f"class-{fold % 10}",
                    }
                )
            (raw_dir / "metadata").mkdir()
            pd.DataFrame(rows).to_csv(raw_dir / "metadata" / "UrbanSound8K.csv", index=False)

            dataset = UrbanSound8KWaveformDataset(
                raw_dir,
                split="train",
                test_fold=10,
                val_fold=4,
                sample_rate=32_000,
                clip_duration_seconds=0.25,
            )
            self.assertNotIn(4, {item.fold for item in dataset.items})
            self.assertNotIn(10, {item.fold for item in dataset.items})
            waveform, target = dataset[0]
            self.assertEqual(tuple(waveform.shape), (8000,))
            self.assertEqual(target.dtype, torch.long)

    def test_linear_probe_only_trains_final_head(self) -> None:
        model = PretrainedEfficientATClassifier(
            pretrained=False,
            stage="linear_probe",
            sample_rate=32_000,
            fmin_aug_range=1,
            fmax_aug_range=1,
        )
        counts = model.parameter_counts()
        self.assertEqual(counts["trainable"], 12_810)
        self.assertEqual(sum(parameter.numel() for parameter in model.classification_head.parameters()), 12_810)
        output = model(torch.randn(2, 32_000))
        self.assertEqual(tuple(output.shape), (2, 10))

    def test_partial_finetune_unfreezes_encoder_tail(self) -> None:
        model = PretrainedEfficientATClassifier(pretrained=False, stage="partial_finetune", partial_last_blocks=2)
        counts = model.parameter_counts()
        self.assertGreater(counts["trainable"], 12_810)
        self.assertGreater(counts["frozen"], 0)

    def test_mn20_is_larger_than_mn10(self) -> None:
        mn10 = PretrainedEfficientATClassifier(pretrained=False, variant="mn10_as", stage="linear_probe")
        mn20 = PretrainedEfficientATClassifier(pretrained=False, variant="mn20_as", stage="linear_probe")
        self.assertGreater(mn20.parameter_counts()["total"], mn10.parameter_counts()["total"])

    def test_frontend_and_backbone_can_be_called_separately_for_mixup(self) -> None:
        model = PretrainedEfficientATClassifier(
            pretrained=False,
            stage="linear_probe",
            sample_rate=32_000,
            fmin_aug_range=1,
            fmax_aug_range=1,
        )
        model.eval()
        waveform = torch.randn(2, 32_000)
        mel = model.waveform_to_mel(waveform)
        self.assertEqual(mel.ndim, 3)
        self.assertEqual(tuple(model.forward_mel(mel).shape), (2, 10))

    def test_partial_stage_can_be_refrozen_for_gradual_unfreezing(self) -> None:
        model = PretrainedEfficientATClassifier(pretrained=False, stage="partial_finetune", partial_last_blocks=2)
        partial_trainable = model.parameter_counts()["trainable"]
        model.set_training_stage("linear_probe")
        self.assertEqual(model.parameter_counts()["trainable"], 12_810)
        model.set_training_stage("partial_finetune", partial_last_blocks=2)
        self.assertEqual(model.parameter_counts()["trainable"], partial_trainable)

    def test_gradual_unfreezing_adds_encoder_optimizer_group(self) -> None:
        model = PretrainedEfficientATClassifier(pretrained=False, stage="linear_probe", partial_last_blocks=2)
        optimizer = torch.optim.AdamW(model.optimizer_parameter_groups(encoder_lr=2e-5, head_lr=3e-4))
        self.assertEqual([group["group_name"] for group in optimizer.param_groups], ["head"])
        model.set_training_stage("partial_finetune", partial_last_blocks=2)
        _add_unfrozen_encoder_group(model, optimizer, encoder_lr=2e-5, head_lr=3e-4)
        self.assertEqual([group["group_name"] for group in optimizer.param_groups], ["head", "encoder"])


if __name__ == "__main__":
    unittest.main()
