from __future__ import annotations

import tempfile
import unittest
from pathlib import Path


try:
    import numpy as np
    import pandas as pd
    import torch  # noqa: F401

    from src.data import UrbanSound8KMelDataset, build_feature_channels, temporal_delta
except Exception as exc:  # pragma: no cover - dependency availability controls skip
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@unittest.skipIf(IMPORT_ERROR is not None, f"PyTorch unavailable: {IMPORT_ERROR}")
class ProcessedDatasetTests(unittest.TestCase):
    def test_delta_channels_preserve_shape_and_capture_time_slope(self) -> None:
        mel = np.tile(np.arange(12, dtype=np.float32), (4, 1))
        delta = temporal_delta(mel, width=2)
        channels = build_feature_channels(mel, "mel_delta_delta")

        self.assertEqual(channels.shape, (3, 4, 12))
        self.assertTrue(np.allclose(delta[:, 2:-2], 1.0))
        self.assertTrue(np.isfinite(channels).all())

    def test_fold_split_and_tensor_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for fold in range(1, 11):
                fold_dir = root / f"fold{fold}"
                fold_dir.mkdir()
                path = fold_dir / f"item-{fold}.npz"
                np.savez_compressed(path, mel=np.zeros((128, 173), dtype=np.float32))
                rows.append(
                    {
                        "slice_file_name": f"item-{fold}.wav",
                        "fold": fold,
                        "classID": fold % 10,
                        "class": "synthetic",
                        "path": str(path.relative_to(root)),
                    }
                )
            pd.DataFrame(rows).to_csv(root / "metadata.csv", index=False)

            dataset = UrbanSound8KMelDataset(root, split="test", test_fold=10)
            x, y = dataset[0]
            self.assertEqual(tuple(x.shape), (1, 128, 173))
            self.assertEqual(int(y.item()), 0)

            limited = UrbanSound8KMelDataset(root, split="train", test_fold=10, max_samples=3, preload=True)
            self.assertEqual(len(limited), 3)
            x, _ = limited[0]
            self.assertEqual(tuple(x.shape), (1, 128, 173))

            delta_dataset = UrbanSound8KMelDataset(
                root,
                split="test",
                test_fold=10,
                feature_representation="mel_delta_delta",
            )
            x, _ = delta_dataset[0]
            self.assertEqual(tuple(x.shape), (3, 128, 173))


if __name__ == "__main__":
    unittest.main()
