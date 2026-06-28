from __future__ import annotations

import unittest


try:
    import torch

    from src.models.cnn import SpectrogramCNN
    from src.models.spectrogram_transformer import SpectrogramTransformer
except Exception as exc:  # pragma: no cover - dependency availability controls skip
    torch = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@unittest.skipIf(torch is None, f"PyTorch unavailable: {IMPORT_ERROR}")
class ModelShapeTests(unittest.TestCase):
    def test_cnn_output_shape(self) -> None:
        model = SpectrogramCNN(num_classes=10)
        output = model(torch.randn(2, 1, 128, 173))
        self.assertEqual(tuple(output.shape), (2, 10))

    def test_transformer_output_shape(self) -> None:
        model = SpectrogramTransformer(num_classes=10, embed_dim=64, depth=1, num_heads=4)
        output = model(torch.randn(2, 1, 128, 173))
        self.assertEqual(tuple(output.shape), (2, 10))


if __name__ == "__main__":
    unittest.main()
