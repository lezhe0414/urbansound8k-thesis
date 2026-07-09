from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - exercised only without deps
    raise RuntimeError("PyTorch is required. Install dependencies with `pip install -r requirements.txt`.") from exc


class SpectrogramCNN(nn.Module):
    """Compact CNN baseline for Mel-spectrogram image classification."""

    def __init__(self, in_channels: int = 1, num_classes: int = 10, dropout: float = 0.25) -> None:
        super().__init__()
        self.features = nn.Sequential(
            self._block(in_channels, 32),
            self._block(32, 64),
            self._block(64, 128),
            self._block(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    @staticmethod
    def _block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


class ResidualBlock(nn.Module):
    """Small residual block for spectrogram CNN experiments."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.main(x) + self.shortcut(x))


class SpectrogramResNetCNN(nn.Module):
    """ResNet-style CNN backbone for Mel-spectrogram classification."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        dropout: float = 0.35,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.features = nn.Sequential(
            ResidualBlock(base_channels, base_channels),
            ResidualBlock(base_channels, base_channels * 2, stride=2),
            ResidualBlock(base_channels * 2, base_channels * 2),
            ResidualBlock(base_channels * 2, base_channels * 4, stride=2),
            ResidualBlock(base_channels * 4, base_channels * 4),
            ResidualBlock(base_channels * 4, base_channels * 8, stride=2),
            ResidualBlock(base_channels * 8, base_channels * 8),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(base_channels * 8, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)
