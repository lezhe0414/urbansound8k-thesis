"""Model factory for spectrogram classifiers."""

from src.models.cnn import SpectrogramCNN, SpectrogramResNetCNN
from src.models.spectrogram_transformer import SpectrogramTransformer


def build_model(config: dict):
    model_config = dict(config.get("model", {}))
    name = model_config.pop("name", "cnn")

    if name == "cnn":
        return SpectrogramCNN(**model_config)
    if name == "resnet_cnn":
        return SpectrogramResNetCNN(**model_config)
    if name == "spectrogram_transformer":
        return SpectrogramTransformer(**model_config)

    raise ValueError(f"Unsupported model name: {name}")


__all__ = ["SpectrogramCNN", "SpectrogramResNetCNN", "SpectrogramTransformer", "build_model"]
