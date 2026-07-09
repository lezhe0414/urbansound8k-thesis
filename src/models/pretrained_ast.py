from __future__ import annotations

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError as exc:  # pragma: no cover - exercised only without deps
    raise RuntimeError("PyTorch is required. Install dependencies with `pip install -r requirements.txt`.") from exc


class PretrainedASTClassifier(nn.Module):
    """Fine-tune a pretrained Audio Spectrogram Transformer on cached Mel features."""

    def __init__(
        self,
        pretrained_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
        num_classes: int = 10,
        num_mel_bins: int = 128,
        max_length: int = 1024,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        try:
            from transformers import ASTForAudioClassification
        except ImportError as exc:  # pragma: no cover - exercised only without deps
            raise RuntimeError("Install `transformers` to use the pretrained AST baseline.") from exc

        self.num_mel_bins = int(num_mel_bins)
        self.max_length = int(max_length)
        self.model = ASTForAudioClassification.from_pretrained(
            pretrained_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
        if freeze_encoder:
            for name, parameter in self.model.named_parameters():
                if not name.startswith("classifier"):
                    parameter.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.squeeze(1)
        x = F.interpolate(
            x.unsqueeze(1),
            size=(self.num_mel_bins, self.max_length),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        input_values = x.transpose(1, 2)
        return self.model(input_values=input_values).logits
