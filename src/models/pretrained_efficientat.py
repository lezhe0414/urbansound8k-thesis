from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.parse import urlparse

import torch
from torch import nn
from torch.hub import download_url_to_file

from src.vendor.efficientat.mn.model import get_model
from src.vendor.efficientat.preprocess import AugmentMelSTFT


EFFICIENTAT_UPSTREAM_COMMIT = "a425fdce92572e602a1d5634799bd9f1f2efa806"
MN10_AUDIOSET_CHECKPOINT = (
    "https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/"
    "mn10_as_mAP_471.pt"
)
MN20_AUDIOSET_CHECKPOINT = (
    "https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/"
    "mn20_as_mAP_478.pt"
)
MN30_AUDIOSET_CHECKPOINT = (
    "https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/"
    "mn30_as_mAP_482.pt"
)
MN40_AUDIOSET_CHECKPOINT = (
    "https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/"
    "mn40_as_mAP_484.pt"
)
EFFICIENTAT_VARIANTS = {
    "mn10_as": {"width_mult": 1.0, "checkpoint_url": MN10_AUDIOSET_CHECKPOINT},
    "mn20_as": {"width_mult": 2.0, "checkpoint_url": MN20_AUDIOSET_CHECKPOINT},
    "mn30_as": {"width_mult": 3.0, "checkpoint_url": MN30_AUDIOSET_CHECKPOINT},
    "mn40_as": {"width_mult": 4.0, "checkpoint_url": MN40_AUDIOSET_CHECKPOINT},
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older supported PyTorch
        state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict):
        raise ValueError(f"Expected a state dict in EfficientAT checkpoint: {path}")
    return state


class PretrainedEfficientATClassifier(nn.Module):
    """AudioSet-pretrained EfficientAT encoder with an UrbanSound8K head."""

    def __init__(
        self,
        num_classes: int = 10,
        variant: str = "mn10_as",
        checkpoint_url: str | None = None,
        width_mult: float | None = None,
        model_cache_dir: str | Path = ".model_cache/efficientat",
        pretrained: bool = True,
        stage: str = "linear_probe",
        partial_last_blocks: int = 2,
        freeze_encoder_batchnorm: bool = False,
        frontend_augmentation: bool = False,
        sample_rate: int = 32_000,
        win_length: int = 800,
        hop_size: int = 320,
        n_fft: int = 1024,
        n_mels: int = 128,
        fmin: float = 0.0,
        fmax: float | None = None,
        frequency_mask_param: int = 0,
        time_mask_param: int = 0,
        fmin_aug_range: int = 1,
        fmax_aug_range: int = 1,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.variant = str(variant)
        if self.variant not in EFFICIENTAT_VARIANTS:
            raise ValueError(f"Unsupported EfficientAT variant: {self.variant}")
        variant_config = EFFICIENTAT_VARIANTS[self.variant]
        self.checkpoint_url = str(checkpoint_url or variant_config["checkpoint_url"])
        self.width_mult = float(
            variant_config["width_mult"] if width_mult is None else width_mult
        )
        self.model_cache_dir = Path(model_cache_dir)
        self.freeze_encoder_batchnorm = bool(freeze_encoder_batchnorm)
        self.frontend_augmentation = bool(frontend_augmentation)

        self.frontend = AugmentMelSTFT(
            n_mels=int(n_mels),
            sr=int(sample_rate),
            win_length=int(win_length),
            hopsize=int(hop_size),
            n_fft=int(n_fft),
            freqm=int(frequency_mask_param),
            timem=int(time_mask_param),
            fmin=float(fmin),
            fmax=None if fmax is None else float(fmax),
            fmin_aug_range=int(fmin_aug_range),
            fmax_aug_range=int(fmax_aug_range),
        )
        self.backbone = get_model(
            num_classes=self.num_classes,
            pretrained_name=None,
            width_mult=self.width_mult,
            head_type="mlp",
            input_dim_f=int(n_mels),
            input_dim_t=500,
            se_dims="c",
        )
        self.checkpoint_path: Path | None = None
        self.checkpoint_sha256: str | None = None
        if pretrained:
            self.checkpoint_path = self._download_checkpoint()
            self.checkpoint_sha256 = _sha256(self.checkpoint_path)
            self._load_audio_set_weights(self.checkpoint_path)

        self.stage = ""
        self.partial_last_blocks = int(partial_last_blocks)
        self.set_training_stage(stage, partial_last_blocks=self.partial_last_blocks)

    @property
    def classification_head(self) -> nn.Linear:
        head = self.backbone.classifier[-1]
        if not isinstance(head, nn.Linear):
            raise TypeError("EfficientAT MLP head did not end in a Linear layer.")
        return head

    def _download_checkpoint(self) -> Path:
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        filename = Path(urlparse(self.checkpoint_url).path).name
        checkpoint_path = self.model_cache_dir / filename
        if not checkpoint_path.exists():
            download_url_to_file(self.checkpoint_url, checkpoint_path, progress=True)
        return checkpoint_path

    def _load_audio_set_weights(self, checkpoint_path: Path) -> None:
        state_dict = _load_state_dict(checkpoint_path)
        state_dict.pop("classifier.5.weight", None)
        state_dict.pop("classifier.5.bias", None)
        incompatible = self.backbone.load_state_dict(state_dict, strict=False)
        expected_missing = {"classifier.5.weight", "classifier.5.bias"}
        if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                f"EfficientAT checkpoint did not match the pinned {self.variant} architecture: "
                f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
            )

    def set_training_stage(self, stage: str, partial_last_blocks: int | None = None) -> None:
        stage = str(stage).lower()
        if stage not in {"linear_probe", "partial_finetune"}:
            raise ValueError("stage must be 'linear_probe' or 'partial_finetune'.")
        if partial_last_blocks is not None:
            self.partial_last_blocks = int(partial_last_blocks)
        if not 1 <= self.partial_last_blocks <= len(self.backbone.features):
            raise ValueError("partial_last_blocks is outside the EfficientAT feature stack.")

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        for parameter in self.classification_head.parameters():
            parameter.requires_grad = True

        if stage == "partial_finetune":
            for block in self.backbone.features[-self.partial_last_blocks :]:
                for parameter in block.parameters():
                    parameter.requires_grad = True
        self.stage = stage
        self.train(self.training)

    def parameter_counts(self) -> dict[str, int]:
        all_parameters = list(self.parameters())
        trainable = sum(parameter.numel() for parameter in all_parameters if parameter.requires_grad)
        total = sum(parameter.numel() for parameter in all_parameters)
        return {"trainable": trainable, "frozen": total - trainable, "total": total}

    def optimizer_parameter_groups(self, encoder_lr: float, head_lr: float) -> list[dict]:
        head_parameters = [parameter for parameter in self.classification_head.parameters() if parameter.requires_grad]
        head_ids = {id(parameter) for parameter in head_parameters}
        encoder_parameters = [
            parameter
            for parameter in self.backbone.parameters()
            if parameter.requires_grad and id(parameter) not in head_ids
        ]
        groups: list[dict] = []
        if encoder_parameters:
            groups.append({"params": encoder_parameters, "lr": float(encoder_lr), "group_name": "encoder"})
        groups.append({"params": head_parameters, "lr": float(head_lr), "group_name": "head"})
        return groups

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            if not self.frontend_augmentation:
                self.frontend.eval()
            for module in self.backbone.modules():
                parameters = list(module.parameters(recurse=True))
                if parameters and not any(parameter.requires_grad for parameter in parameters):
                    module.eval()
            if self.freeze_encoder_batchnorm:
                for module in self.backbone.features.modules():
                    if isinstance(module, nn.modules.batchnorm._BatchNorm):
                        module.eval()
        return self

    def waveform_to_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim == 3 and waveform.size(1) == 1:
            waveform = waveform.squeeze(1)
        if waveform.ndim != 2:
            raise ValueError(f"Expected waveform shape [batch, samples], received {tuple(waveform.shape)}")
        with torch.autocast(device_type=waveform.device.type, enabled=False):
            return self.frontend(waveform.float())

    def forward_mel(self, mel: torch.Tensor) -> torch.Tensor:
        if mel.ndim != 3:
            raise ValueError(f"Expected log-Mel shape [batch, mel, time], received {tuple(mel.shape)}")
        logits, _ = self.backbone(mel.unsqueeze(1))
        return logits

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.forward_mel(self.waveform_to_mel(waveform))
