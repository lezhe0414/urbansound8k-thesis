from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - exercised only without deps
    raise RuntimeError("PyTorch is required. Install dependencies with `pip install -r requirements.txt`.") from exc


class SpectrogramTransformer(nn.Module):
    """Lightweight Transformer-style classifier for spectrogram patches."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        patch_size: tuple[int, int] | list[int] = (16, 16),
        embed_dim: int = 128,
        depth: int = 3,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        max_patches: int = 512,
    ) -> None:
        super().__init__()
        patch_size = tuple(patch_size)
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, max_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self._init_parameters()

    def _init_parameters(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        if x.size(1) + 1 > self.pos_embed.size(1):
            raise ValueError(
                f"Spectrogram produced {x.size(1)} patches, but max_patches is {self.pos_embed.size(1) - 1}."
            )
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed[:, : x.size(1), :]
        x = self.dropout(x)
        x = self.encoder(x)
        x = self.norm(x[:, 0])
        return self.head(x)
