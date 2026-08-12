from __future__ import annotations
from typing import Any
from torch import nn

from .cnn_branch import CNNBranch
from .fusion import FusionBlock
from .regressor import Regressor
from .stem import Stem
from .transformer_branch import TransformerBranch


class PCTN(nn.Module):
    """
    Parallel CNN-Transformer Network for PPG-based cuff-less BP estimation.

    Architecture (paper Fig. 1):
      Stem -> [CNN branch || Transformer branch] -> Fusion -> Regressor
    """

    def __init__(
        self,
        stem_cfg: dict[str, Any] | None = None,
        cnn_cfg: dict[str, Any] | None = None,
        transformer_cfg: dict[str, Any] | None = None,
        regressor_cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        stem_cfg = stem_cfg or {}
        cnn_cfg = cnn_cfg or {}
        transformer_cfg = transformer_cfg or {}
        regressor_cfg = regressor_cfg or {}

        self.stem = Stem(**stem_cfg)
        stem_out = stem_cfg.get("out_channels", 64)

        self.cnn = CNNBranch(
            in_channels=stem_out,
            layers=cnn_cfg.get("layers"),
            channels=cnn_cfg.get("channels"),
        )
        self.transformer = TransformerBranch(
            in_channels=transformer_cfg.get("in_channels", 512),
            num_layers=transformer_cfg.get("num_layers", 6),
            num_heads=transformer_cfg.get("num_heads", 4),
            mlp_ratio=transformer_cfg.get("mlp_ratio", 4),
            dropout=transformer_cfg.get("dropout", 0.1),
        )
        self.fusion = FusionBlock(self.cnn.out_channels, self.transformer.out_channels)
        self.regressor = Regressor(
            in_channels=self.fusion.out_channels,
            hidden_dim=regressor_cfg.get("hidden_dim", 128),
            output_dim=regressor_cfg.get("output_dim", 2),
        )

    def forward(self, x):
        x = self.stem(x)
        cnn_feat = self.cnn(x)
        trans_feat = self.transformer(x)

        min_len = min(cnn_feat.shape[-1], trans_feat.shape[-1])
        cnn_feat = cnn_feat[:, :, :min_len]
        trans_feat = trans_feat[:, :, :min_len]

        fused = self.fusion(cnn_feat, trans_feat)
        return self.regressor(fused)


def build_pctn(model_cfg: dict[str, Any]) -> PCTN:
    return PCTN(
        stem_cfg=model_cfg.get("stem", {}),
        cnn_cfg=model_cfg.get("cnn", {}),
        transformer_cfg=model_cfg.get("transformer", {}),
        regressor_cfg=model_cfg.get("regressor", {}),
    )
