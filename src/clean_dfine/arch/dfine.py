"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import torch.nn as nn

from clean_dfine.arch.dfine_decoder import DFINETransformer


__all__ = [
    "DFINE",
]


class DFINE(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: DFINETransformer,
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder

    def forward(self, x, targets=None):
        x = self.backbone(x)
        x = self.encoder(x)
        x = self.decoder(x, targets)

        return x

    def deploy(
        self,
    ):
        self.eval()
        for m in self.modules():
            if hasattr(m, "convert_to_deploy"):
                m.convert_to_deploy()
        return self
