"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch


__all__ = [
    "YOLO",
]


class YOLO(torch.nn.Module):
    def __init__(self, backbone: torch.nn.Module, neck, head):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head

    def forward(self, x, **kwargs):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x

    def deploy(
        self,
    ):
        self.eval()
        for m in self.modules():
            if m is not self and hasattr(m, "deploy"):
                m.deploy()
        return self
