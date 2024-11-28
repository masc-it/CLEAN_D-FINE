"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import argparse
from pathlib import Path

import torch

from clean_dfine.arch.dfine import DFINE
from clean_dfine.arch.dfine_decoder import DFINETransformer
from clean_dfine.arch.hgnetv2 import HGNetv2
from clean_dfine.arch.hybrid_encoder import HybridEncoder
from clean_dfine.config import ExperimentConfig
from clean_dfine.dataset import (
    DataLoader,
    HFImageDataset,
    batch_image_collate_fn,
)
from clean_dfine.trainer import train


def main(
    args,
) -> None:
    cfg = ExperimentConfig(
        exp_name="dfine-test",
        model="dfine-det-s",
        batch_size=32,
        device="cuda",
        out_dir="runs/exp",
        num_classes=1,
        lr0=8e-4,
        num_epochs=2,
        img_size=512,
    )

    model = _build_dfine(cfg)
    # TODO save postprocessor as well
    #model.load_state_dict(torch.load("data/dfine.pt", weights_only=True, map_location="cpu"))

    print(sum(p.numel() for p in model.parameters()))

    dataset_train = HFImageDataset.from_path(
        Path("./data/it_it"), "train", cfg.img_size
    )
    #dataset_train.ds = dataset_train.ds.shuffle(seed=42).take(2000)
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        collate_fn=batch_image_collate_fn,
    )

    dataset_val = HFImageDataset.from_path(Path("./data/it_it"), "val", cfg.img_size)
    dataset_val.ds = dataset_val.ds.take(512)  # tmp

    dataloader_val = DataLoader(
        dataset_val,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=2,
        persistent_workers=True,
        collate_fn=batch_image_collate_fn,
    )

    train(cfg, model, dataloader_train, dataloader_val)


def _build_dfine(cfg: ExperimentConfig):
    backbone = HGNetv2(
        name="B0",
        return_idx=[2, 3],
        freeze_at=-1,
        freeze_norm=False,
        use_lab=True,
        pretrained=False,
    )
    encoder = HybridEncoder(
        in_channels=[512, 1024],
        feat_strides=[16, 32],
        # intra
        hidden_dim=128,
        use_encoder_idx=[1],
        dim_feedforward=512,
        # cross
        expansion=0.34,
        depth_mult=0.5,
    )

    decoder = DFINETransformer(
        feat_channels=[128, 128],
        feat_strides=[16, 32],
        hidden_dim=128,
        dim_feedforward=512,
        num_queries=300,
        num_levels=2,
        num_layers=3,
        eval_idx=-1,
        num_points=[6, 6],
        num_classes=cfg.num_classes,
    )

    return (
        DFINE(backbone=backbone, encoder=encoder, decoder=decoder)
        .to(cfg.device)
        .train()
    )


if __name__ == "__main__":
    """ parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config", type=str, required=True)

    args = parser.parse_args() """

    main(None)
