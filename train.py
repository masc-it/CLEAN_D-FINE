"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import argparse

from clean_dfine.config import ExperimentConfig
from clean_dfine.trainer import train


def main(
    args,
) -> None:
    cfg = ExperimentConfig(
        exp_name="dfine-test",
        model="dfine-det-s",
        batch_size=2,
        device="mps",
        out_dir="runs/exp",
    )

    train(cfg)


if __name__ == "__main__":
    """ parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config", type=str, required=True)

    args = parser.parse_args() """

    main(None)
