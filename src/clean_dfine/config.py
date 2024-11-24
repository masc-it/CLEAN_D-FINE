from pathlib import Path
from typing import Literal
from pydantic import BaseModel
import yaml


class ExperimentConfig(BaseModel):
    exp_name: str
    model: Literal["dfine-det-s"]
    device: Literal["cuda", "mps", "cpu"]
    batch_size: int
    out_dir: str
    num_epochs: int
    lr0: float
    num_classes: int
    img_size: int

    def _setup_dirs(self):
        Path(self.out_dir).mkdir(exist_ok=True, parents=True)

    def get_out_dir(self):
        return Path(self.out_dir)

    def from_yaml(yaml_path: str | Path):
        yaml_path = Path(yaml_path)

        with open(yaml_path) as fp:
            obj = yaml.safe_load(fp)

        cfg = ExperimentConfig(**obj)
        cfg._setup_dirs()
        return cfg
