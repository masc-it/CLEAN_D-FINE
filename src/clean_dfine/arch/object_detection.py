from abc import ABC, abstractmethod
import numpy as np
from pydantic import BaseModel
from PIL import Image
from PIL.ImageDraw import ImageDraw


class BBox(BaseModel):
    xmin: int
    ymin: int
    width: int
    height: int
    xmax: int
    ymax: int
    rel_xmin: float = -1
    rel_ymin: float = -1
    rel_xmax: float = -1
    rel_ymax: float = -1
    rel_width: float = -1
    rel_height: float = -1

    label_idx: int
    label: str
    score: float = -1

    def to_list_xyxy(self):
        output = [self.xmin, self.ymin, self.xmax, self.ymax, self.label_idx]
        if self.score > -1:
            output += [self.score]
        return output

    def to_numpy_xyxy(self):
        return np.asarray(self.to_list_xyxy())


class DetModel(ABC):
    def __init__(self, model) -> None:
        self.model = model

    @abstractmethod
    def predict(self, img: Image.Image) -> list[BBox]:
        raise NotImplementedError()

    @abstractmethod
    def predict_batch(self, imgs: list[Image.Image | None]) -> list[list[BBox]]:
        raise NotImplementedError()

    def show(self, img: Image.Image, preds: list[BBox], ground_truth: list[BBox] = []):
        """green: ground truth, blue: preds"""
        canvas = img.copy()
        draw = ImageDraw(canvas)
        for pred in preds:
            draw.rectangle(
                ((pred.xmin, pred.ymin), (pred.xmax, pred.ymax)),
                outline="blue",
                width=2,
            )

        for pred in ground_truth:
            draw.rectangle(
                ((pred.xmin, pred.ymin), (pred.xmax, pred.ymax)),
                outline="green",
                width=2,
            )
        return canvas
