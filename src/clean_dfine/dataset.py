"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

from io import BytesIO
from pathlib import Path
from typing import Any, Literal
import torch
import torch.utils.data as data
import torchvision
from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat, Mask
from torchvision.transforms import Compose
from torchvision import transforms
from datasets import Dataset, load_from_disk, concatenate_datasets
from PIL import Image
import logging

from clean_dfine.arch.object_detection import BBox

_boxes_keys = ['format', 'canvas_size']

class DetDataset(data.Dataset):
    def __init__(self):
        super(DetDataset, self).__init__()

    def __getitem__(self, index):
        raise NotImplementedError(
            "Please implement this function to return item before `transforms`."
        )

    def load_item(self, index):
        raise NotImplementedError(
            "Please implement this function to return item before `transforms`."
        )

    def set_epoch(self, epoch) -> None:
        self._epoch = epoch

    @property
    def epoch(self):
        return self._epoch if hasattr(self, "_epoch") else -1


class HFImageDataset(DetDataset):
    def __init__(
        self,
        ds_data: Dataset,
        ds_images: Dataset,
        img_col: str,
        target_col: str,
        transforms,
        img_size: int,
        target_format: Literal["train", "val"],
    ):
        self.ds = concatenate_datasets(
            [ds_data.select_columns([target_col]), ds_images.select_columns([img_col])],
            axis=1,
        )
        self.img_col = img_col
        self.target_col = target_col
        self.transforms = transforms
        self.img_size = img_size
        self.target_format = target_format
        self.get_item_fn = (
            self._get_item_train if target_format == "train" else self._get_item_eval
        )
        logging.info(f"[DATASET] {self.target_format=}")

    def __getitem__(self, index):
        return self.get_item_fn(index)

    def _get_item_train(self, index: int) -> tuple[torch.Tensor, dict]:
        sample = self.ds[index]
        image = self._load_img_from_bytes(sample[self.img_col])
        target = sample[self.target_col]
        boxes = self._parse_boxes(
            self._resize_bounding_boxes(
                target, image.width, image.height, self.img_size, self.img_size
            )
        )

        img: torch.Tensor = self.transforms(image)
        boxes = convert_to_tv_tensor(
            boxes,
            key="boxes",
            box_format="xyxy",
            spatial_size=img.size()[::-1],
        )
        boxes = torchvision.ops.box_convert(boxes, in_fmt="xyxy", out_fmt="cxcywh")
        boxes = boxes / self.img_size

        output = {"boxes": boxes, "labels": self._parse_labels(target)}
        return img, output

    def _get_item_eval(self, index: int) -> tuple[torch.Tensor, dict[str, list[BBox]]]:
        sample = self.ds[index]
        image = self._load_img_from_bytes(sample[self.img_col])
        target = sample[self.target_col]
        boxes = self._resize_bounding_boxes(
            target, image.width, image.height, self.img_size, self.img_size
        )

        img: torch.Tensor = self.transforms(image)

        output = {"boxes": [BBox(**b) for b in boxes]}
        return img, output

    @staticmethod
    def from_path(ds_path: Path, split: Literal["train", "val"], img_size: int):
        ds_data = load_from_disk((ds_path / "data" / split).as_posix())
        ds_images = load_from_disk((ds_path / "data_images" / split).as_posix())
        assert isinstance(ds_data, Dataset) and isinstance(ds_images, Dataset)
        return HFImageDataset(
            ds_data,
            ds_images,
            "image",
            "gibs",
            Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()]),
            img_size,
            target_format=split,
        )

    def _load_img_from_bytes(self, img_bytes: bytes) -> Image.Image:
        return Image.open(BytesIO(img_bytes)).convert("RGB")

    def _parse_boxes(self, boxes: list[dict[str, Any]]) -> torch.Tensor:
        return (
            torch.tensor(
                [[box["xmin"], box["ymin"], box["xmax"], box["ymax"]] for box in boxes]
            )
            if len(boxes)
            else torch.zeros(0, 4)
        )

    def _resize_bounding_boxes(
        self, bounding_boxes, original_width, original_height, new_width, new_height
    ):
        """
        Resizes bounding boxes based on new image dimensions.

        Args:
            bounding_boxes (list of dict): List of bounding box dictionaries. Each dictionary should have
                'xmin', 'ymin', 'xmax', 'ymax' keys among others.
            original_width (int or float): Width of the original image.
            original_height (int or float): Height of the original image.
            new_width (int or float): Width of the resized image.
            new_height (int or float): Height of the resized image.

        Returns:
            list of dict: List of resized bounding box dictionaries.
        """
        # Calculate scaling factors
        x_scale = new_width / original_width
        y_scale = new_height / original_height

        resized_boxes = []
        for box in bounding_boxes:
            resized_box = box
            resized_box["xmin"] = int(box["xmin"] * x_scale)
            resized_box["xmax"] = int(box["xmax"] * x_scale)
            resized_box["ymin"] = int(box["ymin"] * y_scale)
            resized_box["ymax"] = int(box["ymax"] * y_scale)
            resized_boxes.append(resized_box)

        return resized_boxes

    def _parse_labels(self, boxes: list[dict[str, Any]]) -> torch.Tensor:
        return torch.tensor([box["label_idx"] for box in boxes], dtype=torch.long)

    def __len__(self):
        return len(self.ds)


class DataLoader(data.DataLoader):
    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for n in ["dataset", "batch_size", "num_workers", "drop_last", "collate_fn"]:
            format_string += "\n"
            format_string += "    {0}: {1}".format(n, getattr(self, n))
        format_string += "\n)"
        return format_string

    def set_epoch(self, epoch):
        self._epoch = epoch
        self.dataset.set_epoch(epoch)
        self.collate_fn.set_epoch(epoch)

    @property
    def epoch(self):
        return self._epoch if hasattr(self, "_epoch") else -1

    @property
    def shuffle(self):
        return self._shuffle

    @shuffle.setter
    def shuffle(self, shuffle):
        assert isinstance(shuffle, bool), "shuffle must be a boolean"
        self._shuffle = shuffle


def batch_image_collate_fn_(items):
    """only batch image"""
    return torch.stack([x[0] for x in items]), [x[1] for x in items]


def batch_image_collate_fn(items):
    """only batch image"""
    return torch.cat([x[0][None] for x in items], dim=0), [x[1] for x in items]


class BaseCollateFunction(object):
    def set_epoch(self, epoch):
        self._epoch = epoch

    @property
    def epoch(self):
        return self._epoch if hasattr(self, "_epoch") else -1

    def __call__(self, items):
        raise NotImplementedError("")


def generate_scales(base_size, base_size_repeat):
    scale_repeat = (base_size - int(base_size * 0.75 / 32) * 32) // 32
    scales = [int(base_size * 0.75 / 32) * 32 + i * 32 for i in range(scale_repeat)]
    scales += [base_size] * base_size_repeat
    scales += [int(base_size * 1.25 / 32) * 32 - i * 32 for i in range(scale_repeat)]
    return scales


class BatchImageCollateFunction(BaseCollateFunction):
    def __init__(
        self,
        stop_epoch=None,
        ema_restart_decay=0.9999,
        base_size=640,
        base_size_repeat=None,
    ) -> None:
        super().__init__()
        self.base_size = base_size
        self.scales = (
            generate_scales(base_size, base_size_repeat)
            if base_size_repeat is not None
            else None
        )
        self.stop_epoch = stop_epoch if stop_epoch is not None else 100000000
        self.ema_restart_decay = ema_restart_decay

    def __call__(self, items):
        images = torch.cat([x[0][None] for x in items], dim=0)
        targets = [x[1] for x in items]

        return images, targets


def convert_to_tv_tensor(
    tensor: torch.Tensor, key: str, box_format="xyxy", spatial_size=None
) -> torch.Tensor:
    """
    Args:
        tensor (Tensor): input tensor
        key (str): transform to key

    Return:
        Dict[str, TV_Tensor]
    """
    assert key in (
        "boxes",
        "masks",
    ), "Only support 'boxes' and 'masks'"

    if key == "boxes":
        box_format = getattr(BoundingBoxFormat, box_format.upper())
        _kwargs = dict(zip(_boxes_keys, [box_format, spatial_size]))
        return BoundingBoxes(tensor, **_kwargs)

    if key == "masks":
        return Mask(tensor)
