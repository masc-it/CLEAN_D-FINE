import torch
from clean_dfine.arch.object_detection import BBox


def convert_dataset_targets_to_bbox(targets: list[dict[str, float]]):
    boxes = [
        BBox(
            xmin=int(targ),
            ymin=int(box[1]),
            xmax=int(box[2]),
            ymax=int(box[-1]),
            width=int(box[2] - box[0]),
            height=int(box[-1] - box[1]),
            label="",
            label_idx=label,
            score=score,
        )
        for target in targets
    ]
