from tqdm import tqdm
from torch.utils.data import DataLoader
from clean_dfine.arch.object_detection import BBox, DetModel
from clean_dfine.evaluation.metrics import MeanAveragePrecision
import numpy as np


def start_eval(
    model: DetModel,
    dataloader: DataLoader,
):
    all_preds: list[list[BBox]] = []
    all_targets: list[list[BBox]] = []
    for images, targets in tqdm(dataloader, total=len(dataloader)):
        imgs_preds = model.predict_batch(images)
        all_preds.extend(imgs_preds)
        all_targets.extend([t["boxes"] for t in targets])

    # prepare data
    # compute metrics

    eval_inputs = []
    for img_preds, img_targets in zip(all_preds, all_targets):
        eval_inputs.append(
            (
                np.array([bbox.to_list_xyxy() for bbox in img_preds]),
                np.array([bbox.to_list_xyxy() for bbox in img_targets]),
            )
        )
    map_report = MeanAveragePrecision.from_tensors([el[0] for el in eval_inputs], [el[1] for el in eval_inputs])

    return map_report
