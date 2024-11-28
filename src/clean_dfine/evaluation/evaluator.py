from tqdm import tqdm
from torch.utils.data import DataLoader
from clean_dfine.arch.object_detection import BBox, DetModel
from clean_dfine.evaluation.metrics import MeanAveragePrecision


def start_eval(
    model: DetModel,
    dataloader: DataLoader,
):
    all_preds: list[list[BBox]] = []
    all_targets: list[list[BBox]] = []
    for images, targets in tqdm(dataloader, total=len(dataloader)):
        imgs_preds = model.predict_batch(images)
        all_preds.extend(imgs_preds)
        all_targets.extend(targets["boxes"])

    # prepare data
    # compute metrics
    eval_results = []

    for img_preds, img_targets in zip(all_preds, all_targets):
        preds_np = [bbox.to_numpy_xyxy() for bbox in img_preds]
        targets_np = [bbox.to_numpy_xyxy() for bbox in img_targets]

        map_report = MeanAveragePrecision.from_tensors(preds_np, targets_np)
        eval_results.append(map_report)

    return eval_results
