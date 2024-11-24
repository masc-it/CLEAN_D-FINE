from tqdm import tqdm
from clean_dfine.dataset import DataLoader
from clean_dfine.evaluation.object_detection import DetModel


def start_eval(
    model: DetModel,
    dataloader: DataLoader,
):
    preds = []
    for images, targets in tqdm(dataloader, total=len(dataloader)):
        imgs_preds = model.predict_batch(images)
        # convert targets to bbox as well..
        preds.extend(imgs_preds)
    # compute metrics
    # return metrics
    pass
