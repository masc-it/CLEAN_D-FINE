import torch
from clean_dfine.arch.dfine import DFINE
from clean_dfine.arch.postprocessor import DFINEPostProcessor
from clean_dfine.arch.object_detection import BBox, DetModel


class DFineModel(DetModel):
    """High Level interface to run predictions with DFINE"""

    def __init__(
        self, model: DFINE, posprocessor: DFINEPostProcessor, img_size: int, device: str
    ) -> None:
        self.model = model.eval()
        self.model.decoder.training = False  # :vomit fix her asap
        self.postprocessor = posprocessor
        self.img_size = img_size
        self.device = device

    @torch.inference_mode()
    def predict(self, img: torch.Tensor) -> list[BBox]:
        return []

    @torch.inference_mode()
    def predict_batch(self, imgs: torch.Tensor) -> list[list[BBox]]:
        imgs = imgs.to(self.device, non_blocking=True)
        outputs = self.model(imgs)
        outputs = self.postprocessor(
            outputs,
            torch.tensor(
                [[self.img_size, self.img_size]] * imgs.size(0), device=imgs.device
            ),
        )

        return self._raw_preds_to_bboxes(outputs)

    def _raw_preds_to_bboxes(self, preds: list[dict]) -> list[list[BBox]]:
        processes_preds = []
        for img_preds in preds:
            boxes = img_preds["boxes"].tolist()
            labels = img_preds["labels"].tolist()
            scores = img_preds["scores"].tolist()

            img_bboxes = []
            for box, label, score in zip(boxes, labels, scores):
                bbox = BBox(
                    xmin=int(box[0]),
                    ymin=int(box[1]),
                    xmax=int(box[2]),
                    ymax=int(box[-1]),
                    width=int(box[2] - box[0]),
                    height=int(box[-1] - box[1]),
                    label="",
                    label_idx=label,
                    score=score,
                )
                img_bboxes.append(bbox)
        processes_preds.append(img_bboxes)

        return processes_preds
