
from pydantic import BaseModel
from clean_dfine.arch.dfine import DFINE
from clean_dfine.arch.postprocessor import DFINEPostProcessor
from clean_dfine.dataset import DataLoader
from clean_dfine.evaluation.report import DetEvalReport


class DetEvaluator:

    def __init__(self, model: DFINE, postprocessor: DFINEPostProcessor, dataloader: DataLoader) -> None:
        self.model = model.eval()
        self.postprocessor = postprocessor
        self.dataloader = dataloader

    def eval(self) -> DetEvalReport:
        pass