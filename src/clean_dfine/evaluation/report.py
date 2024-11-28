from pydantic import BaseModel, computed_field

from clean_dfine.arch.object_detection import BBox


class ODEvalScoreReport(BaseModel):
    score_avg: float = -1.0
    score_std: float = -1.0

    score_min: float = -1.0
    score_max: float = -1.0
    score_percentiles: list[float] = []

    score_min_xmin: float = -1
    score_min_ymin: float = -1
    score_min_width: float = -1
    score_min_height: float = -1
    score_min_bbox: BBox | None = None
    score_status: str = "OK"


def mAP_at(maps: list[float], thresh: int):
    maps = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
    return maps[maps.index(thresh)]


class DetEvalReport(BaseModel):
    mAPs: list[float] = [-1.0] * 10

    precisions: list[float] = [-1.0] * 10
    recalls: list[float] = [-1.0] * 10
    map_status: str = "OK"

    @computed_field
    @property
    def mAP_50(self) -> float:
        return self.mAP_at(50)

    @computed_field
    @property
    def mAP_75(self) -> float:
        return self.mAP_at(75)

    @computed_field
    @property
    def mAP_85(self) -> float:
        return self.mAP_at(85)

    @computed_field
    @property
    def mAP_90(self) -> float:
        return self.mAP_at(90)

    @computed_field
    @property
    def mAP_95(self) -> float:
        return self.mAP_at(95)

    @computed_field
    @property
    def p_50(self) -> float:
        return self.p_at(50)

    @computed_field
    @property
    def p_75(self) -> float:
        return self.p_at(75)

    @computed_field
    @property
    def p_85(self) -> float:
        return self.p_at(85)

    @computed_field
    @property
    def p_90(self) -> float:
        return self.p_at(90)

    @computed_field
    @property
    def p_95(self) -> float:
        return self.p_at(95)

    @computed_field
    @property
    def r_50(self) -> float:
        return self.r_at(50)

    @computed_field
    @property
    def r_75(self) -> float:
        return self.r_at(75)

    @computed_field
    @property
    def r_85(self) -> float:
        return self.r_at(85)

    @computed_field
    @property
    def r_90(self) -> float:
        return self.r_at(90)

    @computed_field
    @property
    def r_95(self) -> float:
        return self.r_at(95)

    def mAP_at(self, thresh: int):
        maps = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
        return self.mAPs[maps.index(thresh)]

    def p_at(self, thresh: int):
        maps = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
        return self.precisions[maps.index(thresh)]

    def r_at(self, thresh: int):
        maps = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
        return self.recalls[maps.index(thresh)]


class ODEvalClass(BaseModel):
    map_report: DetEvalReport
    score_report: ODEvalScoreReport

    label: str = "obj"
    notes: str = ""


class ODEvalReport(BaseModel):
    metrics: list[ODEvalClass]
