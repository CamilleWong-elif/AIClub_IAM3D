from core.nms import non_max_suppression
from config.defaults import (
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_SCORE_THRESHOLD
)

class BoundingBoxEngine:
    def __init__(
        self,
        iou_thresh=DEFAULT_IOU_THRESHOLD,
        score_thresh=DEFAULT_SCORE_THRESHOLD
    ):
        self.iou_thresh = iou_thresh
        self.score_thresh = score_thresh

    def process(self, raw_boxes):
        return non_max_suppression(
            raw_boxes,
            self.iou_thresh,
            self.score_thresh
        )
