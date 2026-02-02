from core.bbox import BoundingBox
from detectors.base import BaseDetector

class MockDetector(BaseDetector):
    def detect(self, image):
        return [
            BoundingBox(10, 10, 100, 120, 0.91, 0),
            BoundingBox(15, 15, 105, 125, 0.86, 0),
            BoundingBox(200, 200, 260, 260, 0.88, 1),
        ]
