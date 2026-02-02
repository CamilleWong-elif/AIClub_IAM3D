from detectors.mock_detector import MockDetector
from core.engine import BoundingBoxEngine

detector = MockDetector()
engine = BoundingBoxEngine()

raw_boxes = detector.detect(None)
final_boxes = engine.process(raw_boxes)

for box in final_boxes:
    print(box)
