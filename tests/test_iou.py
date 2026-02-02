def test_iou_identity():
    from core.bbox import BoundingBox
    from core.iou import compute_iou

    box = BoundingBox(0, 0, 10, 10, 1.0, 0)
    assert compute_iou(box, box) == 1.0
