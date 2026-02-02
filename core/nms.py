from core.iou import compute_iou

def non_max_suppression(boxes, iou_threshold, score_threshold):
    boxes = [b for b in boxes if b.score >= score_threshold]
    final_boxes = []

    boxes_by_class = {}
    for box in boxes:
        boxes_by_class.setdefault(box.class_id, []).append(box)

    for _, class_boxes in boxes_by_class.items():
        class_boxes.sort(key=lambda b: b.score, reverse=True)

        while class_boxes:
            best = class_boxes.pop(0)
            final_boxes.append(best)

            class_boxes = [
                box for box in class_boxes
                if compute_iou(best, box) < iou_threshold
            ]

    return final_boxes
