import torch
from config.anchors import ANCHORS

def build_targets(targets, preds_shape, num_classes):
    """
    targets: List[Tensor] length B, each tensor Nx5
             (class_id, cx, cy, w, h)
    preds_shape: (B, A, 5+C, H, W)
    """

    B, A, _, H, W = preds_shape
    device = targets[0].device

    y_true = torch.zeros(
        (B, A, 5 + num_classes, H, W),
        device=device
    )

    for b in range(B):
        for t in targets[b]:
            class_id, cx, cy, w, h = t

            gx = int(cx * W)
            gy = int(cy * H)

            best_iou = 0
            best_anchor = 0

            for a, (aw, ah) in enumerate(ANCHORS):
                inter_w = min(w, aw)
                inter_h = min(h, ah)
                inter = inter_w * inter_h
                union = w*h + aw*ah - inter
                iou = inter / union if union > 0 else 0

                if iou > best_iou:
                    best_iou = iou
                    best_anchor = a

            tx = cx * W - gx
            ty = cy * H - gy
            tw = torch.log(w / ANCHORS[best_anchor][0] + 1e-6)
            th = torch.log(h / ANCHORS[best_anchor][1] + 1e-6)

            y_true[b, best_anchor, 0, gy, gx] = tx
            y_true[b, best_anchor, 1, gy, gx] = ty
            y_true[b, best_anchor, 2, gy, gx] = tw
            y_true[b, best_anchor, 3, gy, gx] = th
            y_true[b, best_anchor, 4, gy, gx] = 1.0
            y_true[b, best_anchor, 5 + int(class_id), gy, gx] = 1.0

    return y_true
