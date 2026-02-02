import torch
import numpy as np

from core.bbox import BoundingBox
from detectors.base import BaseDetector
from models.backbone import SimpleBackbone
from models.detection_head import DetectionHead
from config.classes import TRASH_CLASSES
from config.anchors import ANCHORS

class CNNDetector(BaseDetector):
    def __init__(self, device="cpu"):
        self.device = device
        self.num_classes = len(TRASH_CLASSES)
        self.num_anchors = len(ANCHORS)

        self.backbone = SimpleBackbone().to(device)
        self.head = DetectionHead(
            self.num_classes,
            self.num_anchors
        ).to(device)

        self.backbone.eval()
        self.head.eval()

    def detect(self, image):
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float()

        image = image.unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.backbone(image)
            preds = self.head(features)

        return self._decode(preds, image.shape[-2:])

    def _decode(self, preds, image_size):
        B, _, H, W = preds.shape
        preds = preds.view(
            B,
            self.num_anchors,
            5 + self.num_classes,
            H,
            W
        )

        boxes = []
        img_h, img_w = image_size

        for a, (aw, ah) in enumerate(ANCHORS):
            for y in range(H):
                for x in range(W):
                    tx, ty, tw, th, tobj = preds[0, a, :5, y, x]

                    obj = torch.sigmoid(tobj).item()
                    if obj < 0.3:
                        continue

                    cx = (x + torch.sigmoid(tx).item()) / W
                    cy = (y + torch.sigmoid(ty).item()) / H

                    bw = aw * torch.exp(tw).item()
                    bh = ah * torch.exp(th).item()

                    class_logits = preds[0, a, 5:, y, x]
                    class_probs = torch.softmax(class_logits, dim=0)
                    class_id = torch.argmax(class_probs).item()

                    confidence = obj * class_probs[class_id].item()

                    x1 = (cx - bw / 2) * img_w
                    y1 = (cy - bh / 2) * img_h
                    x2 = (cx + bw / 2) * img_w
                    y2 = (cy + bh / 2) * img_h

                    boxes.append(
                        BoundingBox(x1, y1, x2, y2, confidence, class_id)
                    )

        return boxes
