"""
Datasets for the custom CNN detector.

RoverDataset:
    General-purpose loader. Returns image tensor + raw YOLO boxes.
    Use for evaluation, visualization, or custom pipelines.

TrashDetectionDataset:
    Training-specific loader. Encodes GT into anchor-based target tensors
    matching DetectionHead output shape. Use with train_cnn.py + DetectionLoss.

Both expect standard YOLO folder layout:
    split_root/
        images/
        labels/

Labels: class_id cx cy w h (all normalized 0-1).
"""

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from config.anchors import ANCHORS


# -------------------------------------------------------
# Original general-purpose dataset
# -------------------------------------------------------
class RoverDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_dir = os.path.join(root_dir, "images")
        self.label_dir = os.path.join(root_dir, "labels")
        self.transform = transform

        self.images = sorted(os.listdir(self.image_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)

        image = Image.open(img_path).convert("RGB")

        label_path = os.path.join(
            self.label_dir,
            os.path.splitext(img_name)[0] + ".txt"
        )

        boxes = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    class_id, cx, cy, w, h = map(float, line.split())
                    boxes.append([class_id, cx, cy, w, h])

        boxes = torch.tensor(boxes, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, boxes


# -------------------------------------------------------
# Training dataset with anchor-based target encoding
# -------------------------------------------------------
class TrashDetectionDataset(Dataset):
    """
    Args:
        root:        Path to split folder (e.g. dataset/train/)
        img_size:    Resize all images to (img_size, img_size)
        num_classes: Number of object classes (3: object, sand, large_collection)
        augment:     If True, apply horizontal flip augmentation
    """

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, root, img_size=416, num_classes=3, augment=False):
        self.root = root
        self.img_size = img_size
        self.num_classes = num_classes
        self.augment = augment
        self.anchors = ANCHORS

        images_dir = os.path.join(root, "images")
        self.image_files = sorted(
            f for f in os.listdir(images_dir)
            if os.path.splitext(f)[1].lower() in self.IMG_EXTS
        )
        self.images_dir = images_dir
        self.labels_dir = os.path.join(root, "labels")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # --- Load and resize image ---
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))

        # --- Augmentation ---
        flip_h = False
        if self.augment and np.random.rand() > 0.5:
            img = np.fliplr(img).copy()
            flip_h = True

        # Normalize to [0,1], shape (3, H, W)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        # --- Load YOLO labels ---
        stem = os.path.splitext(img_name)[0]
        label_path = os.path.join(self.labels_dir, stem + ".txt")
        gt_boxes = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:5])
                    if flip_h:
                        cx = 1.0 - cx
                    gt_boxes.append((cls_id, cx, cy, bw, bh))

        # --- Encode into target tensor ---
        target = self._encode_targets(gt_boxes)
        return img_tensor, target

    def _encode_targets(self, gt_boxes):
        """
        Encode GT boxes into target tensor matching DetectionHead output.

        Backbone downsamples 8x (three stride-2 convs):
            grid_size = img_size // 8

        Target shape: (num_anchors, 5 + num_classes, grid_h, grid_w)
        Per anchor per cell: [tx, ty, tw, th, objectness, class_0, class_1, class_2]
            - objectness = 1 where GT box is assigned, 0 elsewhere
            - tx, ty = fractional offset within grid cell
            - tw, th = log(gt_size / anchor_size)
            - class channels = one-hot
        """
        num_anchors = len(self.anchors)
        grid_size = self.img_size // 8

        target = torch.zeros(num_anchors, 5 + self.num_classes, grid_size, grid_size)

        for cls_id, cx, cy, bw, bh in gt_boxes:
            gx = min(int(cx * grid_size), grid_size - 1)
            gy = min(int(cy * grid_size), grid_size - 1)

            # Best-matching anchor (IoU with both centered at origin)
            best_anchor = 0
            best_iou = 0.0
            for a_idx, (aw, ah) in enumerate(self.anchors):
                inter_w = min(aw, bw)
                inter_h = min(ah, bh)
                inter = inter_w * inter_h
                union = aw * ah + bw * bh - inter
                iou = inter / union if union > 0 else 0.0
                if iou > best_iou:
                    best_iou = iou
                    best_anchor = a_idx

            aw, ah = self.anchors[best_anchor]

            tx = cx * grid_size - gx
            ty = cy * grid_size - gy
            tw = np.log(bw / aw + 1e-8)
            th = np.log(bh / ah + 1e-8)

            target[best_anchor, 0, gy, gx] = tx
            target[best_anchor, 1, gy, gx] = ty
            target[best_anchor, 2, gy, gx] = tw
            target[best_anchor, 3, gy, gx] = th
            target[best_anchor, 4, gy, gx] = 1.0

            if cls_id < self.num_classes:
                target[best_anchor, 5 + cls_id, gy, gx] = 1.0

        return target