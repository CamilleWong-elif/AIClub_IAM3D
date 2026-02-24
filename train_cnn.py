"""
Training script for the custom CNN trash detector.

Trains SimpleBackbone + DetectionHead using DetectionLoss
on YOLO-format labeled data.

Usage:
    python train_cnn.py

Configure DATASET_ROOT below to point to your dataset.
"""

import os
import sys
import time
import torch
from torch.utils.data import DataLoader

# -------------------------
# CONFIG
# -------------------------
DATASET_ROOT = "/path/to/your/dataset"  # change this: expects train/, valid/ with images/ + labels/

IMG_SIZE = 416
BATCH_SIZE = 8
NUM_EPOCHS = 120
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 5e-4
PATIENCE = 15           # early stopping
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 0         # set to 2-4 if stable on your machine
NUM_CLASSES = 1         # single-class (trash)
SAVE_DIR = "runs/cnn"   # where to save checkpoints

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# -------------------------
# Imports (project modules)
# -------------------------
from models.backbone import SimpleBackbone
from models.detection_head import DetectionHead
from config.anchors import ANCHORS
from core.loss import DetectionLoss
from data.dataset import TrashDetectionDataset


def train():
    os.makedirs(SAVE_DIR, exist_ok=True)
    num_anchors = len(ANCHORS)

    # --- Model ---
    backbone = SimpleBackbone().to(DEVICE)
    head = DetectionHead(NUM_CLASSES, num_anchors).to(DEVICE)

    # --- Loss ---
    criterion = DetectionLoss(
        num_classes=NUM_CLASSES,
        num_anchors=num_anchors,
    )

    # --- Optimizer ---
    params = list(backbone.parameters()) + list(head.parameters())
    optimizer = torch.optim.Adam(params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # --- Data ---
    train_ds = TrashDetectionDataset(
        root=os.path.join(DATASET_ROOT, "train"),
        img_size=IMG_SIZE,
        num_classes=NUM_CLASSES,
        augment=True,
    )
    val_ds = TrashDetectionDataset(
        root=os.path.join(DATASET_ROOT, "valid"),
        img_size=IMG_SIZE,
        num_classes=NUM_CLASSES,
        augment=False,
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"),
    )

    print(f"Device: {DEVICE}")
    print(f"Train: {len(train_ds)} images | Val: {len(val_ds)} images")
    print(f"Anchors: {ANCHORS}")
    print(f"Grid size: {IMG_SIZE // 8}x{IMG_SIZE // 8}")
    print("-" * 60)

    # --- Training loop ---
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        # ---- Train ----
        backbone.train()
        head.train()
        train_loss_sum = 0.0
        train_batches = 0

        for images, targets in train_loader:
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            features = backbone(images)
            predictions = head(features)

            loss, loss_dict = criterion(predictions, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss_dict["total_loss"]
            train_batches += 1

        scheduler.step()
        avg_train_loss = train_loss_sum / max(train_batches, 1)

        # ---- Validate ----
        backbone.eval()
        head.eval()
        val_loss_sum = 0.0
        val_batches = 0

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(DEVICE)
                targets = targets.to(DEVICE)

                features = backbone(images)
                predictions = head(features)

                _, loss_dict = criterion(predictions, targets)
                val_loss_sum += loss_dict["total_loss"]
                val_batches += 1

        avg_val_loss = val_loss_sum / max(val_batches, 1)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:3d}/{NUM_EPOCHS} | "
            f"train_loss: {avg_train_loss:.4f} | "
            f"val_loss: {avg_val_loss:.4f} | "
            f"lr: {scheduler.get_last_lr()[0]:.6f} | "
            f"{elapsed:.1f}s"
        )

        # ---- Save best + early stopping ----
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "backbone_state": backbone.state_dict(),
                "head_state": head.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": avg_val_loss,
            }, os.path.join(SAVE_DIR, "best.pt"))
            print(f"  -> Saved best.pt (val_loss={avg_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"  Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
                break

        # Save last checkpoint every epoch
        torch.save({
            "epoch": epoch,
            "backbone_state": backbone.state_dict(),
            "head_state": head.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_loss": avg_val_loss,
        }, os.path.join(SAVE_DIR, "last.pt"))

    print("-" * 60)
    print(f"Training complete. Best val_loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {os.path.abspath(SAVE_DIR)}")


if __name__ == "__main__":
    train()