"""
Detection loss for the custom CNN detector.

Computes three components against anchor-encoded target tensors:
    1. Objectness loss   (BCE) - is there an object in this anchor/cell?
    2. Box regression    (MSE) - tx, ty, tw, th offsets for positive cells
    3. Classification    (BCE) - class probabilities for positive cells

The target tensor comes from TrashDetectionDataset._encode_targets().
The prediction tensor comes from DetectionHead (reshaped).
"""

import torch
import torch.nn as nn


class DetectionLoss(nn.Module):
    """
    Args:
        num_classes:    Number of object classes (1 for single-class)
        num_anchors:    Number of anchors
        lambda_coord:   Weight for box regression loss
        lambda_noobj:   Weight for negative objectness (reduces false positive penalty)
        lambda_obj:     Weight for positive objectness
        lambda_cls:     Weight for classification loss
    """

    def __init__(
        self,
        num_classes=1,
        num_anchors=3,
        lambda_coord=5.0,
        lambda_noobj=0.5,
        lambda_obj=1.0,
        lambda_cls=1.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.lambda_obj = lambda_obj
        self.lambda_cls = lambda_cls

        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, predictions, targets):
        """
        Args:
            predictions: Raw output from DetectionHead.
                         Shape: (B, num_anchors * (5 + num_classes), H, W)
            targets:     Encoded GT from dataset.
                         Shape: (B, num_anchors, 5 + num_classes, H, W)

        Returns:
            total_loss:  Scalar tensor
            loss_dict:   Dict with individual loss components for logging
        """
        B, _, H, W = predictions.shape

        # Reshape predictions to match target layout
        # (B, num_anchors, 5 + num_classes, H, W)
        pred = predictions.view(B, self.num_anchors, 5 + self.num_classes, H, W)

        # Split channels
        pred_tx = pred[:, :, 0, :, :]   # (B, A, H, W)
        pred_ty = pred[:, :, 1, :, :]
        pred_tw = pred[:, :, 2, :, :]
        pred_th = pred[:, :, 3, :, :]
        pred_obj = pred[:, :, 4, :, :]  # raw logits
        pred_cls = pred[:, :, 5:, :, :] # (B, A, C, H, W)

        tgt_tx = targets[:, :, 0, :, :]
        tgt_ty = targets[:, :, 1, :, :]
        tgt_tw = targets[:, :, 2, :, :]
        tgt_th = targets[:, :, 3, :, :]
        tgt_obj = targets[:, :, 4, :, :]
        tgt_cls = targets[:, :, 5:, :, :]

        # Masks
        obj_mask = tgt_obj == 1.0       # positive cells
        noobj_mask = tgt_obj == 0.0     # negative cells

        # --- 1. Objectness loss (BCE on all cells) ---
        obj_loss_all = self.bce(pred_obj, tgt_obj)
        obj_loss = (
            self.lambda_obj * (obj_loss_all * obj_mask).sum()
            + self.lambda_noobj * (obj_loss_all * noobj_mask).sum()
        )

        # --- 2. Box regression loss (MSE on positive cells only) ---
        if obj_mask.sum() > 0:
            coord_loss = (
                self.mse(torch.sigmoid(pred_tx), tgt_tx)[obj_mask].sum()
                + self.mse(torch.sigmoid(pred_ty), tgt_ty)[obj_mask].sum()
                + self.mse(pred_tw, tgt_tw)[obj_mask].sum()
                + self.mse(pred_th, tgt_th)[obj_mask].sum()
            )
            coord_loss = self.lambda_coord * coord_loss
        else:
            coord_loss = torch.tensor(0.0, device=predictions.device)

        # --- 3. Classification loss (BCE on positive cells only) ---
        if obj_mask.sum() > 0 and self.num_classes > 0:
            # Expand obj_mask to class dimension: (B, A, H, W) -> (B, A, 1, H, W)
            cls_mask = obj_mask.unsqueeze(2).expand_as(pred_cls)
            cls_loss = self.bce(pred_cls, tgt_cls)
            cls_loss = self.lambda_cls * (cls_loss * cls_mask).sum()
        else:
            cls_loss = torch.tensor(0.0, device=predictions.device)

        # --- Total ---
        num_pos = max(obj_mask.sum().item(), 1.0)
        total_loss = (obj_loss + coord_loss + cls_loss) / num_pos

        loss_dict = {
            "loss_obj": obj_loss.item() / num_pos,
            "loss_coord": coord_loss.item() / num_pos,
            "loss_cls": cls_loss.item() / num_pos,
            "total_loss": total_loss.item(),
            "num_positives": int(obj_mask.sum().item()),
        }

        return total_loss, loss_dict