import torch
import torch.nn as nn

class DetectionLoss(nn.Module):
    def __init__(
        self,
        lambda_box=5.0,
        lambda_obj=1.0,
        lambda_noobj=0.5,
        lambda_cls=1.0
    ):
        super().__init__()

        self.lambda_box = lambda_box
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.lambda_cls = lambda_cls

        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, preds, targets):
        """
        preds, targets: [B, A, 5+C, H, W]
        """

        obj_mask = targets[:, :, 4] == 1
        noobj_mask = targets[:, :, 4] == 0

        # Box loss (only where object exists)
        box_loss = self.mse(
            preds[:, :, :4][obj_mask],
            targets[:, :, :4][obj_mask]
        ) if obj_mask.any() else torch.tensor(0.0, device=preds.device)

        # Objectness loss
        obj_loss = self.bce(
            preds[:, :, 4][obj_mask],
            targets[:, :, 4][obj_mask]
        ) if obj_mask.any() else torch.tensor(0.0, device=preds.device)

        noobj_loss = self.bce(
            preds[:, :, 4][noobj_mask],
            targets[:, :, 4][noobj_mask]
        )

        # Class loss
        cls_loss = self.ce(
            preds[:, :, 5:][obj_mask],
            torch.argmax(targets[:, :, 5:][obj_mask], dim=1)
        ) if obj_mask.any() else torch.tensor(0.0, device=preds.device)

        return (
            self.lambda_box * box_loss +
            self.lambda_obj * obj_loss +
            self.lambda_noobj * noobj_loss +
            self.lambda_cls * cls_loss
        )
