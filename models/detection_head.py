import torch
import torch.nn as nn

class DetectionHead(nn.Module):
    def __init__(self, num_classes, num_anchors):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.pred_dim = num_anchors * (5 + num_classes)

        self.conv = nn.Conv2d(
            in_channels=128,
            out_channels=self.pred_dim,
            kernel_size=1
        )

    def forward(self, x):
        return self.conv(x)
