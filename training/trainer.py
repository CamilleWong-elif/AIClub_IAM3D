import torch
from torch.utils.data import DataLoader

from data.dataset import RoverDataset
from data.transforms import default_transforms
from training.loss import DetectionLoss
from training.target_builder import build_targets


class Trainer:
    def __init__(
        self,
        detector,
        dataset_path,
        device="cpu",
        batch_size=4,
        learning_rate=1e-4,
        num_workers=0
    ):
        """
        detector: CNNDetector
        dataset_path: path to dataset root
        device: 'cpu' or 'cuda'
        """

        self.device = device
        self.detector = detector

        self.dataset = RoverDataset(
            root_dir=dataset_path,
            transform=default_transforms()
        )

        self.loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(device != "cpu")
        )

        self.loss_fn = DetectionLoss()

        self.optimizer = torch.optim.Adam(
            list(self.detector.backbone.parameters()) +
            list(self.detector.head.parameters()),
            lr=learning_rate
        )

    def train(self, epochs=10):
        self.detector.backbone.train()
        self.detector.head.train()

        for epoch in range(epochs):
            epoch_loss = 0.0

            for images, targets in self.loader:
                images = images.to(self.device)
                targets = targets.to(self.device)

                self.optimizer.zero_grad()

                features = self.detector.backbone(images)
                preds = self.detector.head(features)

                y_true = build_targets(
                    targets=targets,
                    preds_shape=preds.shape,
                    num_classes=self.detector.num_classes
                )

                loss = self.loss_fn(preds, y_true)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / max(len(self.loader), 1)

            print(
                f"[Epoch {epoch + 1}/{epochs}] "
                f"Loss: {avg_loss:.6f}"
            )

    def save(self, path):
        torch.save({
            "backbone": self.detector.backbone.state_dict(),
            "head": self.detector.head.state_dict()
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.detector.backbone.load_state_dict(checkpoint["backbone"])
        self.detector.head.load_state_dict(checkpoint["head"])
