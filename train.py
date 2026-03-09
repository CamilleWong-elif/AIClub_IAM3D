from ultralytics import YOLO

import os
import cv2
import numpy as np
from PIL import Image

from cv2_augmentations import cv2_augmentations
from torchvision_augmentations import torchvision_transforms

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main():
    DATA_YAML = "new_dataset/data.yaml"
    model = YOLO("yolov8m.pt")

    # Patch YOLO dataset loader (in-memory only)
    from ultralytics.data.dataset import YOLODataset
    _original_load_image = YOLODataset.load_image

    def _patched_load_image(self, i):
        img, hw_original, hw_resized = _original_load_image(self, i)

        # Apply CV2 augmentations
        for aug in cv2_augmentations:
            img = aug(img)

        # Convert BGR → RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # numpy → PIL
        pil_img = Image.fromarray(img_rgb)

        # torchvision transforms
        pil_img = torchvision_transforms(pil_img)

        # PIL → numpy
        img_rgb = np.array(pil_img)

        # RGB → BGR
        img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        return img, hw_original, hw_resized

    YOLODataset.load_image = _patched_load_image

    model.train(
        data=DATA_YAML,

        epochs=120,
        imgsz=512,x
        batch=8,
        device=0,   # FIXED

        workers=0,

        optimizer="SGD",
        lr0=4.0e-05,
        lrf=0.01044,
        momentum=0.98,
        weight_decay=0.00057,
        warmup_epochs=3.7522,
        warmup_momentum=0.95,

        box=8.91994,
        cls=0.58976,
        dfl=2.69179,

        hsv_h=0.02384,
        hsv_s=0.89915,
        hsv_v=0.54377,
        degrees=0.00834,
        translate=0.09761,
        scale=0.7178,
        shear=0.004,
        perspective=0.00029,

        flipud=0.00562,
        fliplr=0.44735,
        bgr=0.00755,
        mosaic=0.80562,
        mixup=0.01033,
        cutmix=0.00035,
        copy_paste=0.00139,
        close_mosaic=10,

        patience=15,
        save=True,
        plots=True,
        cache=True,
        amp=True,
    )


if __name__ == "__main__":
    main()