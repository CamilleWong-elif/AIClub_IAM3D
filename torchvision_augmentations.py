# torchvision_augmentations.py

import torchvision.transforms as T

torchvision_transforms = T.Compose([
    T.RandomAdjustSharpness(
        sharpness_factor=1.5,
        p=0.2
    ),
    T.RandomGrayscale(p=0.05),
])