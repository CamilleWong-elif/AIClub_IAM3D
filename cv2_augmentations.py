import cv2
import numpy as np
import random


# 1. Gaussian Noise (Sensor Noise)
class AddGaussianNoise:
    def __init__(self, mean=0, std=12, p=0.5):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, image):
        if random.random() > self.p:
            return image

        noise = np.random.normal(self.mean, self.std, image.shape).astype(np.float32)
        noisy = image.astype(np.float32) + noise
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)

        return noisy


# 2. Motion Blur (Simulates handheld movement)
class MotionBlur:
    def __init__(self, kernel_size=5, p=0.4):
        self.kernel_size = kernel_size
        self.p = p

    def __call__(self, image):
        if random.random() > self.p:
            return image

        kernel = np.zeros((self.kernel_size, self.kernel_size))
        kernel[self.kernel_size // 2, :] = np.ones(self.kernel_size)
        kernel = kernel / self.kernel_size

        return cv2.filter2D(image, -1, kernel)


# 3. Gaussian Blur (Focus variation)
class GaussianBlur:
    def __init__(self, kernel_size=5, p=0.4):
        self.kernel_size = kernel_size
        self.p = p

    def __call__(self, image):
        if random.random() > self.p:
            return image

        return cv2.GaussianBlur(image, (self.kernel_size, self.kernel_size), 0)


# 4. Random Shadow (Outdoor realism for soil scenes)
class RandomShadow:
    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, image):
        if random.random() > self.p:
            return image

        h, w = image.shape[:2]

        x1, y1 = random.randint(0, w), 0
        x2, y2 = random.randint(0, w), h

        mask = np.zeros_like(image[:, :, 0])

        polygon = np.array(
            [[(x1, y1), (x2, y2), (w, h), (0, h)]],
            dtype=np.int32
        )

        cv2.fillPoly(mask, polygon, 255)

        shadow_intensity = random.uniform(0.4, 0.75)

        shadow = image.copy()
        shadow[mask == 255] = (
            shadow[mask == 255] * shadow_intensity
        ).astype(np.uint8)

        return shadow


# List applied sequentially in train.py
cv2_augmentations = [
    AddGaussianNoise(std=15, p=0.5),
    MotionBlur(kernel_size=5, p=0.4),
    GaussianBlur(kernel_size=5, p=0.4),
    RandomShadow(p=0.3),
]