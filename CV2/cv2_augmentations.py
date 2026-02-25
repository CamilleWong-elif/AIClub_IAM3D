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
        if random.random() > self.p:  # Apply only with probability p
            return image

        # Generate Gaussian noise matching image shape (H, W, C)
        noise = np.random.normal(self.mean, self.std, image.shape).astype(np.float32)

        # Convert image to float32 to prevent overflow during addition
        noisy = image.astype(np.float32) + noise

        # Clip values to valid pixel range and convert back to uint8
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

        # Create empty convolution kernel
        kernel = np.zeros((self.kernel_size, self.kernel_size))

        # Set middle row to 1s to create horizontal motion blur effect
        kernel[self.kernel_size // 2, :] = np.ones(self.kernel_size)

        # Normalize so brightness stays consistent
        kernel = kernel / self.kernel_size

        # Apply convolution using OpenCV
        return cv2.filter2D(image, -1, kernel)


# --------------------------------------------------
# 3. Brightness / Contrast Shift (Lighting variation)
# --------------------------------------------------
class RandomBrightnessContrast:
    def __init__(self, brightness=0.25, contrast=0.25, p=0.6):
        self.brightness = brightness
        self.contrast = contrast
        self.p = p

    def __call__(self, image):
        if random.random() > self.p:
            return image

        # Contrast multiplier (scales pixel differences)
        alpha = 1.0 + random.uniform(-self.contrast, self.contrast)

        # Brightness shift (adds or subtracts intensity)
        beta = 255 * random.uniform(-self.brightness, self.brightness)

        # Applies: new_pixel = alpha * pixel + beta
        adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

        return adjusted


# --------------------------------------------------
# 4. Gaussian Blur (Focus variation)
# --------------------------------------------------
class GaussianBlur:
    def __init__(self, kernel_size=5, p=0.4):
        self.kernel_size = kernel_size
        self.p = p

    def __call__(self, image):
        if random.random() > self.p:
            return image

        # Applies Gaussian convolution; last parameter (0) auto-calculates sigma
        return cv2.GaussianBlur(image, (self.kernel_size, self.kernel_size), 0)


# --------------------------------------------------
# 5. Random Shadow (Outdoor realism for soil scenes)
# --------------------------------------------------
class RandomShadow:
    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, image):
        if random.random() > self.p:
            return image

        h, w = image.shape[:2]  # Extract image height and width

        # Random line across image to define shadow boundary
        x1, y1 = random.randint(0, w), 0
        x2, y2 = random.randint(0, w), h

        # Create single-channel mask (same height/width as image)
        mask = np.zeros_like(image[:, :, 0])

        # Define polygon region that will become shadow
        polygon = np.array([[(x1, y1), (x2, y2), (w, h), (0, h)]], dtype=np.int32)

        # Fill polygon region with 255 (white in mask)
        cv2.fillPoly(mask, polygon, 255)

        shadow_intensity = random.uniform(0.4, 0.75)

        shadow = image.copy()

        # Darken only pixels inside mask (mask == 255)
        shadow[mask == 255] = (
            shadow[mask == 255] * shadow_intensity
        ).astype(np.uint8)

        return shadow


# --------------------------------------------------
# 6. Color Shift (Camera white balance variation)
# --------------------------------------------------
class RandomColorShift:
    def __init__(self, shift_limit=20, p=0.5):
        self.shift_limit = shift_limit
        self.p = p

    def __call__(self, image):
        if random.random() > self.p:
            return image

        # Generate random shift for each color channel (B, G, R)
        shifts = np.random.randint(-self.shift_limit, self.shift_limit, 3)

        # Convert to int16 to prevent overflow when adding channel offsets
        img = image.astype(np.int16)

        img[:, :, 0] += shifts[0]  # Blue channel shift
        img[:, :, 1] += shifts[1]  # Green channel shift
        img[:, :, 2] += shifts[2]  # Red channel shift

        # Clip back into valid 0–255 range and convert to uint8
        img = np.clip(img, 0, 255).astype(np.uint8)

        return img
# 7. Random Flip (Horizontal / Vertical)

class RandomFlip:
    def __init__(self, horizontal=True, vertical=False, p=0.5):
        self.horizontal = horizontal
        self.vertical = vertical
        self.p = p

    def __call__(self, image):
        if random.random() > self.p:
            return image

        flip_code = None

        if self.horizontal and self.vertical:
            flip_code = -1   # both
        elif self.horizontal:
            flip_code = 1    # horizontal
        elif self.vertical:
            flip_code = 0    # vertical

        if flip_code is not None:
            return cv2.flip(image, flip_code)

        return image


# --------------------------------------------------
# MASTER LIST
# --------------------------------------------------

# These objects are instantiated once and applied sequentially in train.py
cv2_augmentations = [
    AddGaussianNoise(std=15, p=0.5),
    MotionBlur(kernel_size=5, p=0.4),
    RandomBrightnessContrast(p=0.6),
    GaussianBlur(kernel_size=5, p=0.4),
    RandomShadow(p=0.3),
    RandomColorShift(p=0.5),
]