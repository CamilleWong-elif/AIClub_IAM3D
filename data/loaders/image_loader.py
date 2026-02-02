from PIL import Image
import os

def load_single_image(image_path):
    if not os.path.isfile(image_path):
        raise FileNotFoundError(image_path)
    return Image.open(image_path).convert("RGB")
