import os
from PIL import Image

def load_image_folder(folder_path):
    images = []
    for file in sorted(os.listdir(folder_path)):
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(folder_path, file)
            images.append(Image.open(img_path).convert("RGB"))
    return images
