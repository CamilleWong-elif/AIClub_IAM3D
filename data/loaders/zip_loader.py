import zipfile
from PIL import Image
import io

def load_images_from_zip(zip_path):
    images = []

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for name in zip_ref.namelist():
            if name.lower().endswith((".jpg", ".png", ".jpeg")):
                with zip_ref.open(name) as file:
                    image = Image.open(io.BytesIO(file.read())).convert("RGB")
                    images.append(image)

    return images
