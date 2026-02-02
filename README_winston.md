# Mars Rover Vision – Anchor-Based Bounding Box Detection

This repository contains a **modular, anchor-based object detection pipeline** designed as a **Mars rover computer vision prototype**, with beach trash (bottles, cans, bags, debris) used as a **realistic simulation proxy** for rover obstacles.

The system is built step-by-step with:

* CNN-based candidate generation (YOLO-style)
* Anchor–ground truth matching
* Masked detection loss
* Non-Maximum Suppression (NMS)
* Flexible data ingestion (images, folders, ZIPs)
* Clean separation of math, models, training, and UI (Tkinter comes later)

---

## Project Overview

High-level pipeline:

```
Image
 → CNN Backbone
 → Detection Head (anchors)
 → Raw bounding boxes
 → Anchor–GT matching (training)
 → Detection loss
 → Non-Maximum Suppression (inference)
 → Final bounding boxes
```

The codebase is designed so contributors can:

* Swap backbones
* Improve anchors
* Improve loss functions
* Add failure-case simulations
  **without touching core geometry or NMS math**

---

## Environment Setup

### Required Libraries

Install the following Python packages:

```bash
pip install torch torchvision pillow numpy
```

---

## Project Structure (Important)

```
mars_rover_vision/
├── config/          # constants, anchors, class definitions
├── core/            # IoU, NMS, bounding box math (DO NOT TOUCH lightly)
├── data/            # dataset loading & transforms
├── detectors/       # CNN detector implementations
├── models/          # CNN backbones & detection heads
├── training/        # loss, target building, trainer
├── tests/           # unit tests (recommended to expand)
└── app/             # future Tkinter UI (not active yet)
```

---

## FORMAT

### Dataset Format

Training data **must** resolve to the following structure (whether originally from images, folders, or ZIPs):

```
dataset_root/
├── images/
│   ├── img001.jpg
│   ├── img002.png
│   └── ...
├── labels/
│   ├── img001.txt
│   ├── img002.txt
│   └── ...
```

* Image filenames and label filenames **must match**
* One label file per image

---

### Label Format

Labels use **YOLO-style normalized bounding boxes**.

Each line in a label file:

```
<class_id> <cx> <cy> <w> <h>
```

Where:

* `cx, cy` = box center (normalized, 0–1)
* `w, h` = box width & height (normalized, 0–1)
* `class_id` must match `config/classes.py`

#### Example (`img001.txt`)

```
0 0.52 0.61 0.18 0.42
1 0.33 0.47 0.12 0.20
```

---

## Using Images, Folders, and ZIP Files

### Single Image (Inference / Testing Only)

Single images are supported for **inference**, not training.

```python
from PIL import Image
import numpy as np

from detectors.cnn_detector import CNNDetector
from core.engine import BoundingBoxEngine

detector = CNNDetector()
engine = BoundingBoxEngine()

image = np.array(Image.open("test.jpg"))
raw_boxes = detector.detect(image)
final_boxes = engine.process(raw_boxes)

for box in final_boxes:
    print(box)
```

---

### Folder of Images (Training – Recommended)

Prepare your dataset as described above and point the trainer to the dataset root:

```python
dataset_path = "data/my_dataset"
```

---

### ZIP Files

ZIPs are supported **as an upload / transport format**, but must be extracted before training:

```bash
unzip dataset.zip -d data/my_dataset
```

Training requires random-access file I/O and does **not** read directly from ZIP archives.

---

## Training the Model

Minimal training example:

```python
from detectors.cnn_detector import CNNDetector
from training.trainer import Trainer

detector = CNNDetector(device="cpu")

trainer = Trainer(
    detector=detector,
    dataset_path="data/my_dataset",
    device="cpu",
    batch_size=4
)

trainer.train(epochs=20)
trainer.save("checkpoints/rover_detector.pt")
```

What happens internally:

* Images are loaded and resized
* Ground truth boxes are matched to anchors
* Masked detection loss is computed
* CNN weights are updated
* Loss is printed per epoch

---

## Running Inference (After Training)

```python
from PIL import Image
import numpy as np

from detectors.cnn_detector import CNNDetector
from core.engine import BoundingBoxEngine

detector = CNNDetector()
checkpoint = torch.load("checkpoints/rover_detector.pt")

detector.backbone.load_state_dict(checkpoint["backbone"])
detector.head.load_state_dict(checkpoint["head"])

engine = BoundingBoxEngine()

image = np.array(Image.open("test.jpg"))
raw_boxes = detector.detect(image)
final_boxes = engine.process(raw_boxes)

for box in final_boxes:
    print(box)
```

---

## Files That SHOULD NOT Be Touched (Unless Changing Math)

These are **math-critical** and affect correctness:

* `core/iou.py`
* `core/nms.py`
* `core/engine.py`
* `training/target_builder.py`

Changes here can break training or inference.

---

## Files That SHOULD Be Touched (Experimentation/Improvements)

Safe/expected experimentation areas:

* `models/backbone.py`
  (try ResNet, MobileNet, etc.)

* `models/detection_head.py`
  (multi-scale heads, different layouts)

* `config/anchors.py`
  (manual tuning or k-means anchors)

* `training/loss.py`
  (loss weights, IoU-based loss, etc.)

* `data/transforms.py`
  (augmentation, normalization)

* `detectors/cnn_detector.py`
  (decoding strategy, thresholds)

---

## Step-by-Step: From Repo Clone to Training

### 1. Clone the Repository

```bash
git clone <repo_url>
cd <project-directory>
```

### 2. Install Dependencies

```bash
pip install torch torchvision pillow numpy
```

### 3. Prepare Dataset

* Create `images/` and `labels/`
* Add images
* Add YOLO-format label files

### 4. Verify Class IDs

Check:

```python
config/classes.py
```

### 5. Train the Model

```bash
python train.py   # or run training script manually
```

### 6. Save Weights

```python
trainer.save("checkpoints/rover_detector.pt")
```

### 7. Run Inference

Use `CNNDetector + BoundingBoxEngine` on new images.

---
