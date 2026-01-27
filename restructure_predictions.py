'''
From raw dataset, search for images with sand using a sand detection model.

input folder structure expected:
yolov8 folder structure
ROOT/
    train/
        images/
        labels/
    valid/
        images/
        labels/
    test/
        images/
        labels/

Output structure:
OUT_ROOT/
    keep/
        images/{train,valid,test}/
        labels/{train,valid,test}/          # ORIGINAL labels
        predicted_labels/{train,valid,test}/ # PREDICTED labels
    maybe/
        images/{train,valid,test}/
        labels/{train,valid,test}/
        predicted_labels/{train,valid,test}/
    reject/
        images/{train,valid,test}/
        labels/{train,valid,test}/
        predicted_labels/{train,valid,test}/

categorizes images that might have sand
high confidence goes to keep. 
med conf goes to maybe. 
very low to zero conf goes to reject. 
'''




from ultralytics import YOLO
from pathlib import Path
import shutil

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "/Users/kikihan/Desktop/IAM3D/runs/detect/train2/weights/best.pt"

DATASET_ROOT = Path("/Users/kikihan/Desktop/IAM3D/raw_data/litter1_bbox")  # expects train/valid/test with images/labels
OUT_ROOT = Path("/Users/kikihan/Desktop/IAM3D/processing/litter1_bbox_processed")

SPLITS = ["train", "valid", "test"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# class ids in sand detection model
SAND_CLASS_ID = 1

# High-recall inference for mining
INFER_CONF = 0.01
INFER_IOU = 0.7

# Bucket thresholds
SAND_KEEP_CONF = 0.10  # sand at very low confidence -> KEEP (your rule)

# Copy vs hardlink
USE_COPY = True  # True = copy2, False = hardlink (same disk only)

# -----------------------------
# Helpers
# -----------------------------
def ensure_bucket_dirs(bucket: str):
    """Create output dirs for images + original labels + predicted labels."""
    for split in SPLITS:
        (OUT_ROOT / bucket / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / bucket / "labels" / split).mkdir(parents=True, exist_ok=True)  # ORIGINAL labels
        (OUT_ROOT / bucket / "predicted_labels" / split).mkdir(parents=True, exist_ok=True)  # PREDICTED labels

def transfer(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        return
    if dst.exists():
        dst.unlink()
    if USE_COPY:
        shutil.copy2(src, dst)
    else:
        src.link_to(dst)

def max_conf_for_class(classes, confs, class_id: int) -> float:
    m = 0.0
    for c, cf in zip(classes, confs):
        if int(c) == class_id:
            m = max(m, float(cf))
    return m

def decide_bucket(classes, confs) -> str:
    sand_max = max_conf_for_class(classes, confs, SAND_CLASS_ID)

    # HARD GATE: no sand -> reject
    if sand_max <= 0.0:
        return "reject"

    # sand detected at very low confidence -> keep
    if sand_max >= SAND_KEEP_CONF:
        return "keep"

    # sand exists but too weak -> maybe
    return "maybe"

def write_predicted_labels(out_path: Path, result):
    """
    Write YOLO-format predictions:
      cls xc yc w h   (normalized)
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if result.boxes is None or len(result.boxes) == 0:
        out_path.write_text("")
        return

    xywhn = result.boxes.xywhn.cpu().numpy()
    cls = result.boxes.cls.cpu().numpy()

    lines = []
    for (x, y, w, h), c in zip(xywhn, cls):
        lines.append(f"{int(c)} {float(x):.6f} {float(y):.6f} {float(w):.6f} {float(h):.6f}")

    out_path.write_text("\n".join(lines) + ("\n" if lines else ""))

# -----------------------------
# Main
# -----------------------------
def main():
    model = YOLO(MODEL_PATH)

    for b in ["keep", "maybe", "reject"]:
        ensure_bucket_dirs(b)

    counts = {"keep": 0, "maybe": 0, "reject": 0}

    for split in SPLITS:
        images_dir = DATASET_ROOT / split / "images"
        labels_dir = DATASET_ROOT / split / "labels"

        if not images_dir.exists():
            print(f"[WARN] Missing images: {images_dir}")
            continue

        imgs = [p for p in images_dir.iterdir()
                if p.is_file() and p.suffix.lower() in IMG_EXTS]

        print(f"{split}: {len(imgs)} images")

        for img_path in imgs:
            result = model(img_path, verbose=False, conf=INFER_CONF, iou=INFER_IOU)[0]

            if result.boxes is None or len(result.boxes) == 0:
                classes, confs = [], []
            else:
                classes = result.boxes.cls.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()

            bucket = decide_bucket(classes, confs)
            counts[bucket] += 1

            # --- Copy ORIGINAL image into bucket ---
            out_img = OUT_ROOT / bucket / "images" / split / img_path.name
            transfer(img_path, out_img)

            # --- Copy ORIGINAL (ground-truth) label into bucket/labels ---
            out_lbl = OUT_ROOT / bucket / "labels" / split / f"{img_path.stem}.txt"
            src_lbl = labels_dir / f"{img_path.stem}.txt"
            if src_lbl.exists():
                transfer(src_lbl, out_lbl)
            else:
                out_lbl.write_text("")  # empty label is valid for YOLO

            # --- Write PREDICTED label into bucket/predicted_labels ---
            out_pred = OUT_ROOT / bucket / "predicted_labels" / split / f"{img_path.stem}.txt"
            write_predicted_labels(out_pred, result)

    print("\nDone.")
    print(counts)
    print("Output:", OUT_ROOT.resolve())

if __name__ == "__main__":
    main()
