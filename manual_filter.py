"""
Bucket reviewer for YOLO datasets.

Folder structure expected:

ROOT/
  keep/
    images/{train,valid,test}/
    labels/{train,valid,test}/
    orig_labels/{train,valid,test}/
  maybe/
    images/{train,valid,test}/
    labels/{train,valid,test}/
    orig_labels/{train,valid,test}/
  reject/
    images/{train,valid,test}/
    labels/{train,valid,test}/
    orig_labels/{train,valid,test}/
  kept/                      <-- will be created if missing
    images/{train,valid,test}/
    labels/{train,valid,test}/
    orig_labels/{train,valid,test}/

You can review ONE bucket at a time by changing REVIEW_BUCKET.
# RED is the original dataset labels.
# BLUE is the labels predicted by sand detection model. Comment out 
Keys while viewing:
  k = MOVE sample to kept/
  m = MOVE sample to maybe/
  r = MOVE sample to reject/
  s = skip
  q = quit
"""

from pathlib import Path
import shutil
import cv2
from PIL import Image, ImageOps
import numpy as np


# -----------------------------
# CONFIG 
# -----------------------------
ROOT = Path("/Users/kikihan/Desktop/IAM3D/processing/litter1_bbox_processed")  # <-- change to  bucket root
REVIEW_BUCKET = "maybe"  # <-- "keep" or "maybe" or "reject"

SPLITS = ["train", "valid", "test"] 
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Destinations for key presses
BUCKET_KEPT = "kept"
BUCKET_MAYBE = "maybe"
BUCKET_REJECT = "reject"

# What to do on key press:
# - 'k' is MOVE to kept
# - 'm' is MOVE to maybe
# - 'r' is MOVE to reject
KEY_ACTIONS = {
    "k": ("move", BUCKET_KEPT),
    "m": ("move", BUCKET_MAYBE),
    "r": ("move", BUCKET_REJECT),
}

# -----------------------------
# Path helpers
# -----------------------------
def bucket_path(bucket: str, sub: str, split: str) -> Path:
    """Return ROOT/bucket/sub/split"""
    return ROOT / bucket / sub / split

def ensure_bucket_structure(bucket: str):
    """Create images/labels/predicted_labels + split subfolders if missing."""
    for split in SPLITS:
        for sub in ["images", "labels", "predicted_labels"]:
            bucket_path(bucket, sub, split).mkdir(parents=True, exist_ok=True)

# -----------------------------
# YOLO label helpers (for drawing)
# -----------------------------
def load_yolo_boxes(label_file: Path):
    boxes = []
    if not label_file.exists():
        return boxes

    for line in label_file.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            xc, yc, bw, bh = map(float, parts[1:5])

            # warn if looks non-normalized (pixels or wrong format)
            if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 <= bw <= 1 and 0 <= bh <= 1):
                print(f"[WARN GT not normalized?] {label_file.name}: {xc:.3f} {yc:.3f} {bw:.3f} {bh:.3f}")

            boxes.append((cls, xc, yc, bw, bh))
        except Exception:
            continue
    return boxes

  

def draw_boxes(img, boxes, color_bgr, thickness=2):
    """
    Draw YOLO normalized boxes on an OpenCV image.
    color_bgr: (B,G,R)
    """
    h, w = img.shape[:2]
    for cls, xc, yc, bw, bh in boxes:
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)

        # clamp to image bounds
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))

        cv2.rectangle(img, (x1, y1), (x2, y2), color_bgr, thickness)

# -----------------------------
# File transfer helpers
# -----------------------------





def safe_move(src: Path, dst: Path):
    """Move src -> dst, ALWAYS replacing dst if it exists."""
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)

    # if you're moving within the same bucket/same path, do nothing
    try:
        if src.resolve() == dst.resolve():
            return
    except Exception:
        pass

    # remove existing dst so move won't error
    if dst.exists():
        try:
            dst.unlink()
        except IsADirectoryError:
            shutil.rmtree(dst)

    shutil.move(str(src), str(dst))


def transfer_triplet(split: str, img_path: Path, src_bucket: str, mode: str, dst_bucket: str):
    """
    Transfer ONE sample consisting of:
      - image file
      - labels/<stem>.txt              (original/GT)
      - predicted_labels/<stem>.txt    (predictions)

    Collision behavior:
      - If destination already has same name, it gets replaced.
    """
    if src_bucket == dst_bucket:
        return
    
    stem = img_path.stem
    img_name = img_path.name

    src_img = img_path
    src_lbl = bucket_path(src_bucket, "labels", split) / f"{stem}.txt"
    src_pred = bucket_path(src_bucket, "predicted_labels", split) / f"{stem}.txt"

    dst_img = bucket_path(dst_bucket, "images", split) / img_name
    dst_lbl = bucket_path(dst_bucket, "labels", split) / f"{stem}.txt"
    dst_pred = bucket_path(dst_bucket, "predicted_labels", split) / f"{stem}.txt"

    if mode == "move":
        safe_move(src_img, dst_img)
        safe_move(src_lbl, dst_lbl)
        safe_move(src_pred, dst_pred)
    else:
        raise ValueError(f"Unknown mode: {mode}")


# -----------------------------
# UI: show image with boxes
# -----------------------------
def imread_exif_fixed(path: Path):
    """Load image with EXIF orientation applied, return OpenCV BGR."""
    img_pil = Image.open(path)
    img_pil = ImageOps.exif_transpose(img_pil)   # fixes rotated phone images
    img = np.array(img_pil.convert("RGB"))
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def label_stats(boxes):
    if not boxes:
        return "no_boxes"
    bad = 0
    for cls, xc, yc, bw, bh in boxes:
        if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
            bad += 1
    return f"n={len(boxes)} bad={bad}"


WIN = "review"

def show_image_with_boxes(img_path, split, idx, total):
    img = cv2.imread(str(img_path))
    if img is None:
        return None

    stem = img_path.stem
    orig_lbl = bucket_path(REVIEW_BUCKET, "labels", split) / f"{stem}.txt"
    orig_boxes = load_yolo_boxes(orig_lbl)
    draw_boxes(img, orig_boxes, (0,0,255), 2)

    # predicted labels are BLUE 
    # can comment out this block since the labels are not too accurate
    '''pred_lbl = bucket_path(REVIEW_BUCKET, "predicted_labels", split) / f"{stem}.txt"
    pred_boxes = load_yolo_boxes(pred_lbl)
    draw_boxes(img, pred_boxes, color_bgr=(255, 0, 0), thickness=1)  # BLUE'''
    

    # draw UI text directly on image (fast)
    ui = f"{idx}/{total} {REVIEW_BUCKET} | k kept  m maybe  r reject  s skip  q quit"
    cv2.putText(img, ui, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

    cv2.imshow(WIN, img)
    key = cv2.waitKey(0) & 0xFF
    return key


   

# -----------------------------
# Main
# -----------------------------
def main():
    # Make sure destination buckets exist
    for b in [BUCKET_KEPT, BUCKET_MAYBE, BUCKET_REJECT, "keep","maybe","reject"]:
        ensure_bucket_structure(b)

    moved = {"kept_move": 0, "maybe_move": 0, "reject_move": 0}
    skipped = 0

    for split in SPLITS:
        src_images_dir = bucket_path(REVIEW_BUCKET, "images", split)
        if not src_images_dir.exists():
            print(f"[WARN] Missing: {src_images_dir}")
            continue

        images = [p for p in sorted(src_images_dir.iterdir())
                  if p.is_file() and p.suffix.lower() in IMG_EXTS]

        total = len(images)
        print(f"\n{split}: {total} images in {REVIEW_BUCKET}/images/{split}")

        for idx, img_path in enumerate(images, start=1):
            key = show_image_with_boxes(img_path, split, idx, total)
            if key is None:
                skipped += 1
                continue

            ch = chr(key).lower()

            if ch in KEY_ACTIONS:
                mode, dst_bucket = KEY_ACTIONS[ch]

                # transfer image + labels + orig_labels
                transfer_triplet(split, img_path, REVIEW_BUCKET, mode, dst_bucket)

                # counters
                if ch == "k":
                    moved["kept_move"] += 1
                elif ch == "m":
                    moved["maybe_move"] += 1
                elif ch == "r":
                    moved["reject_move"] += 1

            elif ch == "s":
                skipped += 1

            elif ch == "q":
                print("\nQuit.")
                break

            else:
                # any other key = skip
                skipped += 1

        # Stop all splits if user quit
        if ch == "q":
            break

    print("\nDone.")
    print(f"Review bucket: {REVIEW_BUCKET}")
    print(f"MOVED to kept: {moved['kept_move']}")
    print(f"MOVED to maybe: {moved['maybe_move']}")
    print(f"MOVED to reject: {moved['reject_move']}")
    print(f"Skipped: {skipped}")

if __name__ == "__main__":
    main()

