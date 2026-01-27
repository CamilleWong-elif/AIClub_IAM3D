'''
For datasets already in yolov8 folder structure.


input Yolov8 folder structure expected:
ROOT/
  test/
    images/
    labels/
  train/
    images/
    labels/
  valid/
    images/
    labels/

output:
relabels all label classes to one object class 0
'''


from pathlib import Path

# -------------------------
# CONFIG
# -------------------------
DATASET_ROOT = Path("/Users/kikihan/Desktop/IAM3D/raw_data/trash_bbox")  # change this to raw dataset root path

SPLITS = ["train", "valid", "test"]
LABELS_DIR_NAME = "labels"
overwrite_class_id = 0  # The class ID to set for all boxes, obj is class 0 

# -------------------------
# Main
# -------------------------
total_files = 0
total_boxes = 0

for split in SPLITS:
    labels_dir = DATASET_ROOT / split / LABELS_DIR_NAME

    if not labels_dir.exists():
        print(f"Skipping {split}: no labels dir found")
        continue

    label_files = list(labels_dir.glob("*.txt"))
    print(f"{split}: found {len(label_files)} label files")

    for label_path in label_files:
        lines = label_path.read_text().strip().splitlines()

        if not lines:
            continue  # empty label file

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue  # malformed line, skip

            # parts[0] is class id — overwrite to overwrite_class_id
            new_line = f"{overwrite_class_id} " + " ".join(parts[1:])
            new_lines.append(new_line)
            total_boxes += 1

        label_path.write_text("\n".join(new_lines) + "\n")
        total_files += 1

print("\nDone.")
print(f"Rewritten label files: {total_files}")
print(f"Total boxes rewritten: {total_boxes}")
