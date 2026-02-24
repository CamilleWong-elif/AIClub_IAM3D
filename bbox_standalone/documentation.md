# Bounding Box Algorithm — Public Documentation

This document describes the **bbox_standalone** module: what it does, how to use it, and how to change behavior via configuration.

---

## What This Module Does

The module provides a full pipeline for **detection** and **multi-object tracking (MOT)** on video:

1. **Extract frames** from a video into image files.
2. **Run a detector** (YOLO, CNN, or mock) on those images and save YOLO-format labels.
3. **Process a video in one pass**: run the detector frame-by-frame and save all bounding boxes to a single JSON file (no intermediate image folder).
4. **Track objects**: re-read the video and the saved boxes, assign consistent **track IDs** across frames using IoU-based association, and optionally render an output video with boxes and IDs drawn.

All of this is configurable via **config.json** in the same folder.

---

## Files

| File | Purpose |
|------|--------|
| **mot_analysis.py** | Main controller: video/image handling, detection, NMS, and tracking. |
| **tracker_utils.py** | Tracking math: IoU between boxes, cost matrix, and track–detection association. |
| **config.json** | Configuration (detector, thresholds, paths). Edit this to change behavior. |

---

## Functions in `mot_analysis.py`

### `images(video_path, output_dir, config_path=None, frame_skip=None, image_ext="jpg")`

Converts a video file into a folder of images.

- **video_path**: Path to the input video.
- **output_dir**: Folder where frame images will be saved (e.g. `frames/`). Created if it doesn’t exist.
- **config_path**: Optional. Path to a JSON config file; if omitted, uses `bbox_standalone/config.json`.
- **frame_skip**: Optional. If set (e.g. `2`), only every Nth frame is saved. If not set, the value from config `video.frame_skip` is used (default 1 = every frame).
- **image_ext**: File extension for images (e.g. `"jpg"`, `"png"`).

**Returns:** The number of images written.

**Use case:** Getting a set of frames for training, manual labeling, or running `labels()` on a folder.

---

### `labels(input_images_dir, output_labels_dir, config_path=None)`

Runs the configured detector on each image in a folder, applies NMS, and writes **YOLO-format** label files.

- **input_images_dir**: Folder containing images (e.g. from `images()`). Supports `.jpg`, `.jpeg`, `.png`.
- **output_labels_dir**: Folder where `.txt` label files will be written. Each file has the same base name as the image (e.g. `frame_000001.jpg` → `frame_000001.txt`).
- **config_path**: Optional. Path to config.json.

Label format: one line per object: `class_id cx cy w h` (normalized 0–1). This matches common YOLO datasets and the project’s `RoverDataset`.

**Returns:** The number of label files written.

**Use case:** Turning a folder of images into detection labels (numbers/tensors in file form) for training or evaluation.

---

### `process_video(video_path, output_boxes_path, config_path=None)`

Processes a video **without** saving intermediate images: reads the video, runs the detector and NMS on each frame, and saves all bounding boxes to one JSON file.

- **video_path**: Path to the input video.
- **output_boxes_path**: Path for the output JSON (e.g. `boxes.json`).
- **config_path**: Optional. Path to config.json.

**Returns:** A small dict with `num_frames`, `frame_size`, and `output_path`.

**Output JSON shape:**

- `frame_size`: `[width, height]`
- `num_frames`: Number of frames.
- `frames`: List of frames; each frame is a list of objects with `x1`, `y1`, `x2`, `y2`, `score`, `class_id`.

This file is the input for `run_tracking()`.

**Use case:** Fast way to get per-frame detections and later run tracking or analysis without re-running the detector.

---

### `run_tracking(video_path, boxes_path, output_path, config_path=None, write_video=True)`

Re-reads the video and the AI-generated boxes (from `process_video`), associates detections across frames to assign **track IDs**, and writes the result.

- **video_path**: Same video that was used to generate the boxes.
- **boxes_path**: JSON file produced by `process_video()`.
- **output_path**: Base path for outputs:
  - `output_path.json` (or `output_path` if it already ends in `.json`): tracked boxes per frame, each with `track_id`.
  - If `write_video=True`: `output_path.mp4` (or same base as `output_path`) with bounding boxes and track IDs drawn.
- **config_path**: Optional. Path to config.json.
- **write_video**: If `True`, an output video is rendered; if `False`, only the JSON is written.

**Returns:** A dict with `num_frames`, `num_tracks`, `output_json_path`, and (if a video was written) `output_video_path`.

**Use case:** Turn per-frame detections into consistent identities over time and visualize or export tracked results.

---

## Functions in `tracker_utils.py`

### `bbox_iou(box1, box2)`

Measures **overlap** between two bounding boxes using Intersection over Union (IoU).

- **box1**, **box2**: Can be objects with `.x1`, `.y1`, `.x2`, `.y2`, or 4-element sequences `(x1, y1, x2, y2)`, or dicts with keys `x1`, `y1`, `x2`, `y2`.
- **Returns:** A number between 0 (no overlap) and 1 (identical boxes).

Used inside the tracker and available for custom analysis.

---

### `cost_matrix_iou(tracks, detections)`

Builds a **cost matrix** between existing tracks and new detections.

- **tracks**: List of track boxes (same formats as in `bbox_iou`).
- **detections**: List of detection boxes.

**Returns:** A 2D numpy array of shape `(len(tracks), len(detections))`. Entry `[i, j]` is `1 - IoU(tracks[i], detections[j])`. Minimizing total cost is equivalent to maximizing total IoU.

Used by `track_association()`; can be used for custom assignment logic.

---

### `track_association(tracks, detections, iou_th=0.3)`

Finds the **best matchings** between tracks and detections using IoU.

- **tracks**: List of current track boxes.
- **detections**: List of detection boxes for the current frame.
- **iou_th**: Minimum IoU to accept a match (default `0.3`). Pairs below this are treated as unmatched.

**Returns:** A tuple of three lists:

1. **matches**: `(track_index, detection_index)` for each assigned pair.
2. **unmatched_track_indices**: Track indices that did not get a detection.
3. **unmatched_detection_indices**: Detection indices that did not match any track (candidates for new tracks).

Used by `run_tracking()` in `mot_analysis.py` to link detections across frames.

---

## Configuration: `config.json`

All tunable behavior is in **bbox_standalone/config.json**. Changing these values changes how the pipeline runs **without** editing Python code.

### Top-level

| Key | Meaning |
|-----|--------|
| **device** | Device for the detector: `"cpu"` or `"cuda"` (if available). |

### `video`

| Key | Meaning | Typical effect |
|-----|--------|----------------|
| **frame_skip** | Use every Nth frame when extracting images (e.g. `2` = every 2nd frame). | Fewer frames → faster and less data. |
| **image_ext** | Extension for extracted images (`"jpg"`, `"png"`). | Affects disk size and compatibility. |
| **fps** | FPS used when writing output video in `run_tracking`. | Only affects rendered video. |

### `detector`

| Key | Meaning | Effect |
|-----|--------|--------|
| **type** | `"mock"` \| `"cnn"` \| `"yolo"` | Which detector runs in `labels()` and `process_video()`. |
| **model_path** | Path to the model file. | For `"yolo"`: e.g. `"yolov8n.pt"`. Ignored for mock/CNN. |

- **mock**: Fixed example boxes; no real detection. Good for testing the pipeline.
- **cnn**: Uses the project’s CNN detector (same as in the main codebase).
- **yolo**: Uses Ultralytics YOLO; requires `pip install ultralytics` and a valid **model_path**.

### `nms`

| Key | Meaning | Effect |
|-----|--------|--------|
| **iou_threshold** | NMS overlap threshold (e.g. `0.5`). | Higher → more overlapping boxes removed. |
| **score_threshold** | Minimum confidence to keep a box (e.g. `0.3`). | Higher → fewer, more confident boxes. |

These control how many and which detections are kept after the detector runs.

### `tracking`

| Key | Meaning | Effect |
|-----|--------|--------|
| **iou_threshold** | Minimum IoU to accept a track–detection match (e.g. `0.3`). | **Higher** → stricter matching, fewer ID switches but more “new” tracks when objects move fast. **Lower** → more matches, but more risk of wrong associations when objects are close. |

### `paths`

Default path hints (e.g. default folders for frames/labels or default output filenames). The code may use these when you don’t pass explicit paths; exact usage depends on how you call the functions.

---

## How Parameters Affect the Algorithm

- **Detector type and model_path**: Determine **what** is detected (e.g. YOLO classes vs project CNN classes) and **speed/accuracy**.
- **NMS (nms.iou_threshold, nms.score_threshold)**: Control **how many boxes per frame** and their quality. Stricter NMS → fewer, cleaner boxes; looser → more boxes, possible duplicates.
- **Tracking (tracking.iou_threshold)**: Controls **how easily** a detection is assigned to an existing track vs starting a new track. This directly affects **identity consistency** over time.

Changing these in **config.json** is the intended way to tune the bounding box and tracking behavior for your data and use case.
