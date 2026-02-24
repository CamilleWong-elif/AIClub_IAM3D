"""
MOT (Multi-Object Tracking) analysis: video → images, detection labels, process_video,
and tracking that re-reads video and associates AI-generated boxes across frames.
"""

import json
import os
import sys
from typing import List, Dict, Any, Optional

# Allow importing from project root and from this package
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from tracker_utils import track_association, bbox_iou

# Project imports (core, detectors)
from core.bbox import BoundingBox
from core.engine import BoundingBoxEngine
from core.nms import non_max_suppression


# ---------- Config ----------
def _load_config(config_path: Optional[str] = None) -> dict:
    path = config_path or os.path.join(_SCRIPT_DIR, "config.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_detector(cfg: dict):
    """Build detector from config (yolo, cnn, or mock)."""
    det_cfg = cfg.get("detector", {})
    dtype = det_cfg.get("type", "mock").lower()
    device = cfg.get("device", "cpu")

    if dtype == "yolo":
        try:
            from ultralytics import YOLO
            model_path = det_cfg.get("model_path", "yolov8n.pt")
            model = YOLO(model_path)
            return _YOLOWrapper(model, device)
        except ImportError:
            raise ImportError("YOLO detector requires 'ultralytics'. Install with: pip install ultralytics")
    if dtype == "cnn":
        from models.cnn_detector import CNNDetector
        return CNNDetector(device=device)
    if dtype == "mock":
        from detectors.mock_detector import MockDetector
        return MockDetector()
    raise ValueError(f"Unknown detector type: {dtype}")


class _YOLOWrapper:
    """Wraps ultralytics YOLO to return List[BoundingBox] (xyxy, score, class_id)."""
    def __init__(self, model, device: str):
        self.model = model
        self.device = device

    def detect(self, image):
        if image is None:
            return []
        import cv2
        if hasattr(image, "size"):  # PIL
            import numpy as np
            frame = np.array(image)
            frame = frame[:, :, ::-1]  # RGB -> BGR
        else:
            frame = image
        results = self.model(frame, device=self.device, verbose=False)
        boxes = []
        for r in results:
            if r.boxes is None:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            conf = r.boxes.conf.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            for i in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[i]
                boxes.append(BoundingBox(float(x1), float(y1), float(x2), float(y2), float(conf[i]), int(cls[i])))
        return boxes


def _box_to_yolo_line(box: BoundingBox, img_w: int, img_h: int) -> str:
    """Convert one BoundingBox to YOLO format: class_id cx cy w h (normalized)."""
    cx = (box.x1 + box.x2) / 2.0
    cy = (box.y1 + box.y2) / 2.0
    w = box.x2 - box.x1
    h = box.y2 - box.y1
    cx_n = cx / img_w
    cy_n = cy / img_h
    w_n = w / img_w
    h_n = h / img_h
    return f"{box.class_id} {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}\n"


def _box_to_dict(box: BoundingBox) -> dict:
    return {"x1": box.x1, "y1": box.y1, "x2": box.x2, "y2": box.y2, "score": box.score, "class_id": box.class_id}


# ---------- Public API ----------

def images(
    video_path: str,
    output_dir: str,
    config_path: Optional[str] = None,
    frame_skip: Optional[int] = None,
    image_ext: str = "jpg",
) -> int:
    """
    Convert a video file into a folder of images (one per frame, or every frame_skip frames).

    Args:
        video_path: Path to the input video file.
        output_dir: Directory to write frame images (created if needed).
        config_path: Optional path to config.json; uses bbox_standalone/config.json if not set.
        frame_skip: If set, write every Nth frame (1 = every frame, 2 = every 2nd, etc.).
        image_ext: Image extension (e.g. 'jpg', 'png').

    Returns:
        Number of images written.
    """
    import cv2
    cfg = _load_config(config_path)
    skip = frame_skip if frame_skip is not None else cfg.get("video", {}).get("frame_skip", 1)
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    count = 0
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % skip == 0:
            out_name = f"frame_{count:06d}.{image_ext}"
            out_path = os.path.join(output_dir, out_name)
            cv2.imwrite(out_path, frame)
            count += 1
        frame_idx += 1
    cap.release()
    return count


def labels(
    input_images_dir: str,
    output_labels_dir: str,
    config_path: Optional[str] = None,
) -> int:
    """
    Run the configured detector on each image in input_images_dir, apply NMS,
    and write YOLO-format label files (class_id cx cy w h normalized) to output_labels_dir.
    Label filenames match image basenames with .txt extension.

    Args:
        input_images_dir: Folder containing images (.jpg, .png, .jpeg).
        output_labels_dir: Folder where .txt label files will be written.
        config_path: Optional path to config.json.

    Returns:
        Number of label files written.
    """
    import cv2
    cfg = _load_config(config_path)
    detector = _get_detector(cfg)
    nms_cfg = cfg.get("nms", {})
    iou_th = nms_cfg.get("iou_threshold", 0.5)
    score_th = nms_cfg.get("score_threshold", 0.3)
    engine = BoundingBoxEngine(iou_thresh=iou_th, score_thresh=score_th)

    os.makedirs(output_labels_dir, exist_ok=True)
    exts = (".jpg", ".jpeg", ".png")
    names = sorted(
        f for f in os.listdir(input_images_dir)
        if f.lower().endswith(exts)
    )
    written = 0
    for name in names:
        img_path = os.path.join(input_images_dir, name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        raw = detector.detect(img)
        final = engine.process(raw)
        base = os.path.splitext(name)[0]
        label_path = os.path.join(output_labels_dir, base + ".txt")
        with open(label_path, "w", encoding="utf-8") as f:
            for box in final:
                f.write(_box_to_yolo_line(box, w, h))
        written += 1
    return written


def process_video(
    video_path: str,
    output_boxes_path: str,
    config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Process a video: run detector + NMS per frame and save bounding boxes to a single JSON file.
    Does not save intermediate images. Output can be used by run_tracking().

    Args:
        video_path: Path to the input video.
        output_boxes_path: Path to write JSON (e.g. boxes.json) with per-frame boxes and metadata.
        config_path: Optional path to config.json.

    Returns:
        Metadata dict with num_frames, frame_size, output_path.
    """
    import cv2
    cfg = _load_config(config_path)
    detector = _get_detector(cfg)
    nms_cfg = cfg.get("nms", {})
    iou_th = nms_cfg.get("iou_threshold", 0.5)
    score_th = nms_cfg.get("score_threshold", 0.3)
    engine = BoundingBoxEngine(iou_thresh=iou_th, score_thresh=score_th)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    frame_size = None
    frames_boxes = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_size is None:
            frame_size = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        raw = detector.detect(frame_rgb)
        final = engine.process(raw)
        frames_boxes.append([_box_to_dict(b) for b in final])
    cap.release()

    out_data = {
        "frame_size": frame_size,
        "num_frames": len(frames_boxes),
        "frames": frames_boxes,
    }
    os.makedirs(os.path.dirname(os.path.abspath(output_boxes_path)) or ".", exist_ok=True)
    with open(output_boxes_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)
    return {"num_frames": len(frames_boxes), "frame_size": frame_size, "output_path": output_boxes_path}


def run_tracking(
    video_path: str,
    boxes_path: str,
    output_path: str,
    config_path: Optional[str] = None,
    write_video: bool = True,
) -> Dict[str, Any]:
    """
    Re-read the video and the AI-generated boxes JSON (from process_video), run track association
    across frames to assign consistent track IDs, and write the result (tracked boxes JSON and
    optionally a video with boxes and IDs drawn).

    Args:
        video_path: Path to the same video used to generate boxes_path.
        boxes_path: JSON file produced by process_video (frames of detections).
        output_path: Base path for outputs: output_path.json (tracked per-frame boxes) and
                     if write_video, output_path.mp4 (video with drawn boxes and track IDs).
        config_path: Optional path to config.json.
        write_video: If True, render an output video with bounding boxes and track IDs drawn.

    Returns:
        Metadata with num_frames, num_tracks, output_json_path, output_video_path (if written).
    """
    import cv2
    cfg = _load_config(config_path)
    with open(boxes_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    frames_boxes = data["frames"]
    frame_size = data.get("frame_size", [640, 480])
    iou_th = cfg.get("tracking", {}).get("iou_threshold", 0.3)

    # Active tracks: list of dicts with track_id and box (x1,y1,x2,y2,score,class_id)
    next_track_id = 0
    active_tracks = []  # each: {"track_id": int, "x1", "y1", "x2", "y2", "score", "class_id"}
    tracked_frames = []  # per frame: list of {track_id, x1, y1, x2, y2, score, class_id}
    all_track_ids_seen = set()

    for frame_dets in frames_boxes:
        det_boxes = [{"x1": d["x1"], "y1": d["y1"], "x2": d["x2"], "y2": d["y2"]} for d in frame_dets]
        track_boxes = [{"x1": t["x1"], "y1": t["y1"], "x2": t["x2"], "y2": t["y2"]} for t in active_tracks]
        matches, unmatched_track_idx, unmatched_det_idx = track_association(
            track_boxes, det_boxes, iou_th=iou_th
        )
        # Update matched tracks with new box from detection (keep full detection info)
        for ti, di in matches:
            det = frame_dets[di]
            active_tracks[ti] = {"track_id": active_tracks[ti]["track_id"], **det}
        # Remove unmatched tracks (optional: could keep for max_age frames)
        new_active = [active_tracks[i] for i in range(len(active_tracks)) if i not in unmatched_track_idx]
        # Add new tracks for unmatched detections
        for di in unmatched_det_idx:
            new_active.append({"track_id": next_track_id, **frame_dets[di]})
            all_track_ids_seen.add(next_track_id)
            next_track_id += 1
        active_tracks = new_active
        tracked_frames.append([dict(t) for t in active_tracks])

    out_json_path = output_path if output_path.lower().endswith(".json") else output_path + ".json"
    base = output_path if not output_path.lower().endswith(".json") else output_path[:-5]
    out_video_path = base + ".mp4" if not base.lower().endswith(".mp4") else base

    out_data = {
        "frame_size": frame_size,
        "num_frames": len(tracked_frames),
        "frames": tracked_frames,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_json_path)) or ".", exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)

    result_meta = {
        "num_frames": len(tracked_frames),
        "num_tracks": len(all_track_ids_seen),
        "output_json_path": out_json_path,
    }

    if write_video:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_w = frame_size[0]
            out_h = frame_size[1]
            writer = cv2.VideoWriter(out_video_path, fourcc, 30.0, (out_w, out_h))
            for frame_objs in tracked_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                for obj in frame_objs:
                    x1, y1 = int(obj["x1"]), int(obj["y1"])
                    x2, y2 = int(obj["x2"]), int(obj["y2"])
                    tid = obj["track_id"]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, str(tid), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                writer.write(frame)
            cap.release()
            writer.release()
        result_meta["output_video_path"] = out_video_path

    return result_meta
