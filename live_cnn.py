"""
Live camera feed + custom CNN detection + MOT tracking for Raspberry Pi 5 + Camera Module 3.

Branch: custom_bbox

Terminal controls:
    1 = Start recording
    2 = Stop recording
    3 = Save (shows file path + size)
    q = Quit

Usage:
    python live.py --weights runs/cnn/best.pt
    python live.py --weights runs/cnn/best.pt --resolution 640 480 --fps 30
    python live.py --weights runs/cnn/best.pt --no-display
"""

import os
import sys
import cv2
import time
import json
import argparse
import threading
from datetime import datetime

from detectors.cnn_detector import CNNDetector
from core.nms import non_max_suppression
from config.classes import TRASH_CLASSES
from tracker_utils import track_association


# ---------------------------------------------------------------------------
# Non-blocking keyboard input
# ---------------------------------------------------------------------------
class KeyboardListener:
    def __init__(self):
        self.last_key = None
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._listen, daemon=True)
        self._thread.start()

    def _listen(self):
        try:
            import termios, tty
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setcbreak(fd)
                while self._running:
                    ch = sys.stdin.read(1)
                    with self._lock:
                        self.last_key = ch
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        except Exception:
            while self._running:
                try:
                    line = input()
                    if line:
                        with self._lock:
                            self.last_key = line[0]
                except EOFError:
                    break

    def get_key(self):
        with self._lock:
            k = self.last_key
            self.last_key = None
            return k

    def stop(self):
        self._running = False


# ---------------------------------------------------------------------------
# Live tracker (incremental, frame-by-frame)
# ---------------------------------------------------------------------------
class LiveTracker:
    def __init__(self, iou_threshold=0.3):
        self.iou_th = iou_threshold
        self.next_track_id = 0
        self.active_tracks = []

    def update(self, detections):
        """
        Args:
            detections: list of BoundingBox objects (after NMS)
        Returns:
            list of dicts with track_id + box info
        """
        det_boxes = [{"x1": d.x1, "y1": d.y1, "x2": d.x2, "y2": d.y2} for d in detections]
        track_boxes = [{"x1": t["x1"], "y1": t["y1"], "x2": t["x2"], "y2": t["y2"]}
                       for t in self.active_tracks]

        matches, unmatched_track_idx, unmatched_det_idx = track_association(
            track_boxes, det_boxes, iou_th=self.iou_th
        )

        for ti, di in matches:
            d = detections[di]
            self.active_tracks[ti].update({
                "x1": d.x1, "y1": d.y1, "x2": d.x2, "y2": d.y2,
                "score": d.score, "class_id": d.class_id,
            })

        new_active = [self.active_tracks[i] for i in range(len(self.active_tracks))
                      if i not in set(unmatched_track_idx)]

        for di in unmatched_det_idx:
            d = detections[di]
            new_active.append({
                "track_id": self.next_track_id,
                "x1": d.x1, "y1": d.y1, "x2": d.x2, "y2": d.y2,
                "score": d.score, "class_id": d.class_id,
            })
            self.next_track_id += 1

        self.active_tracks = new_active
        return [dict(t) for t in self.active_tracks]


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
CLASS_COLORS = {
    0: (0, 255, 0),    # object — green
    1: (0, 255, 255),  # sand — yellow
    2: (0, 0, 255),    # large_collection — red
}


def draw_tracks(frame, tracks):
    for t in tracks:
        x1, y1 = int(t["x1"]), int(t["y1"])
        x2, y2 = int(t["x2"]), int(t["y2"])
        tid = t["track_id"]
        cls_id = t.get("class_id", 0)
        score = t.get("score", 0)

        color = CLASS_COLORS.get(cls_id, (255, 255, 255))
        cls_name = TRASH_CLASSES.get(cls_id, f"cls{cls_id}")

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"ID:{tid} {cls_name} {score:.2f}"
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    return frame


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Live CNN detection + tracking (custom_bbox branch)")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to CNN checkpoint (runs/cnn/best.pt)")
    parser.add_argument("--config", type=str, default="bbox_standalone/config.json",
                        help="Path to config.json for thresholds")
    parser.add_argument("--resolution", type=int, nargs=2, default=[640, 480],
                        help="Width Height (default: 640 480)")
    parser.add_argument("--fps", type=int, default=30,
                        help="Framerate (default: 30)")
    parser.add_argument("--save-dir", type=str, default="recordings",
                        help="Directory for recordings (default: recordings/)")
    parser.add_argument("--no-display", action="store_true",
                        help="Headless mode (no preview window)")
    return parser.parse_args()


def load_config(config_path):
    cfg = {
        "device": "cpu",
        "nms": {"iou_threshold": 0.5, "score_threshold": 0.3},
        "tracking": {"iou_threshold": 0.3},
    }
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg.update(json.load(f))
    return cfg


def main():
    args = parse_args()
    width, height = args.resolution

    # ---- Config ----
    cfg = load_config(args.config)
    nms_iou = cfg.get("nms", {}).get("iou_threshold", 0.5)
    nms_score = cfg.get("nms", {}).get("score_threshold", 0.3)
    track_iou = cfg.get("tracking", {}).get("iou_threshold", 0.3)
    device = cfg.get("device", "cpu")

    # ---- Load CNN detector ----
    print(f"Loading CNN detector (device={device}, weights={args.weights})")
    detector = CNNDetector(device=device, weights_path=args.weights)
    print("Detector loaded.")

    # ---- Init tracker ----
    tracker = LiveTracker(iou_threshold=track_iou)

    # ---- Init camera ----
    try:
        from picamera2 import Picamera2
    except ImportError:
        print("ERROR: picamera2 not installed. sudo apt install python3-picamera2")
        sys.exit(1)

    print(f"Starting camera: {width}x{height} @ {args.fps}fps")
    picam2 = Picamera2()
    video_config = picam2.create_video_configuration(
        main={"format": "RGB888", "size": (width, height)},
    )
    picam2.configure(video_config)
    picam2.start()
    time.sleep(1)

    # ---- State ----
    os.makedirs(args.save_dir, exist_ok=True)
    recording = False
    video_writer = None
    current_path = None
    last_saved_path = None

    kb = KeyboardListener()

    print("\n--- Live CNN detection + tracking running ---")
    print("1 = Record | 2 = Stop | 3 = Save | q = Quit\n")

    frame_count = 0
    fps = 0.0
    fps_timer = time.time()

    try:
        while True:
            # ---- Capture ----
            frame_rgb = picam2.capture_array("main")

            # ---- Detect + NMS ----
            raw_boxes = detector.detect(frame_rgb)
            final_boxes = non_max_suppression(raw_boxes, nms_iou, nms_score)

            # ---- Track ----
            tracked = tracker.update(final_boxes)

            # ---- Draw on BGR frame ----
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            frame_bgr = draw_tracks(frame_bgr, tracked)

            # ---- FPS ----
            frame_count += 1
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_timer = time.time()

            # ---- HUD ----
            status = f"FPS: {fps:.1f} | Tracks: {len(tracked)}"
            if recording:
                status += "  [REC]"
                cv2.circle(frame_bgr, (width - 30, 25), 8, (0, 0, 255), -1)
            cv2.putText(frame_bgr, status, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame_bgr, "1=Rec 2=Stop 3=Save q=Quit", (10, height - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            # ---- Record ----
            if recording and video_writer is not None:
                video_writer.write(frame_bgr)

            # ---- Display ----
            if not args.no_display:
                cv2.imshow("CNN Live", frame_bgr)
                cv2.waitKey(1)

            # ---- Keyboard ----
            key = kb.get_key()

            if key == "1" and not recording:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                current_path = os.path.join(args.save_dir, f"rec_{timestamp}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(current_path, fourcc, args.fps, (width, height))
                recording = True
                print(f"[REC] Started: {current_path}")

            elif key == "2" and recording:
                recording = False
                video_writer.release()
                video_writer = None
                last_saved_path = current_path
                print(f"[STOP] Stopped: {last_saved_path}")

            elif key == "3":
                if last_saved_path and os.path.exists(last_saved_path):
                    size_mb = os.path.getsize(last_saved_path) / (1024 * 1024)
                    print(f"[SAVED] {last_saved_path} ({size_mb:.1f} MB)")
                elif recording:
                    print("[INFO] Stop recording first (press 2)")
                else:
                    print("[INFO] No recording yet. Press 1 to start.")

            elif key == "q":
                print("\nQuitting...")
                break

    except KeyboardInterrupt:
        print("\nInterrupted.")

    finally:
        kb.stop()
        if recording and video_writer is not None:
            video_writer.release()
            print(f"[SAVED] Auto-saved: {current_path}")
        picam2.stop()
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()
