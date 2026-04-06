"""
Laptop-side template pipeline for object detection.

Data flow:
    (Wi-Fi video stream from Pi) -> decode frame -> YOLO inference -> detections

Key design goal:
    LOW LATENCY: always process the *latest* frame, do NOT build up a queue of old frames.
"""

import time
import threading
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO


# ----------------------------
# 1) Data structures
# ----------------------------

@dataclass
class FramePacket:
    """A frame + metadata. Metadata matters for robotics."""
    frame_id: int
    timestamp_s: float
    bgr: np.ndarray


# ----------------------------
# 2) "Latest frame" buffer (thread-safe)
# ----------------------------

class LatestFrameBuffer:
    """
    Holds only the most recent frame packet.
    Capture thread overwrites it.
    Inference thread reads it.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._latest: Optional[FramePacket] = None

    def put(self, pkt: FramePacket) -> None:
        with self._lock:
            self._latest = pkt

    def get(self) -> Optional[FramePacket]:
        with self._lock:
            return self._latest


# ----------------------------
# 3) Capture/Receive thread
# ----------------------------

def capture_thread(stream_url, buffer: LatestFrameBuffer, stop_event: threading.Event) -> None:
    """
    Connects to the Pi video stream over Wi-Fi and continuously grabs frames.

    stream_url examples:
      - RTSP: "rtsp://<pi-ip>:8554/cam"
      - MJPEG: "http://<pi-ip>:8080/?action=stream"
      - local webcam test: 0
    """
    cap = None
    frame_id = 0
    consecutive_failures = 0

    while not stop_event.is_set():
        if cap is None or not cap.isOpened():
            if cap is not None:
                cap.release()
            cap = cv2.VideoCapture(stream_url)
            if not cap.isOpened():
                time.sleep(0.1)
                continue
            consecutive_failures = 0

        ok, frame = cap.read()
        if not ok or frame is None:
            consecutive_failures += 1
            if consecutive_failures >= 5:
                cap.release()
                cap = None
                consecutive_failures = 0
            time.sleep(0.1)
            continue

        consecutive_failures = 0

        pkt = FramePacket(
            frame_id=frame_id,
            timestamp_s=time.monotonic(),
            bgr=frame
        )
        buffer.put(pkt)
        frame_id += 1

    if cap is not None:
        cap.release()


# ----------------------------
# 4) YOLO model runner
# ----------------------------

class YOLOPtModel:
    """
    Runs Ultralytics YOLO using a .pt checkpoint like best.pt
    """
    def __init__(self, weights_path: str, imgsz: int = 640, conf: float = 0.25):
        self.model = YOLO(weights_path)
        self.imgsz = imgsz
        self.conf = conf

        # class id -> class name
        self.names = self.model.names

    def __call__(self, bgr_frame: np.ndarray):
        """
        Accepts one OpenCV BGR frame and returns Ultralytics Results.
        Ultralytics handles resize/letterbox/preprocessing internally.
        """
        results = self.model.predict(
            source=bgr_frame,
            imgsz=self.imgsz,
            conf=self.conf,
            verbose=False
        )
        return results[0]


# ----------------------------
# 5) Drawing helper
# ----------------------------

def draw_detections(frame: np.ndarray, result, class_names) -> np.ndarray:
    """
    Draw boxes from one Ultralytics result onto a copy of frame.
    """
    vis = frame.copy()

    if result.boxes is None or len(result.boxes) == 0:
        return vis

    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    clss = result.boxes.cls.cpu().numpy().astype(int)

    for box, conf, cls_id in zip(boxes_xyxy, confs, clss):
        x1, y1, x2, y2 = map(int, box)
        label = f"{class_names[cls_id]} {conf:.2f}"

        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            vis,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    return vis


# ----------------------------
# 6) Main inference loop
# ----------------------------

def main():
    stream_url = 0                   # replace with Pi stream URL later
    weights_path = "/Users/kikihan/AIClub_IAM3D/best.pt"         # your trained YOLO checkpoint
    model_imgsz = 640                # usually matches training/inference size

    buffer = LatestFrameBuffer()
    stop_event = threading.Event()

    # Start capture thread
    t = threading.Thread(
        target=capture_thread,
        args=(stream_url, buffer, stop_event),
        daemon=True
    )
    t.start()

    model = YOLOPtModel(weights_path=weights_path, imgsz=model_imgsz, conf=0.25)

    last_seen_id = -1
    fps_counter_t0 = time.monotonic()
    frames_processed = 0

    try:
        while True:
            pkt = buffer.get()
            if pkt is None:
                time.sleep(0.005)
                continue

            # Skip duplicate frame if inference is slower than capture
            if pkt.frame_id == last_seen_id:
                time.sleep(0.001)
                continue
            last_seen_id = pkt.frame_id

            # --- Inference directly on raw frame ---
            result = model(pkt.bgr)

            # --- Visualization ---
            vis = draw_detections(pkt.bgr, result, model.names)

            cv2.putText(
                vis,
                f"frame_id={pkt.frame_id}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2
            )

            cv2.imshow("Laptop stream (detections)", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # FPS tracking
            frames_processed += 1
            now = time.monotonic()
            if now - fps_counter_t0 >= 1.0:
                print(f"Pipeline FPS: {frames_processed / (now - fps_counter_t0):.1f}")
                fps_counter_t0 = now
                frames_processed = 0

    finally:
        stop_event.set()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()