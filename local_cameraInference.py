import argparse
import os

import cv2
import torch
from ultralytics import YOLO

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def source_type(value):
    """Accept both int (camera index) and string (file path)"""
    try:
        return int(value)
    except ValueError:
        return value

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLOv8 webcam inference")
    parser.add_argument("--model", type=str, default="best_syn.pt", help="Path to YOLOv8 model")
    parser.add_argument(
        "--source",
        type=source_type,
        default=1,
        help="Camera source: local camera index (default: 1), video file path, or Raspberry Pi stream URL. "
             "Pi examples (172.20.10.6): 'rtsp://172.20.10.6:8554/cam' or 'http://172.20.10.6:8080/?action=stream'"
    )    
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.01, help="Confidence threshold")
    parser.add_argument(
        "--display-conf",
        type=float,
        default=0,
        help="Only draw detections with confidence >= this value (default: --conf)",
    )
    parser.add_argument(
        "--max-box-area-ratio",
        type=float,
        default=0.001,
        help="Reject detections with box area ratio above this value (0.0 to 1.0)",
    )
    parser.add_argument("--iou", type=float, default=0.75, help="NMS IoU threshold")
    parser.add_argument("--inference_frame", type=float, default=1, help="Feed forward every N frames")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    display_conf = args.display_conf if args.display_conf is not None else args.conf
    max_box_area_ratio = min(max(args.max_box_area_ratio, 0.0), 1.0)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = YOLO(args.model)
    cap = cv2.VideoCapture(args.source)

    if not cap.isOpened():
        error_msg = f"Could not open source: {args.source}\n"
        error_msg += "\nConnection options:\n"
        error_msg += "  - Local camera: --source 0  (or 1, 2, etc.)\n"
        error_msg += "  - Video file: --source path/to/video.mp4\n"
        error_msg += "  - Raspberry Pi (RTSP): --source 'rtsp://<pi-ip>:8554/cam'\n"
        error_msg += "  - Raspberry Pi (MJPEG): --source 'http://<pi-ip>:8080/?action=stream'\n"
        error_msg += "\nFor Raspberry Pi: ensure the streaming service is running on the Pi first."
        raise RuntimeError(error_msg)

    print(f"Running webcam inference on {device}.")
    print(f"  Press 'q' to quit")
    print(f"  Press '1' to toggle bbox display")
    print(f"  Press '+' to increase confidence threshold")
    print(f"  Press '-' to decrease confidence threshold")
    print(f"  Current conf: {display_conf:.2f}")

    frame_num = 0
    results = None
    show_bbox = True
    while True:

        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame from camera.")
            break

        frame_h, frame_w = frame.shape[:2]
        frame_area = float(frame_w * frame_h)

        if frame_num % args.inference_frame == 0:
            results = model.predict(
                source=frame,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=device,
                verbose=False,
            )

            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                confs = result.boxes.conf.cpu().tolist()
                clss = result.boxes.cls.cpu().tolist()
                boxes = result.boxes.xyxy.cpu().tolist()

                for i, (conf, cls_id, box) in enumerate(zip(confs, clss, boxes), 1):
                    x1, y1, x2, y2 = box
                    box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                    box_area_ratio = box_area / frame_area if frame_area > 0 else 0.0


        frame_num += 1

        annotated = frame.copy()
        if results is not None and show_bbox:
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                confs = result.boxes.conf.cpu().tolist()
                clss = result.boxes.cls.cpu().tolist()
                boxes = result.boxes.xyxy.cpu().tolist()
                names = result.names

                for conf, cls_id, box in zip(confs, clss, boxes):
                    if conf < display_conf:
                        continue

                    x1f, y1f, x2f, y2f = box
                    box_area = max(0.0, x2f - x1f) * max(0.0, y2f - y1f)
                    box_area_ratio = box_area / frame_area if frame_area > 0 else 0.0
                    if box_area_ratio > max_box_area_ratio:
                        continue

                    x1, y1, x2, y2 = map(int, box)
                    label_name = names.get(int(cls_id), str(int(cls_id)))
                    label = f"{label_name} {conf:.2f}"
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        annotated,
                        '_',
                        (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
        try:
            cv2.imshow("YOLOv8 Camera Inference", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("1"):
                show_bbox = not show_bbox
                print(f"Bbox display: {'ON' if show_bbox else 'OFF'}")
            elif key == ord("+") or key == ord("="):
                display_conf = min(display_conf + 0.01, 1.0)
                print(f"Confidence threshold: {display_conf:.2f}")
            elif key == ord("-") or key == ord("_"):
                display_conf = max(display_conf - 0.01, 0.0)
                print(f"Confidence threshold: {display_conf:.2f}")
        except cv2.error as e:
            print(f"Display warning: {e}")
            print("Continuing without display...")
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("1"):
                show_bbox = not show_bbox
                print(f"Bbox display: {'ON' if show_bbox else 'OFF'}")
            elif key == ord("+") or key == ord("="):
                display_conf = min(display_conf + 0.01, 1.0)
                print(f"Confidence threshold: {display_conf:.2f}")
            elif key == ord("-") or key == ord("_"):
                display_conf = max(display_conf - 0.01, 0.0)
                print(f"Confidence threshold: {display_conf:.2f}")

    cap.release()
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass


if __name__ == "__main__":
    main()
