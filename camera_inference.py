import argparse
import os

import cv2
import torch
from ultralytics import YOLO

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLOv8 webcam inference")
    parser.add_argument("--model", type=str, default="best.pt", help="Path to YOLOv8 model")
    parser.add_argument("--source", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument(
        "--display-conf",
        type=float,
        default=0,
        help="Only draw detections with confidence >= this value (default: --conf)",
    )
    parser.add_argument(
        "--max-box-area-ratio",
        type=float,
        default=1,
        help="Reject detections with box area ratio above this value (0.0 to 1.0)",
    )
    parser.add_argument("--iou", type=float, default=1, help="NMS IoU threshold")
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
        raise RuntimeError(
            f"Could not open camera index {args.source}. Try another index like --source 1"
        )

    print(f"Running webcam inference on {device}. Press 'q' to quit.")

    frame_num = 0
    results = None
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
        if results is not None:
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
                        label,
                        (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
        cv2.imshow("YOLOv8 Camera Inference", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
