from ultralytics import YOLO


def main():
    DATA_YAML = r"C:\github\dataset\Beach and sand.v1140i.yolov8\data.yaml"

    model = YOLO(r"C:\github\AIClub_IAM3D\runs\detect\train6\weights\best.pt")  # Load your best trained model
    metrics = model.val(data=DATA_YAML)
    print(metrics)

if __name__ == '__main__':
    main()