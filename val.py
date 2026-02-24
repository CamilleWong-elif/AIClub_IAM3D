from ultralytics import YOLO


def main():
    DATA_YAML = r"C:\github\AIClub_IAM3D\new_dataset\data.yaml"

    model = YOLO(r"C:\github\AIClub_IAM3D\runs\detect\train9\weights\best.pt")  # Load your best trained model
    metrics = model.val(data=DATA_YAML)
    print(metrics)

if __name__ == '__main__':
    main()