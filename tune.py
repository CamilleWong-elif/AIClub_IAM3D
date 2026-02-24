from ultralytics import YOLO

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def main():
    DATA_YAML = r"C:\github\AIClub_IAM3D\Beach and sand.v1i.yolov8\data.yaml"
    model = YOLO("yolov8m.pt")
    model.tune(data = DATA_YAML, epochs = 20, iterations = 30, optimizer="AdamW", plots = True, save =True)
    
if __name__ == "__main__":
    main()