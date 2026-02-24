# train_yolo.py
from ultralytics import YOLO

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def main():
    DATA_YAML = r"C:\github\AIClub_IAM3D\Beach and sand.v1i.yolov8\data.yaml"
    model = YOLO("yolov8m.pt")
    model.tune(data = DATA_YAML, epochs = 20, iterations = 30, optimizer="AdamW", plots = True, save =True)

    # model.train(
    #     data=DATA_YAML,

    #     # core training params
    #     epochs=120,
    #     imgsz=512,           # Chosen image size
    #     batch=8,
    #     device=0,            # GPU usage
    #     workers=0,           # Windows-safe. If stable, try 1-2.

    #     # optimizer 
    #     optimizer='SGD',
    #     lr0=0.003,
    #     # lrf=1,            
    #     # cos_lr=True,
    #     # warmup_epochs=3,
    #     # weight_decay=5e-4,

    #     # augmentations
    #     mosaic=1.0,          
    #     close_mosaic=10,    
    #     mixup=0.05,
    #     copy_paste=0.0,
    #     fliplr=0.5,

    #     # training behavior
    #     patience=15,         # early stopping patience
    #     save=True,           # saves best.pt and last.pt
    #     plots=True,          # store training plots/label previews
    #     cache=True,          # set True to cache images in RAM (big memory)
    #     amp=True,            # use mixed precision (faster on GPU)
    # )


if __name__ == "__main__":
    main()
