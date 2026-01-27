# AIClub_IAM3D

DATA PROCESSING
1. Load raw dataset in yolov8 folder structure
2. Relabel all labels to one object class 0 using 'relabelling_to_zero.py'
3. 'restructure_predictions.py' runs a sand detection model on the dataset, moves into 'keep', 'maybe', 'reject' buckets depending on sand confidence score 
4. 'manual_filter.py' manually goes through the 'keep' and 'maybe' buckets and select final images in a 'kept' folder