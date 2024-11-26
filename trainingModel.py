from ultralytics import YOLO

model = YOLO("yolov8s.pt")

model.train(data="datasets_rotation2/data.yaml",epochs=300,imgsz=640,project="runs_300_rotation2")