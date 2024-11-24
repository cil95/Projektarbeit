from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(data="datasets_rotation/data.yaml",epochs=300,imgsz=640,project="runs_300_rotation")