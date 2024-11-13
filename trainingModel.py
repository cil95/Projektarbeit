from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(data="datasets/data.yaml",epochs=200,imgsz=640)