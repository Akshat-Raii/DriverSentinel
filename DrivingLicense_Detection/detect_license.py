from ultralytics import YOLO
model=YOLO("runs/detect/train/weights/best.pt")
results=model("dataset/test/images/1_jpg.rf.8478661f116d04ddc6635023ad02b7d8.jpg")
results[0].show()