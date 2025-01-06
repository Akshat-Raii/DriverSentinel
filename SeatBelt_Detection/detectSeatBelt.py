from ultralytics import YOLO
import cv2
import cvzone
import math
 
cap = cv2.VideoCapture(0)  # For Webcam
cap.set(3, 1280)
cap.set(4, 720)
# cap = cv2.VideoCapture("C:/Users/aksha/Downloads/istockphoto-1433925577-640_adpp_is.mp4")  # For Video
 
model = YOLO("D:/DriverSentinal/SeatBelt Detection/runs/detect/train/weights/best.pt")
 
classNames=['phone', 'seatbelt']
myColor = (0, 0, 255)
while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # bounding box surrounding the detected object
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
 
            # confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # class name
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            print(currentClass)
            if conf>0.6:
                if currentClass =='seatbelt' :
                    myColor = (255, 0,255)
                else:
                    myColor = (255, 0, 0)
 
                cvzone.putTextRect(img, f'{classNames[cls]} {conf}',
                                   (max(0, x1), max(35, y1)), scale=1, thickness=1,colorB=myColor,
                                   colorT=(255,255,255),colorR=myColor, offset=5)
                cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)
 
    cv2.imshow("Image", img)
    cv2.waitKey(1)