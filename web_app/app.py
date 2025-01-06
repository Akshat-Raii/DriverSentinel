import cv2
import numpy as np
import streamlit as st
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from threading import Thread
from playsound import playsound
from ultralytics import YOLO
import math
import cvzone


st.set_page_config(
    page_title="Driver Sentinel",
    page_icon="🚗", 
    layout="wide"
)

# Making count variable global so that we can reset only if it is open and do proper updations
count = 0

# Function to trigger alarm
def start_alarm(sound):
    playsound('data/alarm.mp3')

def drowsiness_detection(frame):
    global count
    classes = ['Closed', 'Open']
    face_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")
    left_eye_cascade = cv2.CascadeClassifier("data/haarcascade_lefteye_2splits.xml")
    right_eye_cascade = cv2.CascadeClassifier("data/haarcascade_righteye_2splits.xml")
    
    model = load_model("D:/DriverSentinal/web_app/models/trained_model.h5", compile=False)  # To Ignore the optimizer state we set compile=False
    alarm_on = False
    alarm_sound = "data/alarm.mp3"
    status1 = ''
    status2 = ''
    
    height = frame.shape[0]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        
        left_eye = left_eye_cascade.detectMultiScale(roi_gray)
        right_eye = right_eye_cascade.detectMultiScale(roi_gray)
        
        for (x1, y1, w1, h1) in left_eye:
            cv2.rectangle(roi_color, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 1)
            eye1 = roi_color[y1:y1 + h1, x1:x1 + w1]
            eye1 = cv2.resize(eye1, (145, 145))
            eye1 = eye1.astype('float') / 255.0
            eye1 = img_to_array(eye1)
            eye1 = np.expand_dims(eye1, axis=0)
            pred1 = model.predict(eye1)
            status1 = np.argmax(pred1)
            break

        for (x2, y2, w2, h2) in right_eye:
            cv2.rectangle(roi_color, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 1)
            eye2 = roi_color[y2:y2 + h2, x2:x2 + w2]
            eye2 = cv2.resize(eye2, (145, 145))
            eye2 = eye2.astype('float') / 255.0
            eye2 = img_to_array(eye2)
            eye2 = np.expand_dims(eye2, axis=0)
            pred2 = model.predict(eye2)
            status2 = np.argmax(pred2)
            break

        if status1 == 2 and status2 == 2:
            count += 1
            cv2.putText(frame, "Eyes Closed, Frame count: " + str(count), (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
            if count >= 15:
                cv2.putText(frame, "Drowsiness Alert!!!", (100, height - 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                if not alarm_on:
                    alarm_on = True
                    t = Thread(target=start_alarm, args=(alarm_sound,))
                    t.daemon = True
                    t.start()
        else:
            cv2.putText(frame, "Eyes Open", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
            count = 0
            alarm_on = False
    
    return frame

def license_detection(frame):
    model = YOLO("models/driving_license_model.pt")
    classNames = ['License number']
    myColor = (0, 0, 255)
    
    results = model(frame, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            if conf > 0.6:
                if currentClass == 'License number':
                    myColor = (255, 0, 255)
                else:
                    myColor = (255, 0, 0)

                cvzone.putTextRect(frame, f'{classNames[cls]} {conf}',
                                   (max(0, x1), max(35, y1)), scale=1, thickness=1, colorB=myColor,
                                   colorT=(255, 255, 255), colorR=myColor, offset=5)
                cv2.rectangle(frame, (x1, y1), (x2, y2), myColor, 3)
    return frame

def seatbelt_detection(frame):
    model = YOLO("models/seatbelt_model.pt")
    classNames = ['seatbelt']
    myColor = (0, 0, 255)
    
    results = model(frame, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            if conf > 0.6:
                if currentClass == 'seatbelt':
                    myColor = (255, 0, 255)
                else:
                    myColor = (255, 0, 0)

                cvzone.putTextRect(frame, f'{classNames[cls]} {conf}',
                                   (max(0, x1), max(35, y1)), scale=1, thickness=1, colorB=myColor,
                                   colorT=(255, 255, 255), colorR=myColor, offset=5)
                cv2.rectangle(frame, (x1, y1), (x2, y2), myColor, 3)
    return frame

# Navigation Panel
menu = {
    "Home": "🏠",
    "Dashboard": "📊",
    "About Us": "ℹ️"
}

st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(menu.keys()), format_func=lambda x: f"{menu[x]} {x}")

if selection == "Home":
    st.title("Driver Sentinel ")
    image_path = "data/Cruisin'.gif" 
    st.image(image_path)

    st.markdown("""
    ### Welcome to the Driver Sentinel App!

    This innovative application uses state-of-the-art AI and computer vision technologies to enhance road safety. By detecting drowsiness, seatbelt usage, and verifying driving licenses, the app aims to improve both driver and road safety in real-time.

    - **Drowsiness Detection**: This feature monitors the driver’s eye movement to determine if they are at risk of falling asleep while driving, triggering an alarm for immediate action if necessary.
    
    - **Seatbelt Detection**: Ensures that the driver and passengers are wearing seatbelts, a fundamental safety measure that reduces the risk of injury in case of an accident.
    
    - **License Detection**: Verifies the presence and authenticity of a driving license using advanced image recognition, ensuring that the driver is legally allowed to drive.

    Select from the dashboard to get started and experience real-time safety detection with cutting-edge AI!
""")
    
elif selection == "Dashboard":
    st.title("Safety Detection Dashboard")
    
    detection_option = st.selectbox("Choose the detection model to use", ["💳 License Detection","💺 Seatbelt Detection" ,"👁️ Drowsiness Detection" ])

    # Camera Feed
    camera = cv2.VideoCapture(0)

    # PlaceHolder to store frames of camera feed
    image_placeholder = st.empty()

    while True:
        ret, frame = camera.read()

        if not ret:
            st.error("Failed to capture video. Please check your webcam.")
            break

        # Converting frames captures in BGR from OpenCV to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if detection_option == "👁️ Drowsiness Detection":
            frame_rgb = drowsiness_detection(frame)
        elif detection_option == "💳 License Detection":
            frame_rgb = license_detection(frame)
        else:
            frame_rgb = seatbelt_detection(frame)

        # Frame Display
        image_placeholder.image(frame_rgb, channels="RGB")

    camera.release()

elif selection == "About Us":
    st.markdown("""
    ## Learn More on GitHub

    You can explore the full source code of the **Driver Sentinel** project on our [GitHub repository](https://github.com/YourUsername/DriverSentinel). The repository includes:

    - **Drowsiness Detection**: An AI model for detecting driver drowsiness through real-time video feed.
    - **Seatbelt Detection**: A machine learning model that ensures the driver is wearing a seatbelt while driving.
    - **License Detection**: A computer vision model that automatically recognizes driving licenses from images.
    
    The repository contains detailed documentation on how the models were built, trained, and integrated into the real-time safety system. You can also find instructions for setting up and running the project locally.

    Feel free to contribute, report issues, or fork the project for your own use. Together, we can make the roads safer!

    [Visit GitHub Repository](https://github.com/YourUsername/DriverSentinel)
""")

