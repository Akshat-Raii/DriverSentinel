# Driver Sentinel 🚗

Welcome to the **Driver Sentinel**! This application aims to enhance road safety by incorporating features such as drowsiness detection, seatbelt monitoring, and driving license verification. With AI-powered systems, this app ensures a safer driving experience for everyone.

## Features ✨

- **Drowsiness Detection**: Tracks the driver's eye movements to detect signs of fatigue and triggers an alarm. 💤  
- **Seatbelt Detection**: Monitors if the driver and passengers are wearing seatbelts, ensuring compliance with safety protocols. 🎗️  
- **License Detection**: Identifies and verifies driving licenses using advanced object detection. 📜  
- **Streamlit Dashboard**: Provides a real-time interface for easy monitoring and alerts. 🌐  

## Hosted Version 🌐

A hosted version of the Driver Sentinel app is available at: [Driver Sentinel](https://driversentinel.streamlit.app/)

## How to Use 🚀

### 1. Clone the Repository  
   ```bash
   git clone https://github.com/yourusername/driver-sentinel.git
   cd driver-sentinel
   ```
### 2. Install Dependencies
  ```bash
  pip install -r requirements.txt
  ```
### 3. Running Individual Detection Features
You can run specific detection features by navigating to their respective folders and executing the Python scripts:
1. Drowsiness Detection
    ```bash
    cd DriverDrowsiness
    python detect_drowsiness.py
  
    ```
2. Seatbelt Detection
    ```bash
    cd SeatBelt_Detection
    python detectSeatBelt.py
    ```
3. License Detection
    ```bash
    cd DrivingLicense_Detection
    python detect_license.py
    ```
4. Running the Full Dashboard
   To view all features in a single dashboard:
     ```bash
     cd web_app
     streamlit run app.py
     ```

## Contributing 🤝

1. **Fork the repository.**
   
2. **Create a new branch:**
   
    ```bash
    git checkout -b feature-branch
    ```
3. **Make your changes.** ✏️
   
4. **Commit your changes:**
    ```bash
    git commit -am 'Add new feature'
    ```
5. **Push to the branch:**
   
    ```bash
    git push origin feature-branch
    ```
6. **Open a pull request.**: 📥

## Project Structure 🗂️

- **`DrowsinessDetection/`**: Contains code and resources for detecting drowsiness.
- **`SeatBelt_Detection/`**: Houses scripts for real-time seatbelt usage monitoring.
- **`License_Detection/`**: Includes tools for driving license verification.
- **`web_app`**: Main file to run the Streamlit dashboard.
- **`requirements.txt`**: Python dependencies for the project.

## Dependencies 🧩

The project relies on the following Python packages:

- **TensorFlow**
- **OpenCV**
- **Streamlit**
- **NumPy**
- **Matplotlib**
- **Seaborn**
- **YOLOv8**
- **Keras**
- **scikit-learn**
- **playsound**

**Stay safe and drive responsibly! 🚦**

