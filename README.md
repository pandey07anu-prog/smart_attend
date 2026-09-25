# 🎓 Smart Attendance System

An AI-powered Smart Attendance System using **Face Recognition**, **MediaPipe Liveness Detection**, and **Computer Vision** to automate attendance management.

---

## 🚀 Overview

The Smart Attendance System automatically detects and recognizes faces through a webcam, verifies eye blink liveness, and marks attendance in real time.

This project features a modern glassmorphic Kiosk UI optimized for medium-sized screens (tablets, kiosks, laptops) and high-FPS multi-threaded camera processing.

---

## ✨ Features

* ✅ **Real-Time Face Recognition**: Matches webcam stream with registered student profiles.
* ✅ **AI Liveness Verification**: MediaPipe FaceMesh blink detection prevents photo/video spoofing.
* ✅ **High-FPS Camera Stream**: Multi-threaded async processing eliminates camera lag.
* ✅ **Medium-Screen Kiosk Interface**: Responsive 2-column UI designed for tablets, kiosks, and desktops.
* ✅ **Live Attendance Activity Logs**: Auto-updating table with search and filtering capabilities.
* ✅ **CSV Report Export**: Direct download button for attendance records (`attendance.csv`).

---

## 🛠️ Technologies Used

### 👨‍💻 Backend & AI
* Python 3.x
* Flask
* OpenCV
* MediaPipe (FaceMesh)
* Face Recognition (dlib/HOG)
* NumPy

### 🌐 Frontend
* HTML5 / CSS3 (Vanilla CSS, Glassmorphism design)
* FontAwesome 6
* Google Fonts (Outfit & Space Grotesk)
* JavaScript ES6+ (Web Audio API & Fetch API)

---

## 📂 Project Structure

```bash
Smart_Attend/
│
├── known_faces/            # Enrolled student face images (.jpg / .png)
├── model/                  # Haar cascades and models
├── templates/
│   └── index.html          # Modern Kiosk Web UI
├── .gitignore
├── app.py                  # Flask app with async camera pipeline & API endpoints
├── attendance.csv          # Real-time CSV attendance database
└── README.md
```

---

## ⚙️ Installation & Running

### Step 1: Clone Repository
```bash
git clone https://github.com/pandey07anu-prog/smart_attend.git
cd smart_attend
```

### Step 2: Create & Activate Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / Mac
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install flask opencv-python mediapipe face-recognition numpy
```

### Step 4: Run the Application
```bash
python app.py
```

Access the dashboard in your web browser at **`http://127.0.0.1:5000`**.

---

## 🔄 How It Works

1. Store student face images in `known_faces/` (e.g. `Student_Name.jpg`).
2. The webcam stream scans faces in real-time.
3. MediaPipe tracks eye aspect ratio (EAR) to detect natural blinking.
4. Upon blink verification, attendance is recorded in `attendance.csv` and broadcasted to the dashboard.
5. The UI automatically displays a success notification and updates attendance stats.

---

## 👩‍💻 Authors

* **Anupriya**
* **Avni**
