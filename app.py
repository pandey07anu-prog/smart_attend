import cv2
import csv
import time
import os
import numpy as np
import threading
from datetime import datetime
from flask import Flask, render_template, Response, jsonify, send_file
import mediapipe as mp
import face_recognition

# ==============================
# Flask App
# ==============================
app = Flask(__name__)

# ==============================
# LOAD KNOWN FACES
# ==============================
known_encodings = []
known_names = []

folder = "known_faces"

if not os.path.exists(folder):
    os.makedirs(folder)

for file in os.listdir(folder):
    if file.endswith(".jpg") or file.endswith(".png"):
        path = os.path.join(folder, file)
        try:
            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)
            if len(encodings) > 0:
                known_encodings.append(encodings[0])
                known_names.append(os.path.splitext(file)[0])
        except Exception as e:
            print("Face Load Error:", e)

print("Loaded Students:", known_names)

# ==============================
# MEDIAPIPE SETUP
# ==============================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=3,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ==============================
# THREADED CAMERA STREAM READER
# ==============================
class CameraStream:
    def __init__(self):
        self.cap = None
        self.frame = None
        self.running = False
        self.lock = threading.Lock()
        self.initialize_camera()

    def initialize_camera(self):
        if self.cap:
            try:
                self.cap.release()
            except:
                pass
        
        for index in range(3):
            try:
                cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    ret, frame = cap.read()
                    if ret:
                        print(f"[OK] Camera Opened at Index {index}")
                        self.cap = cap
                        return True
                    else:
                        cap.release()
            except Exception as e:
                print("Camera Error:", e)
        self.cap = None
        return False

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()

    def _update_loop(self):
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                time.sleep(0.5)
                self.initialize_camera()
                continue
            
            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.05)
                continue

            frame = cv2.flip(frame, 1)
            with self.lock:
                self.frame = frame
            time.sleep(0.01)

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

camera = CameraStream()
camera.start()

# ==============================
# CSV FILE & GLOBAL STATE
# ==============================
attendance_file = "attendance.csv"
marked_students = set()
popup_name = None
eye_status = {}
latest_detections = []
detections_lock = threading.Lock()

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# EAR Function
def eye_aspect_ratio(landmarks, eye_indices, w, h):
    points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
    v1 = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
    v2 = np.linalg.norm(np.array(points[2]) - np.array(points[4]))
    h1 = np.linalg.norm(np.array(points[0]) - np.array(points[3]))
    return (v1 + v2) / (2.0 * h1)

# Attendance Function
def mark_attendance(name):
    if name in marked_students:
        return

    now = datetime.now()
    dt = now.strftime('%Y-%m-%d %H:%M:%S')
    file_exists = os.path.isfile(attendance_file)

    with open(attendance_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Name", "Time", "Status"])
        writer.writerow([name, dt, "Present"])

    marked_students.add(name)
    print(f"[OK] {name} Marked Present")

# ==============================
# ASYNC RECOGNITION WORKER THREAD
# ==============================
def recognition_worker():
    global popup_name
    global latest_detections

    while True:
        frame = camera.read()
        if frame is None:
            time.sleep(0.05)
            continue

        # Resize for ultra-fast recognition processing (0.35x)
        scale = 0.35
        small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small)
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

        inv_scale = 1.0 / scale
        scaled_locations = [
            (int(t * inv_scale), int(r * inv_scale), int(b * inv_scale), int(l * inv_scale))
            for (t, r, b, l) in face_locations
        ]

        results = None
        if len(scaled_locations) > 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

        curr_detections = []

        if results and results.multi_face_landmarks:
            for i, ((top, right, bottom, left), face_landmarks) in enumerate(zip(scaled_locations, results.multi_face_landmarks)):
                name = "Unknown"
                if i < len(face_encodings):
                    face_encoding = face_encodings[i]
                    matches = face_recognition.compare_faces(known_encodings, face_encoding)
                    face_distances = face_recognition.face_distance(known_encodings, face_encoding)

                    if len(face_distances) > 0:
                        best_match_index = np.argmin(face_distances)
                        if face_distances[best_match_index] < 0.45:
                            name = known_names[best_match_index]

                curr_detections.append({
                    "box": (top, right, bottom, left),
                    "name": name
                })

                # Blink Detection
                if name != "Unknown":
                    h, w, _ = frame.shape
                    left_ear = eye_aspect_ratio(face_landmarks.landmark, LEFT_EYE, w, h)
                    right_ear = eye_aspect_ratio(face_landmarks.landmark, RIGHT_EYE, w, h)
                    ear = (left_ear + right_ear) / 2.0

                    if name not in eye_status:
                        eye_status[name] = True

                    if ear < 0.20:
                        eye_status[name] = False
                    else:
                        if eye_status[name] == False:
                            eye_status[name] = True
                            if name not in marked_students:
                                mark_attendance(name)
                                popup_name = name

        with detections_lock:
            latest_detections = curr_detections

        time.sleep(0.06)

rec_thread = threading.Thread(target=recognition_worker, daemon=True)
rec_thread.start()

# ==============================
# HIGH FPS FAST STREAM GENERATOR
# ==============================
def gen_frames():
    jpeg_params = [int(cv2.IMWRITE_JPEG_QUALITY), 75]

    while True:
        frame = camera.read()
        if frame is None:
            time.sleep(0.02)
            continue

        # Overlay cached detections onto fresh frame
        with detections_lock:
            dets = list(latest_detections)

        for det in dets:
            (top, right, bottom, left) = det["box"]
            name = det["name"]
            color = (0, 255, 0) if name != "Unknown" else (0, 165, 255)

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        ret, buffer = cv2.imencode('.jpg', frame, jpeg_params)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            frame_bytes +
            b'\r\n'
        )
        time.sleep(0.015)

# ==============================
# ROUTES
# ==============================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(
        gen_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/check_popup')
def check_popup():
    global popup_name
    if popup_name:
        name = popup_name
        popup_name = None
        return name
    return "none"

@app.route('/api/attendance')
def get_attendance():
    records = []
    today_str = datetime.now().strftime('%Y-%m-%d')
    today_count = 0

    if os.path.exists(attendance_file):
        try:
            with open(attendance_file, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    records.append(row)
                    if row.get('Time', '').startswith(today_str):
                        today_count += 1
        except Exception as e:
            print("Error reading attendance CSV:", e)

    records_reversed = list(reversed(records))

    return jsonify({
        'status': 'success',
        'records': records_reversed,
        'total_count': len(records),
        'today_count': today_count,
        'enrolled_count': len(known_names)
    })

@app.route('/download_csv')
def download_csv():
    if os.path.exists(attendance_file):
        return send_file(attendance_file, as_attachment=True, download_name='attendance.csv')
    return "No attendance file found", 404

# ==============================
# RUN APP
# ==============================
if __name__ == "__main__":
    try:
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=False,
            threaded=True
        )
    finally:
        if camera and camera.cap:
            camera.cap.release()
        cv2.destroyAllWindows()