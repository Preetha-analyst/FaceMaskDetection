from flask import Flask, render_template, Response
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pyttsx3
import winsound
import time
import threading

app = Flask(__name__)

# 1. Load the trained model and Haar Cascade classifier
model = load_model("mask_model.h5")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

camera = cv2.VideoCapture(0) # Initialize and turn on the webcam
last_alert_time = 0

# Separate function to trigger the voice alert asynchronously
def speak_alert():
    try:
        engine = pyttsx3.init()
        winsound.Beep(1000, 500)
        engine.say("Warning. No mask detected. Please wear your face mask.")
        engine.runAndWait()
    except Exception as e:
        print("Voice Error:", e)

# 2. Core function to continuously capture and process camera frames
def generate_frames():
    global last_alert_time
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Convert frame to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))

            for (x, y, w, h) in faces:
                # Crop and preprocess the face region
                face = frame[y1:y2, x1:x2]
                face = cv2.resize(face, (224, 224))
                face = face / 255.0
                face = np.expand_dims(face, axis=0)

                # Predict mask/no mask status
                pred = model.predict(face, verbose=0)[0][0]
                
                if pred > 0.7:
                    label = "NO MASK"
                    color = (0, 0, 255) # Red
                    
                    # Alert logic with a 5-second buffer using threading to prevent UI lag
                    current_time = time.time()
                    if current_time - last_alert_time > 5:
                        last_alert_time = current_time
                        t = threading.Thread(target=speak_alert)
                        t.start()
                else:
                    label = "MASK"
                    color = (0, 255, 0) # Green

                # Draw bounding box and text on the video frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # --- Critical Processing Step ---
            # Compress the OpenCV image array into a standard JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            # Using yield allows continuous frame delivery to the browser without breaking the loop
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# 3. Flask Route to render the main dashboard webpage (index.html)
@app.route('/')
def index():
    return render_template('index.html')

# 4. Flask Route to stream the real-time processed video data to the HTML page
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Start the local development server (debug=True enables hot-reloading on code changes)
    app.run(host='0.0.0.0', port=5000, debug=True)