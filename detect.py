import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pyttsx3
import winsound
import time
import threading

# Load trained model
model = load_model("mask_model.h5")

def speak_alert():
    try:
        engine = pyttsx3.init()
        winsound.Beep(1000, 500)
        engine.say("Warning. No mask detected. Please wear your face mask.")
        engine.runAndWait()
    except Exception as e:
        print("Voice error:", e)


# Load Haar Cascade from OpenCV
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Check if Haar Cascade loaded properly
if face_cascade.empty():
    print("Error: Haar Cascade file not loaded!")
    exit()

# Start webcam
cap = cv2.VideoCapture(0)

last_alert_time = 0

while True:

    ret, frame = cap.read()

    if not ret:
        print("Camera not working!")
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=7,
        minSize=(100, 100)
    )

    for (x, y, w, h) in faces:
        padding = 20

        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)
        # Crop face
        face = frame[y:y+h, x:x+w]
        cv2.imwrite("captured_face.jpg", face)
        # Resize
        face = cv2.resize(face, (224, 224), interpolation=cv2.INTER_AREA)
        
        # Normalize
        face = face / 255.0

        # Add batch dimension
        face = np.expand_dims(face, axis=0)

        # Predict
        pred = model.predict(face, verbose=0)[0][0]
        print("Prediction:", pred)
        # Label
        if pred > 0.7:
            label = "NO MASK"
            color = (0, 0, 255)   # Red

            current_time = time.time()
            
            if current_time-last_alert_time>5:

                last_alert_time=current_time

                t = threading.Thread(target=speak_alert)
                t.start()
        else:
            label = "MASK"
            color = (0, 255, 0)   # Green

        # Draw rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            color,
            2
        )

        # Display text
        cv2.putText(
            frame,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )
    
        # Show exit message
        cv2.putText(
            frame,
            "Press Q to Exit",
            (10,30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255,255,255),
            2
        )
    # Show output
    cv2.imshow("Face Mask Detection", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()