from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np
import cv2

# Load face detector & trained model
face_classifier = cv2.CascadeClassifier(r"face_classifier\\haarcascade_frontalface_default.xml")
model = load_model(r"Model\\GD.keras")

class_labels = ["man", "woman"]

# Start webcam
cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)

        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (128, 128), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype("float") / 255.0
            roi = np.expand_dims(roi, axis=-1)  
            roi = np.expand_dims(roi, axis=0)  

            # Sigmoid output to single probability
            prob = model.predict(roi, verbose=0)[0][0]
            label = class_labels[1] if prob >= 0.5 else class_labels[0]
            confidence = prob * 100 if prob >= 0.5 else (1 - prob) * 100

            pred_text = f"{label} ({confidence:.2f}%)"
            cv2.putText(frame, pred_text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 155), 2)
        else:
            cv2.putText(frame, "No Face", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 155), 3)

    cv2.imshow("Gender Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
