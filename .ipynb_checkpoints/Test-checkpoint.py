from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np
import cv2


face_classifier = cv2.CascadeClassifier(r"face_classifier\\haarcascade_frontalface_default.xml")
model = load_model(r"Model\\GD.keras")
class_labels = ["male","female"]

cam = cv2.VideoCapture(0)

while True:
    ret,frame = cam.read()
    frame = cv2.flip(frame,1)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_classifier.detectMultiScale(gray,1.3,5)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,255),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48), interpolation=cv2.INTER_AREA)
        
        if np.sum([roi_gray])!=0:
            roi=roi_gray.astype('float')/255.0
            roi=img_to_array(roi)
            roi=np.expand_dims(roi,axis=0)
            preds = model.predict(roi)[0]
            label=class_labels[preds.argmax()]
            label_pos=(x,y)
            
            # Display prediction values
            pred_text = f"{label} : ({preds.max():.2f}%)"
            
            cv2.putText(frame,pred_text,label_pos,cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,155),2)
        else:
            cv2.putText(frame,'no face found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,155),3)
    cv2.imshow('emotionDetector',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()