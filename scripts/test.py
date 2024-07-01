import cv2 as cv
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

classifier = cv.CascadeClassifier('haarcascade_frontalface.xml')

model = load_model('emotion_model7.keras')

labels = ['Angry','Fear','Happy','Neutral', 'Sad', 'Surprise']

capture = cv.VideoCapture(0)
mode1 = load_model('models/emotion_model5.keras')
labels.insert(0,labels[0])

while True:
    _, frame = capture.read()
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    face_locations = classifier.detectMultiScale(frame_gray)

    for (x,y,w,h) in face_locations:
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        face = frame_gray[y:y+h,x:x+w]
        face = cv.resize(face,(48,48),interpolation=cv.INTER_AREA)



        if np.sum([face])!=0:
            roi = face.astype('float')/255.0
            roi = image.img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

            prediction = model.predict(roi)[0]
            label=labels[prediction.argmax()]
            label_position = (x,y)
            cv.putText(frame,label,label_position,cv.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        else:
            cv.putText(frame,'No Faces',(30,80),cv.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    cv.imshow('video',frame)

    if cv.waitKey(20) & 0xFF==ord('d'):
        break

capture.release()
cv.destroyAllWindows()