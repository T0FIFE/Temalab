import cv2 as cv
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

classifier = cv.CascadeClassifier('haarcascade_frontalface.xml')
model = load_model('models/model.h5')

labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

img = cv.imread('emotions.png')
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

face_locations = classifier.detectMultiScale(img_gray)

for (x,y,w,h) in face_locations:
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
    face = img_gray[y:y+h,x:x+w]
    face = cv.resize(face,(48,48),interpolation=cv.INTER_AREA)


    if np.sum([face])!=0:
        roi = face.astype('float')/255.0
        roi = image.img_to_array(roi)
        roi = np.expand_dims(roi,axis=0)

        prediction = model.predict(roi)[0]
        print(prediction)
        label=labels[prediction.argmax()]
        label_position = (x,y)
        cv.putText(img,label,label_position,cv.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    else:
        cv.putText(img,'No Faces',(30,80),cv.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

while True:

    cv.imshow('color image',img)

    if cv.waitKey(20) & 0xFF==ord('d'):
        break

capture.release()
cv.destroyAllWindows()