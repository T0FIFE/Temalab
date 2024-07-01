from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import ImageDataGenerator 

model = load_model('models/model.h5')
test = ImageDataGenerator().flow_from_directory(directory = 'FER-2013/test',
                                                      target_size = (48,48),
                                                      color_mode = 'grayscale',
                                                      batch_size = 1,
                                                      shuffle = False,
                                                      class_mode='categorical')
test_labels = test.classes
#Create confusion matrix and normalizes it over predicted (columns)
res = model.predict(test,batch_size=1)
print(res)
cm = confusion_matrix(test_labels,res.argmax(axis=1))

disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels=['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise'] )
disp.plot()
plt.show()