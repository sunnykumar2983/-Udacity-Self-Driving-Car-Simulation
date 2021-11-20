import pandas as pd 
import numpy as np
import cv2
print("cv2")
from sklearn.model_selection import train_test_split 
print("sklearn")
from keras.models import Sequential
print("tensorflow")
from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint

from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, BatchNormalization

import os
os.sys.path
import matplotlib.pyplot as plt
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
data_df = pd.read_csv("C:/Users/Asus/Desktop/Track2/driving_log.csv",names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])

images1 = []
X = data_df.iloc[:,0].values
for img in X:
    images1.append((cv2.resize(cv2.imread(img,-1)[80:140,:,:],(200,66))))
    

images1 = np.array(images1)

labels1 = data_df.iloc[:,3].values.reshape(-1,1)

images2 = []
labels2 = []

for i,img in enumerate(images1):
    images2.append(cv2.flip(img,1))
    labels2.append(-labels1[i])
    

images2 = np.array(images2)
labels2 = np.array(labels2)

images=np.append(images1,images2,axis=0)
labels=np.append(labels1,labels2,axis=0)

from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(images,labels,test_size=0.1)


model = Sequential()
model.add(Lambda(lambda x: x/127.5-1.0))
model.add(Conv2D(24, kernel_size = 5, activation='relu', strides = 2))
model.add(BatchNormalization())
model.add(Conv2D(36, kernel_size = 5, activation='relu', strides = 2))
model.add(BatchNormalization())
model.add(Conv2D(48, kernel_size = 5, activation='relu', strides = 2))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size = 3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size = 3, activation='elu'))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(optimizer=Adam(lr=0.00001), loss='mean_squared_error',metrics=['accuracy'])
model.fit(images, labels,validation_data=(X_test, y_test),shuffle=True,epochs=12)

model.save("C:/Users/Asus/Desktop/MOSAIC'20/sunny5T.h5")
for i in range(25):
    plt.imshow(images[i])
    print(labels[i])
    plt.show()

images.shape
