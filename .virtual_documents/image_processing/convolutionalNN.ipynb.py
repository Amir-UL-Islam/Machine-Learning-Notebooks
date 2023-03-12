import numpy as np
import pandas as pd
import tensorflow as tf


array = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
kernel = np.array([-1, 1]) # Changes

conv = np.array([0, 0, 0, 0, 0, 0 ,0, 0, 0])

for i in range(len(conv)):
    conv[i] = (kernel*array[i: i+2]).sum()
print(conv)


image = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]

kernel = np.array([[-1, 1],[-1, 1]])

conv2 = np.zeros([27, 27])


for i in range(27):
    for j in range(27):
        conv2[i, j] = (kernel*image[i: i+2, j: j+2]).sum()
        


from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.models import Sequential

model = Sequential()
model.add(Conv2D(10, kernel_size=3, activation='relu',padding='same', strides=2)) # total input = kernel_size*layer_unit
#                                                                     = (3*3)*10
#                                                                     = 90
model.add(Flatten())
model.add(Dense(3, activation='softmax'))


# Compilation
model.compile(optimizer='adam',
             loss='categorical_crossentropy',
             metrics=['accuracy'])


# # Fit the model on a training set
# model.fit(train_data, train_labels, 
#           validation_split=0.2, 
#           epochs=3, batch_size=10)



























