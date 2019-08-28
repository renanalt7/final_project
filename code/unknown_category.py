# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 12:20:28 2019

@author: renan
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 10:52:04 2019

@author: renan
"""

from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import matplotlib.pyplot as plt
import ReadingAudio
import numpy as np
import random
from keras.models import load_model

batch_size = 32
num_classes = 10
epochs = 500

# input image dimensions
img_rows, img_cols = 32, 32

(Data,TrueLabels,Precentages)=ReadingAudio.ReadingAudio()

i=0
num_complex=0
num_upward=0
x_train = [None] * len(Data)
y_train = [None] * len(Data)

for ind in range(0, len(Data)): 
    if TrueLabels[ind]== 'Complex':
        x_train[i]=Data[ind]
        y_train[i]= 1 # Complex = 1
        i+=1
        num_complex+=1
    if TrueLabels[ind]== 'Upward':
        x_train[i]=Data[ind]
        y_train[i]= 2 # Upward = 2
        i+=1      
        num_upward+=1
    if TrueLabels[ind]== 'Downward':
        x_train[i]=Data[ind]
        y_train[i]= 3 # Upward = 2
        i+=1      
        num_upward+=1
    if TrueLabels[ind]== 'Flat':
        x_train[i]=Data[ind]
        y_train[i]= 3 # Upward = 2
        i+=1      
        num_upward+=1
    if TrueLabels[ind]== 'Short':
        x_train[i]=Data[ind]
        y_train[i]= 3 # Upward = 2
        i+=1      
        num_upward+=1

x_train=[x for x in x_train if x is not None]
y_train=[x for x in y_train if x is not None]

random.seed(4)
random.shuffle(x_train)
random.seed(4)
random.shuffle(y_train)

num_of_test_samples=int(np.floor(0.15*len(x_train)))
x_test=x_train[0:num_of_test_samples]
x_train=x_train[(num_of_test_samples+1):len(x_train)]

y_test=y_train[0:num_of_test_samples]
y_train=y_train[(num_of_test_samples+1):len(y_train)]

y_test = np.asarray(y_test)
y_train = np.asarray(y_train)
x_train=np.asarray(x_train)
x_test=np.asarray(x_test)

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

#y_train_cat = keras.utils.to_categorical(y_train, num_classes)
#y_test_cat = keras.utils.to_categorical(y_test, num_classes)

#from numpy.random import seed
#seed(1)
#from tensorflow import set_random_seed
#set_random_seed(2)
#
#model = Sequential()
#model.add(Conv2D(32, kernel_size=(3, 3),
#                 activation='relu',
#                 input_shape=input_shape))
#model.add(Conv2D(64, (3, 3), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
#model.add(Flatten())
#model.add(Dense(128, activation='relu'))
#model.add(Dropout(0.5))
#model.add(Dense(num_classes, activation='softmax'))
#
#CallBack=keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=0, mode='auto', baseline=None, restore_best_weights=True)
#
#model.compile(loss=keras.losses.categorical_crossentropy,
#              optimizer=keras.optimizers.Adagrad(lr=0.012, epsilon=None, decay=0.0),
#              metrics=['accuracy'])
#
#history=model.fit(x_train, y_train_cat,
#          batch_size=batch_size,
#          epochs=epochs,
#          verbose=1,
#          validation_data=(x_test, y_test_cat),
#          callbacks=[CallBack])
#
#model.save('model2syllables.h5')


############################################################################
#optimize thershold
############################################################################
model = load_model('model2syllables.h5')


probability=model.predict(x_test)
score_data=np.amax(probability,axis=1)
threshold=[0.5, 0.6, 0.7, 0.8, 0.9]
accuracy=[None] * len(threshold)
num=2

for th in range (0, len(threshold)):
    y_test_cat = keras.utils.to_categorical(y_test, num_classes)
    for x in range (0,len(score_data)):
        if score_data[x]<threshold[th]:
            y_test_cat[x]=num+1            
    score = model.evaluate(x_test, y_test_cat, verbose=0)
    accuracy[th]=score[1]

plt.scatter(threshold,accuracy)
plt.savefig('ROC.jpg')
