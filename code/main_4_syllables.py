# -*- coding: utf-8 -*-
"""
Created on Wed May 15 11:35:52 2019

@author: renan
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 10:01:05 2019

@author: Shachar
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
from sklearn.metrics import classification_report, confusion_matrix
import random
from Aug_size import Aug_size
from keras import optimizers
from keras.models import Model 
#import generator
from keras.applications.vgg16 import VGG16

img_width, img_height = 32, 32
model = VGG16(weights = 'imagenet', include_top=False, input_shape = (img_width, img_height, 3))
#model =VGG19(include_top=False, weights='imagenet', input_shape = (img_width, img_height, 3), pooling=None, classes=1000)

# freeze all layers
for layer in model.layers[4:10]:
    layer.trainable = False

# Adding custom Layers 
x = model.output
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(10, activation="softmax")(x)

# creating the final model 
model_final = Model(input = model.input, output = predictions)

# compile the model 
rms = optimizers.RMSprop(lr=1e-5)
#adadelta = optimizers.Adadelta(lr=0.001, rho=0.5, epsilon=None, decay=0.0)

model_final.compile(loss = "categorical_crossentropy", optimizer = rms, metrics=["accuracy"])


batch_size = 32
num_classes = 10
epochs = 100

# input image dimensions
img_rows, img_cols = 32, 32

(Data,TrueLabels,Precentages)=ReadingAudio.ReadingAudio()

i=0
num_complex=0
num_upward=0
num_chevron=0
num_flat=0
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
    if TrueLabels[ind]== 'Flat':
        x_train[i]=Data[ind]
        y_train[i]= 3 # Flat = 3
        i+=1
        num_flat+=1
    if TrueLabels[ind]== 'Short':
        x_train[i]=Data[ind]
        y_train[i]= 4 # Chevron = 4
        i+=1
        num_chevron+=1
        
        
#do Augmentation for size (random_degree & horizontal_flip) 
num=int(np.amax([num_complex,num_upward,num_chevron,num_flat]))
DataNewComplex = [None] * (num-num_complex)
DataNewUpward = [None] * (num-num_upward)
DataNewChevron = [None] * (num-num_chevron)
DataNewFlat = [None] * (num-num_flat)

labels1 = [None] * (num-num_complex)
labels2 = [None] * (num-num_upward)
labels3 = [None] * (num-num_chevron)
labels4 = [None] * (num-num_flat)

if num_complex!=num:
    DataNewComplex=Aug_size((num-num_complex),'Complex',Data,TrueLabels)
    labels1=[1]*(num-num_complex)
if num_upward!=num:
    DataNewUpward=Aug_size((num-num_upward),'Upward',Data,TrueLabels)
    labels2=[2]*(num-num_upward)
if num_chevron!=num:
    DataNewChevron=Aug_size((num-num_chevron),'Chevron',Data,TrueLabels)
    labels3=[3]*(num-num_chevron)
if num_flat!=num:
    DataNewFlat=Aug_size((num-num_flat),'Flat',Data,TrueLabels)
    labels4=[3]*(num-num_flat)
    
x_train=[x for x in x_train if x is not None]
y_train=[x for x in y_train if x is not None]

x_train=x_train+DataNewComplex+DataNewUpward+DataNewChevron+DataNewFlat
y_train=y_train+labels1+labels2+labels3+labels4


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


#transfer learning:
y_train_cat = keras.utils.to_categorical(y_train, num_classes)
y_test_cat = keras.utils.to_categorical(y_test, num_classes)
                                             
x_test = np.repeat(x_test[:, :, :], 3, axis=3)
x_train = np.repeat(x_train[:, :, :], 3, axis=3)

CallBack=keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=25, verbose=1, mode='auto', baseline=None, restore_best_weights=True)

history=model_final.fit(x_train, y_train_cat, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test_cat)
 ,callbacks=[CallBack])

#our model:
y_train_cat = keras.utils.to_categorical(y_train, num_classes)
y_test_cat = keras.utils.to_categorical(y_test, num_classes)

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

CallBack=keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='auto', baseline=None, restore_best_weights=False)


model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adagrad(lr=0.012, epsilon=None, decay=0.0),
              metrics=['accuracy'])

history=model.fit(x_train, y_train_cat,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test_cat),
          callbacks=[CallBack])

score = model.evaluate(x_test, y_test_cat, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save('model4syllables.h5')


####################################################################
test=Data
test=np.asarray(test)
if K.image_data_format() == 'channels_first':
    test = test.reshape(1, 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    test = test.reshape(1, img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    
test = test.astype('float32')
test /= 255
pred = model.predict(test)

score_data=np.amax(pred,axis=1)
threshold=0.5
num=2
for x in range (0,len(score_data)):
    if score_data[x]<threshold:
        y_test_cat[x]=num+1            
score = model.evaluate(x_test, y_test_cat, verbose=0)


output=np.unravel_index(np.argmax(pred, axis=1), pred.shape)