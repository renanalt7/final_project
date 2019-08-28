# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 12:01:50 2019

@author: renan
"""

import numpy as np
import keras
from keras import optimizers
from keras.models import Model 
from keras.layers import Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D
import ReadingAudio
from Aug_size import Aug_size
from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.applications import InceptionResNetV2
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import random
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D

# INITIALIZE MODEL

img_width, img_height = 32,32
#model = VGG16(weights = 'imagenet', include_top=False, input_shape = (img_width, img_height, 3))
model = VGG16(weights = 'imagenet', include_top=False, input_shape = (img_width, img_height, 3))
#model=InceptionResNetV2(include_top=False, weights='imagenet', input_tensor=None, input_shape=(img_width, img_height, 3), pooling=None, classes=1000)


# freeze all layers
for layer in model.layers:
    layer.trainable = False

# Adding custom Layers 
x = model.output
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(128, activation="relu")(x)
predictions = Dense(10, activation="softmax")(x)
#x=(Conv2D(32, kernel_size=(3, 3), activation='relu'))
#x = Conv2D(32, kernel_size=(3, 3),
#                 activation='relu',
#                 input_shape=(img_width, img_height, 3))
#x = Conv2D(32, kernel_size=(3, 3), activation='relu')(x)

#x = Flatten()(x)
#x = MaxPooling2D(pool_size=(2, 2))(x)
#x = Dropout(0.25)(x)
#x = Flatten()(x)
predictions = Dense(10, activation="softmax")(x)
#model.summary()

# creating the final model 
model_final = Model(input = model.input, output = predictions)

# compile the model 
rms = optimizers.RMSprop(lr=1e-4)
#adadelta = optimizers.Adadelta(lr=0.001, rho=0.5, epsilon=None, decay=0.0)

model_final.compile(loss = "categorical_crossentropy", optimizer = rms, metrics=["accuracy"])

model_final.summary()
batch_size = 32
num_classes = 10
epochs = 100

# input image dimensions
img_rows, img_cols = 32, 32

#reading data
(Data,TrueLabels,Precentages)=ReadingAudio.ReadingAudio()

#size DataAugentation
i=0
num_complex=0
num_upward=0
num_chevron=0
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
        y_train[i]= 3 # Chevron = 2
        i+=1
        num_chevron+=1
 
#do Augmentation for size (random_degree & horizontal_flip) 
num=int(np.amax([num_complex,num_upward,num_chevron]))
DataNewComplex = [None] * (num-num_complex)
DataNewUpward = [None] * (num-num_upward)
DataNewChevron = [None] * (num-num_chevron)
labels1 = [None] * (num-num_complex)
labels2 = [None] * (num-num_upward)
labels3 = [None] * (num-num_chevron)
if num_complex!=num:
    DataNewComplex=Aug_size((num-num_complex),'Complex',Data,TrueLabels)
    labels1=[1]*(num-num_complex)
if num_upward!=num:
    DataNewUpward=Aug_size((num-num_upward),'Upward',Data,TrueLabels)
    labels2=[2]*(num-num_upward)
if num_chevron!=num:
    DataNewChevron=Aug_size((num-num_chevron),'Chevron',Data,TrueLabels)
    labels3=[3]*(num-num_chevron)

x_train=[x for x in x_train if x is not None]
y_train=[x for x in y_train if x is not None]

x_train=x_train+DataNewComplex+DataNewUpward+DataNewChevron
y_train=y_train+labels1+labels2+labels3


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
x_train.shape

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

y_train_cat = keras.utils.to_categorical(y_train, num_classes)
y_test_cat = keras.utils.to_categorical(y_test, num_classes)

train_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

x_test = np.repeat(x_test[:, :, :], 3, axis=3)
x_train = np.repeat(x_train[:, :, :], 3, axis=3)

CallBack=keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1, mode='auto', baseline=None, restore_best_weights=True)

history=model_final.fit(x_train, y_train_cat, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test_cat)
 ,callbacks=[CallBack])

score = model_final.evaluate(x_test, y_test_cat, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.figure(0)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('Original Accuracy.jpg')
# summarize history for loss
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('Original Loss.jpg')

#Confution Matrix and Classification Report
Y_pred = model_final.predict(x_test)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(y_test, y_pred))

print('Classification Report')
target_names = ['Complex', 'Upward', 'Chevron']
print(classification_report(y_test, y_pred, target_names=target_names))

        
