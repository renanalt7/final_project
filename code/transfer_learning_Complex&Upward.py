# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 11:06:59 2019

@author: renan
"""

import numpy as np
import keras
from keras_preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Model 
from keras.layers import Dropout, Flatten, Dense
import ReadingAudio
from Aug_size import Aug_size
from keras import backend as K
#import generator
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications import ResNet50
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import random
# INITIALIZE MODEL

img_width, img_height = 32, 32
model = VGG16(weights = 'imagenet', include_top=False, input_shape = (img_width, img_height, 3))
model =VGG19(include_top=False, weights='imagenet', input_shape = (img_width, img_height, 3), pooling=None, classes=1000)
model=ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape = (img_width, img_height, 3), pooling=None, classes=1000)

# freeze all layers
for layer in model.layers[:10]:
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
rms = optimizers.RMSprop(lr=1e-4)
#adadelta = optimizers.Adadelta(lr=0.001, rho=0.5, epsilon=None, decay=0.0)

model_final.compile(loss = "categorical_crossentropy", optimizer = rms, metrics=["accuracy"])

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
        
if num_complex<num_upward:
    DataNew=Aug_size((num_upward-num_complex),'Complex',Data,TrueLabels)    
if num_upward<num_complex:
    DataNew=Aug_size((num_complex-num_upward),'Upward',Data,TrueLabels)    
    
x_train=[x for x in x_train if x is not None]
y_train=[x for x in y_train if x is not None]

if num_complex<num_upward:
    labels=[1]*(num_upward-num_complex)
if num_upward<num_complex:
    labels=[2]*(num_complex-num_upward)

x_train=x_train+DataNew    
y_train=y_train+labels

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

y_train_cat = keras.utils.to_categorical(y_train, num_classes)
y_test_cat = keras.utils.to_categorical(y_test, num_classes)

train_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_dataframe(dataframe=x_train,
                                                    directory=None,
                                                    has_ext=False,
                                                    target_size = (img_height,
                                                                   img_width,3),
                                                    batch_size = batch_size, 
                                                    class_mode = 'categorical')
                                                    
x_test = np.repeat(x_test[:, :, :], 3, axis=3)
x_train = np.repeat(x_train[:, :, :], 3, axis=3)

CallBack=keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=35, verbose=1, mode='auto', baseline=None, restore_best_weights=True)

history=model_final.fit(x_train, y_train_cat, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test_cat)
 ,callbacks=[CallBack])
                                             


score = model_final.evaluate(x_test, y_test_cat, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# list all data in history
print(history.history.keys())
# summarize history for accuracy
#plt.subplot(211)
plt.figure(0)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('Original Accuracy.jpg')
# summarize history for loss
#plt.subplot(212)
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
target_names = ['Complex', 'Upward','Chevron']
print(classification_report(y_test, y_pred, target_names=target_names))