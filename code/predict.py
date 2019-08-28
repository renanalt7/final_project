# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 15:35:02 2019

@author: renan
"""


from __future__ import print_function
from keras import backend as K
import ReadingAudio
import numpy as np
from keras.models import load_model

# input image dimensions
img_rows, img_cols = 32, 32

model = load_model('model2syllables.h5')

(Data,TrueLabels,Precentages)=ReadingAudio.ReadingAudio()

test=Data
del test[0]
test=np.asarray(test)

if K.image_data_format() == 'channels_first':
    test = test.reshape(test.shape[0], 1, img_rows, img_cols)
else:
    test = test.reshape(test.shape[0], img_rows, img_cols, 1)
    
test = test.astype('float32')
test /= 255
pred = model.predict(test)

score_data=np.amax(pred,axis=1)
threshold=0.9
num=2
cat=np.argmax(pred, axis=1)
for x in range (0,len(score_data)):
    if score_data[x]<threshold:
        cat[x]=num+1            

output=np.unravel_index(cat, pred.shape)[1]