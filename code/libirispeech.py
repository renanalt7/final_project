# -*- coding: utf-8 -*-
"""
Created on Thu May  2 17:08:28 2019

@author: renan
"""

import numpy as np                                                             
import soundfile as sf                                                       
import scipy
from scipy import signal
from scipy.misc import imresize
from skimage import img_as_ubyte                                        
import matplotlib.pyplot as plt
import os

p='C:/Users/renan/Desktop/project/LibiriSpeech'

files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(p):
    for file in f:
            files.append(os.path.join(r, file))
            
Data = [None] * len(files)
for pa in range(0,len(files)-1):
    #path = 'C:/Users/renan/Desktop/project/LibiriSpeech/84-121123-0000.flac'     
    path=files[pa]                                             
    samples, sample_rate = sf.read(path)  
    #PreProcessedAudio=AudioPreProcessing(sample_rate, samples)
    #eithour pre-processing!!!!!
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

    x=spectrogram
    
     # DC removal
    x-=scipy.mean(x)
     
     # zero padding to size 200
    size=3000
    c,r=np.shape(x)
    padr1=int((size-r)/2)
    padr2=size-r-padr1
    padc1=int((size-c)/2)
    padc2=size-c-padc1
    Padded=np.pad(x, [(padc1, padc2), (padr1, padr2)], mode='constant')
    
     # Decimation to a size of (32,32)
    DecIm=imresize(Padded, (32,32), interp='bilinear', mode=None)
    
     # normalization
    Image=DecIm/np.amax(DecIm)
    
    Data[pa] = img_as_ubyte(Image)

#plt.figure(1)
#plt.imshow(Image)
    
