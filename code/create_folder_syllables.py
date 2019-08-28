# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 11:27:07 2019

@author: renan
"""

#this function create new spectrogram image and save in folder
#we need to pay attention that we create folder with syllabel name befor use this
#function and change the path (line 63)
def create_folder_syllables(syllabel,Data,TrueLabels):
    import numpy as np
#    import scipy.io.wavfile
#    import xlrd
#    from scipy.io import wavfile
#    from pathlib import Path
#    from AudioPreProcessing import AudioPreProcessing
#    from scipy import signal
#    from scipy.misc import imresize
    import os
    import matplotlib.pyplot as plt
    from PIL import Image


    
#    xl_workbook = xlrd.open_workbook(r"C:\Users\renan\Desktop\project\gold standard-Adults.xlsx")
#    xl_sheet = xl_workbook.sheet_by_index(0)
        
#    num_rows = xl_sheet.nrows   # Number of rows
    
    i=0
    for row_idx in range(1, len(Data)): # Iterate through rows
#        current_row = xl_sheet.row_values(row_idx)
        if TrueLabels[row_idx]==syllabel:
            
#        if current_row[6]==syllabel:
#            data_folder = Path("E:/USV Adult 2017/%s/%s/ch1" %(current_row[2], current_row[1]))
#            file = "%s.wav" %(current_row[3])
#            file_to_open = data_folder / file
#            sample_rate, samples = wavfile.read(file_to_open)
#            PreProcessedAudio=AudioPreProcessing(sample_rate, samples)
#            frequencies, times, spectrogram = signal.spectrogram(PreProcessedAudio, sample_rate)
#            list_times=list( times)
#            first_num = times[times>=current_row[4]][0]
#            first_ind = list_times.index(first_num)
#            sec_num = times[times>=current_row[5]][0]
#            sect_ind = list_times.index(sec_num)
#            x=spectrogram[:,first_ind:sect_ind]
#            
#            x-=scipy.mean(x)
#             
#             # zero padding to size 200
#            size=400
#            c,r=np.shape(x)
#            padr1=int((size-r)/2)
#            padr2=size-r-padr1
#            padc1=int((size-c)/2)
#            padc2=size-c-padc1
#            Padded=np.pad(x, [(padc1, padc2), (padr1, padr2)], mode='constant')
#            
#             # Decimation to a size of (32,32)
#            DecIm=imresize(Padded, (32,32), interp='bilinear', mode=None)
#            
#             # normalization
#            Images=DecIm/np.amax(DecIm)
            
            path = 'C:/Users/renan/Desktop/project/May/%s'%(syllabel)
            filename = syllabel+'%s.jpg' %(i)
            filename=syllabel+'%s.npy'%(i)
            filename = os.path.join(path, filename)
            np.save(filename,Data[i])
            
#            img = Image.fromarray(Data[row_idx])
#            img.save(filename)
            i+=1
            
            
    return (path)

