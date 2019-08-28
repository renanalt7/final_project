# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 11:20:13 2019

@author: renan
"""
#this function create new data by augmentation: rotation 10 degree&horizontal flip
#input: number new syllables that we want co create, syllabel name, Data,Truelabels
#output: new data
def Aug_size(num_files_desired,syllabel,Data,TrueLabels):
    import random
    from scipy import ndarray
    import skimage as sk
    import os
    #from skimage import transform
    #from skimage import util
    import numpy as np
    from skimage import img_as_ubyte
    from create_folder_syllables import create_folder_syllables
    
    def random_rotation(image_array: ndarray):
        # pick a random degree of rotation between 25% on the left and 25% on the right
        random_degree = random.uniform(-10, 10)
        return sk.transform.rotate(image_array, random_degree)
    
    def horizontal_flip(image_array: ndarray):
        # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !
        return image_array[:, ::-1]
    
    def random_noise(image_array):
        return sk.util.random_noise(image_array, mode='gaussian', seed=None, clip=True, var=0.0001)
    
    
    # our folder path containing some images
    folder_path=create_folder_syllables(syllabel,Data,TrueLabels)
    # the number of file to generate
    #num_files_desired = 1000
    
    # loop on all files of the folder and build a list of files paths
    images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    num_generated_files = 0
    DataNew=[None] *num_files_desired
    while num_generated_files <num_files_desired:
        # random image from the folder
        image_path = random.choice(images)
        # read image as an two dimensional array of pixels
        image_to_transform =img_as_ubyte(np.load(image_path))
#        image_to_transform =img_as_ubyte(.load(image_path))
        
    
        
        #dictionary of the transformations functions we defined earlier
        available_transformations = {
            'rotate': random_rotation
#            'noise': random_noise
#            'horizontal_flip': horizontal_flip
        }
        
        # random num of transformations to apply
        num_transformations_to_apply = random.randint(1, len(available_transformations))
        
        num_transformations = 0
        transformed_image = None
        while num_transformations <= num_transformations_to_apply:
            # choose a random transformation to apply for a single image
            key = random.choice(list(available_transformations))
            transformed_image = available_transformations[key](image_to_transform)
            transformed_image=img_as_ubyte(transformed_image)
            num_transformations += 1
            
            
        # define a name for our new file
        DataNew[num_generated_files]=transformed_image
        num_generated_files+=1
        
    return (DataNew)
