import sys
sys.path.append('/home/axel/dev/fetus_GA_prediction/source')
import os
import pandas
import numpy as np
from utils import get_nii_data, get_nii_affine, save_image
import cv2
import imgaug.augmenters as iaa
import torchio as tio 

def main():

    n_subject = 451
    subject_list = np.arange(n_subject)
    image_size = [128, 128, 128]
    
    # Get path data
    file_dir = "/home/axel/dev/fetus_GA_prediction/data/subjectList.xlsx"
    data = pandas.read_excel(file_dir)
    data_length = data.shape
    subject_names = np.array(data['Subject_name'])
    GA = np.array(data['GA'])
    img_dir = '/home/axel/dev/fetus_GA_prediction/data/preprocessing/MRI/histogramMatching'
    
    X = np.zeros((n_subject, image_size[0], image_size[1], image_size[2]), dtype=np.float32)
    affines = []
    
    for n in range(2):
        X[n] =  get_nii_data(img_dir + '/' + subject_names[n] + '.nii' )
        affines.append(get_nii_affine(img_dir + '/' + subject_names[n] + '.nii'))

    image = X[0]
    
    data_augmentation = tio.RandomFlip(axes = 0, flip_probability = 1)
    augmented_image = data_augmentation(np.expand_dims(image, 0))
    	
    	
    save_image(image, affines[0], './test_image.nii')
    
    save_image(augmented_image[0], affines[0], './test_aug_image.nii')    
    
#     cv2.imwrite('./test.jpg', slice)
   
   
if __name__ == '__main__':
    main()
