import numpy as np
import os
import pandas
import sys
sys.path.append('/home/axel/dev/fetus_GA_prediction/source')
from utils import get_nii_data


# Get the image quality label and patient name from the xlsx file
file_dir = "/home/axel/dev/fetus_GA_prediction/data/data_info_GA_corrected.xlsx"
data = pandas.read_excel(file_dir, dtype=str)
data_length = data.shape

study_names = np.array(data['study'])
fetus_IDs = np.array(data['fetus_ID'])
scan_numbers = np.array(data['scan'])
        
# Load the training images from the patient names 
img_dir = '/home/axel/dev/fetus_GA_prediction/data/preprocessing/MRI/boundingBox'
images = []
shapes = []

for study_name, fetus_ID, scan_number in zip(study_names, fetus_IDs, scan_numbers):
 	 images.append(get_nii_data(img_dir + '/' + str(study_name) + "_" + str(fetus_ID) +  "_scan_0" + str(scan_number) + '.nii.gz' )) 
 	 shapes.append(get_nii_data(img_dir + '/' + str(study_name) + "_" + str(fetus_ID) +  "_scan_0" + str(scan_number) + '.nii.gz' ).shape[2])       
 	 
print(np.amax(shapes))
print(np.amin(shapes))
print(np.mean(shapes))
#print(shapes)