import sys
sys.path.append('/home/axel/dev/fetus_GA_prediction/source')
import os
import pandas
import numpy as np
from utils import get_nii_data, get_nii_affine, save_image, percentOfForegroundInSlice

import tensorflow as tf
from model.model import ResNet34_2D
from model.generator_array import Generator_2D
from metrics.metrics import MAE, ME
from tensorflow.keras import optimizers
from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model
from tensorflow.keras import backend as K
# from tensorflow.keras import mixed_precision

# from keras.callbacks import ModelCheckpoint

# from model.dataio import import_data_filename, write_nii
from sklearn.metrics import mean_absolute_error
import time
import pickle
import gc

# from viewer import view3plane
# Just disables the warning and gpu info, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
# check the gpu connection
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_global_policy(policy)


def main():

    n_subject = 217
    subject_list = np.arange(n_subject)
    image_size = [128, 128, 128]
    
    # Get path data
    file_dir = "/home/axel/dev/fetus_GA_prediction/data/subjectList_CHD.xlsx"
    data = pandas.read_excel(file_dir)
    data_length = data.shape
    subject_names = np.array(data['Subject_name'])
    GA = np.array(data['GA'])
    img_dir = '/home/axel/dev/fetus_GA_prediction/data/preprocessing/MRI/healthy_fetuses/histogramMatching'
    delineation_dir = "/home/axel/dev/fetus_GA_prediction/data/preprocessing/delineation/healthy_fetuses/resampled_boundingBox"
    
    X = np.zeros((n_subject, image_size[0], image_size[1], image_size[2]), dtype=np.float32)
    affines = []
    delineation = np.zeros((n_subject, image_size[0], image_size[1], image_size[2]), dtype=np.float32)
    
    for n in range(n_subject):
        X[n] =  get_nii_data(img_dir + '/' + subject_names[n] + '.nii' )
        affines.append(get_nii_affine(img_dir + '/' + subject_names[n] + '.nii'))
        delineation[n] =  get_nii_data(delineation_dir + '/' + subject_names[n] + '.nii.gz' )

    print('..data size:' + str(X.shape), flush=True)
   
    
    GA_predictions = np.zeros((n_subject, 1), dtype=np.float32)
    uncertainty = np.zeros((n_subject, 1), dtype=np.float32)  
    	   	
    ## Instantiate the models
    model_axial = ResNet34_2D([image_size[0], image_size[1]] +[1])
    model_sagittal = ResNet34_2D([image_size[1], image_size[2]] +[1])
    model_coronal = ResNet34_2D([image_size[0], image_size[2]] +[1])
    	
    	
    model_axial.load_weights('/home/axel/dev/fetus_GA_prediction/data/output/deepEnsembleResNet_slices2D_wholeCohort/exp5/model_axial.h5')
    model_sagittal.load_weights('/home/axel/dev/fetus_GA_prediction/data/output/deepEnsembleResNet_slices2D_wholeCohort/exp5/model_sagittal.h5')
    model_coronal.load_weights('/home/axel/dev/fetus_GA_prediction/data/output/deepEnsembleResNet_slices2D_wholeCohort/exp5/model_coronal.h5')
    
 
    start_execution_time = time.time()  	
    for iSubject in range(n_subject):
    		
    	current_axial_slices = []
    	for iSlice in range(image_size[2]):
    		if(percentOfForegroundInSlice(delineation[iSubject][ :, :, iSlice]) > 0.5):
    			current_axial_slices.append(X[iSubject][ :, :, iSlice])
    		
    	current_sagittal_slices = []
    	for iSlice in range(image_size[0]):
    		if(percentOfForegroundInSlice(delineation[iSubject].transpose(1, 2, 0)[ :, :, iSlice]) > 0.5):
    			current_sagittal_slices.append(X[iSubject].transpose(1, 2, 0)[ :, :, iSlice])
    		
    	current_coronal_slices = []
    	for iSlice in range(image_size[1]):
    		if(percentOfForegroundInSlice(delineation[iSubject].transpose(0, 2, 1)[ :, :, iSlice]) > 0.5):
    			current_coronal_slices.append(X[iSubject].transpose(0, 2, 1)[ :, :, iSlice])


    	current_axial_slices = np.array(current_axial_slices)
    	current_sagittal_slices = np.array(current_sagittal_slices)
    	current_coronal_slices = np.array(current_coronal_slices)
    		
    	predictions_axial = model_axial.predict(np.expand_dims(current_axial_slices, -1))
    	predictions_sagittal = model_sagittal.predict(np.expand_dims(current_sagittal_slices, -1))
    	predictions_coronal = model_coronal.predict(np.expand_dims(current_coronal_slices, -1))
    		

    	predictions = np.concatenate((predictions_axial, predictions_sagittal, predictions_coronal))

    		
    	GA_predictions[iSubject] = np.mean(predictions)
    	uncertainty[iSubject] = np.var(predictions)
    		 	   	
    end_execution_time = time.time()
    print('executation time:' + str((end_execution_time - start_execution_time)/n_subject))

    
    print("Mean absolute error whole cohort")
    print(mean_absolute_error(GA, GA_predictions))

    print("Mean error whole cohort")
    print(ME(GA, GA_predictions))
    	
    np.save('/home/axel/dev/fetus_GA_prediction/data/output/deepEnsembleResNet_slices2D_wholeCohort/exp5/GA.npy', GA)
    np.save('/home/axel/dev/fetus_GA_prediction/data/output/deepEnsembleResNet_slices2D_wholeCohort/exp5/GA_prediction.npy', GA_predictions)
    np.save('/home/axel/dev/fetus_GA_prediction/data/output/deepEnsembleResNet_slices2D_wholeCohort/exp5/uncertainty.npy', uncertainty)
    
if __name__ == '__main__':
    main()
