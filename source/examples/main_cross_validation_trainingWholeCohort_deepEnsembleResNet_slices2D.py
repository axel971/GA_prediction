import sys
sys.path.append('/home/axel/dev/fetus_GA_prediction/source')
import os
import pandas
import numpy as np
from utils import get_nii_data, get_nii_affine, save_image, percentOfForegroundInSlice

import tensorflow as tf
from model.model import ResNet34_2D
from model.generator_array import Generator_2D, Generator_2D_dataAugmentation, Generator_2D_weight, Generator_2D_dataAugmentation_weight
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

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV


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
   
   
  	## Extract the axial slices of the training  and validation images
    slice_axial_training = []
    GA_axial_training = []
    for iImage in range(len(X)):
    	for iSlice in range(X[0].shape[-1]):			
    		if(percentOfForegroundInSlice(delineation[iImage][ :, :, iSlice])> 0.5):
    			slice_axial_training.append(X[iImage][:, :, iSlice])
    			GA_axial_training.append(GA[iImage])
    			    	
    slice_axial_training = np.array(slice_axial_training)
    GA_axial_training = np.array(GA_axial_training)


    ## Extract the sagittal slices of the training  and validation images
    slice_sagittal_training = []
    GA_sagittal_training = []
    for iImage in range(len(X)):
    	for iSlice in range(X[0].transpose(1, 2, 0).shape[-1]):		
    		if(percentOfForegroundInSlice(delineation[iImage].transpose(1, 2, 0)[ :, :, iSlice])> 0.5):
    			slice_sagittal_training.append(X[iImage].transpose(1, 2, 0)[:, :, iSlice])
    			GA_sagittal_training.append(GA[iImage])
    			   	
    slice_sagittal_training = np.array(slice_sagittal_training)
    GA_sagittal_training = np.array(GA_sagittal_training)


	## Extract the coronal slices of the training  and validation images
    slice_coronal_training = []
    GA_coronal_training = []
    for iImage in range(len(X)):
    	for iSlice in range(X[0].transpose(0, 2, 1).shape[-1]):		
    		if(percentOfForegroundInSlice(delineation[iImage].transpose(0, 2, 1)[ :, :, iSlice])> 0.5):
    			slice_coronal_training.append(X[iImage].transpose(0, 2, 1)[:, :, iSlice])
    			GA_coronal_training.append(GA[iImage])
    			   	
    slice_coronal_training = np.array(slice_coronal_training)
    GA_coronal_training = np.array(GA_coronal_training)
    	   	
    ## Instantiate the models
    model_axial = ResNet34_2D([image_size[0], image_size[1]] +[1])
    model_sagittal = ResNet34_2D([image_size[1], image_size[2]] +[1])
    model_coronal = ResNet34_2D([image_size[0], image_size[2]] +[1])

    		   		   	
    optimizer = optimizers.Adam(learning_rate=1e-3)
    
    kde_axial = KernelDensity(kernel = 'gaussian', bandwidth = 1).fit(GA_axial_training[:, np.newaxis])
    kde_sagittal = KernelDensity(kernel = 'gaussian', bandwidth = 1).fit(GA_sagittal_training[:, np.newaxis])
    kde_coronal = KernelDensity(kernel = 'gaussian', bandwidth = 1).fit(GA_coronal_training[:, np.newaxis])
    	
    log_dens_axial = kde_axial.score_samples(GA_axial_training[:, np.newaxis])
    log_dens_sagittal = kde_sagittal.score_samples(GA_sagittal_training[:, np.newaxis])
    log_dens_coronal = kde_coronal.score_samples(GA_coronal_training[:, np.newaxis])
    	
    dens_axial = np.exp(log_dens_axial)
    dens_sagittal  = np.exp(log_dens_sagittal)
    dens_coronal = np.exp(log_dens_coronal)
    	
    normalized_dens_axial = (dens_axial - np.min(dens_axial)) / (np.max(dens_axial) - np.min(dens_axial))
    normalized_dens_sagittal  = (dens_sagittal - np.min(dens_sagittal)) / (np.max(dens_sagittal) - np.min(dens_sagittal))
    normalized_dens_coronal = (dens_coronal - np.min(dens_coronal)) / (np.max(dens_coronal) - np.min(dens_coronal))
    	
    weight_axial = 1 - normalized_dens_axial
    weight_sagittal = 1 - normalized_dens_sagittal
    weight_coronal = 1 - normalized_dens_coronal
    	
    eps = 1e-4
    weight_axial[weight_axial < eps] = eps
    weight_sagittal[weight_sagittal < eps] = eps
    weight_coronal[weight_coronal < eps] = eps
    	
    weight_axial = weight_axial / np.mean(weight_axial)
    weight_sagittal = weight_sagittal/ np.mean(weight_sagittal)
    weight_coronal = weight_coronal / np.mean(weight_coronal)

	
    training_axial_generator = Generator_2D_dataAugmentation_weight(slice_axial_training, GA_axial_training, weight_axial, batch_size = 32)
    training_sagittal_generator = Generator_2D_weight(slice_sagittal_training, GA_sagittal_training, weight_sagittal, batch_size = 32)
    training_coronal_generator = Generator_2D_dataAugmentation_weight(slice_coronal_training, GA_coronal_training, weight_coronal, batch_size = 32)
    	
    model_axial.compile(optimizer=optimizer, loss = 'mse', metrics=[tf.keras.metrics.MeanAbsoluteError()])
    model_sagittal.compile(optimizer=optimizer, loss = 'mse', metrics=[tf.keras.metrics.MeanAbsoluteError()])
    model_coronal.compile(optimizer=optimizer, loss = 'mse', metrics=[tf.keras.metrics.MeanAbsoluteError()])
    	
    start_training_time = time.time()
    print("#### training axial ####")	
    model_axial.fit(x = training_axial_generator, epochs = 120, verbose = 2)
    print("#### training sagittal ####")
    model_sagittal.fit(x = training_sagittal_generator, epochs = 120, verbose = 2)
    print("#### training coronal ####")
    model_coronal.fit(x = training_coronal_generator, epochs = 120, verbose = 2)
    end_training_time = time.time()	
    
    print('training time: ' + str(end_training_time - start_training_time))
    	
    model_axial.save_weights('/home/axel/dev/fetus_GA_prediction/data/output/deepEnsembleResNet_slices2D_wholeCohort/exp1/model_axial.h5')
    model_sagittal.save_weights('/home/axel/dev/fetus_GA_prediction/data/output/deepEnsembleResNet_slices2D_wholeCohort/exp1/model_sagittal.h5')
    model_coronal.save_weights('/home/axel/dev/fetus_GA_prediction/data/output/deepEnsembleResNet_slices2D_wholeCohort/exp1/model_coronal.h5')
    
   
if __name__ == '__main__':
    main()
