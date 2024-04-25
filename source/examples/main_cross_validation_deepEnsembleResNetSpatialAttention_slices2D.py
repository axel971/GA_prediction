import sys
sys.path.append('/home/axel/dev/fetus_GA_prediction/source')
import os
import pandas
import numpy as np
from utils import get_nii_data, get_nii_affine, save_image, percentOfForegroundInSlice

import tensorflow as tf
from model.model import ResNet34SpatialAttention_2D
from model.generator_array import Generator_2D, Generator_2D_dataAugmentation, Generator_2D_weight, Generator_2D_dataAugmentation_weight
from metrics.metrics import MAE, ME
from tensorflow.keras import optimizers
from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model
from tensorflow.keras import backend as K
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
   
    nFold = 10
    foldSize = int(n_subject/nFold)
    
    GA_predictions = np.zeros((n_subject, 1), dtype=np.float32)
    uncertainty = np.zeros((n_subject, 1), dtype=np.float32)  
    
    for iFold in range(nFold):
    	
    	# Split the data i training and the validation sets for the current fold
    	if iFold < nFold - 1 :
    		test_id = np.array(range(iFold * foldSize, (iFold + 1) * foldSize))
    		train_id = np.setdiff1d(np.array(range(n_subject)), test_id)
    		train_id = train_id[0:((nFold - 1) * foldSize)]
    	else:
    		test_id = np.array(range(iFold * foldSize, n_subject)) # Manage the case where the number of subject is not a multiple of the number of fold
    		train_id = np.setdiff1d(np.array(range(n_subject)), test_id)
 
    	
    	print('---------fold--'+str(iFold + 1)+'---------')
    	print('training set: ', train_id)
    	print('training set size: ', len(train_id))
    	print('testing set: ', test_id)
    	print('testing set size: ', len(test_id))
    	
    	## Extract the axial slices of the training  and validation images
    	slice_axial_training = []
    	GA_axial_training = []
    	for iImage in range(len(X[train_id])):
    		for iSlice in range(X[train_id[0]].shape[-1]):
    			
    			if(percentOfForegroundInSlice(delineation[train_id[iImage]][ :, :, iSlice])> 0.5):
    				slice_axial_training.append(X[train_id[iImage]][:, :, iSlice])
    				GA_axial_training.append(GA[train_id[iImage]])
    			
    	slice_axial_validation = []
    	GA_axial_validation = []
    	for iImage in range(len(X[test_id])):
    		for iSlice in range(X[test_id[0]].shape[-1]):
    			if(percentOfForegroundInSlice(delineation[test_id[iImage]][ :, :, iSlice]) > 0.5):
    				slice_axial_validation.append(X[test_id[iImage]][:, :, iSlice])
    				GA_axial_validation.append(GA[test_id[iImage]])
    	
    	slice_axial_training = np.array(slice_axial_training)
    	GA_axial_training = np.array(GA_axial_training)
    	slice_axial_validation = np.array(slice_axial_validation)
    	GA_axial_validation = np.array(GA_axial_validation)

    	## Extract the sagittal slices of the training  and validation images
    	slice_sagittal_training = []
    	GA_sagittal_training = []
    	for iImage in range(len(X[train_id])):
    		for iSlice in range(X[train_id[0]].transpose(1, 2, 0).shape[-1]):
    			
    			if(percentOfForegroundInSlice(delineation[train_id[iImage]].transpose(1, 2, 0)[ :, :, iSlice])> 0.5):
    				slice_sagittal_training.append(X[train_id[iImage]].transpose(1, 2, 0)[:, :, iSlice])
    				GA_sagittal_training.append(GA[train_id[iImage]])
    			
    	slice_sagittal_validation = []
    	GA_sagittal_validation = []
    	for iImage in range(len(X[test_id])):
    		for iSlice in range(X[test_id[0]].transpose(1, 2, 0).shape[-1]):
    			if(percentOfForegroundInSlice(delineation[test_id[iImage]].transpose(1, 2, 0)[ :, :, iSlice]) > 0.5):
    				slice_sagittal_validation.append(X[test_id[iImage]].transpose(1, 2, 0)[:, :, iSlice])
    				GA_sagittal_validation.append(GA[test_id[iImage]])
    	
    	slice_sagittal_training = np.array(slice_sagittal_training)
    	GA_sagittal_training = np.array(GA_sagittal_training)
    	slice_sagittal_validation = np.array(slice_sagittal_validation)
    	GA_sagittal_validation = np.array(GA_sagittal_validation)

		## Extract the coronal slices of the training  and validation images
    	slice_coronal_training = []
    	GA_coronal_training = []
    	for iImage in range(len(X[train_id])):
    		for iSlice in range(X[train_id[0]].transpose(0, 2, 1).shape[-1]):
    			
    			if(percentOfForegroundInSlice(delineation[train_id[iImage]].transpose(0, 2, 1)[ :, :, iSlice])> 0.5):
    				slice_coronal_training.append(X[train_id[iImage]].transpose(0, 2, 1)[:, :, iSlice])
    				GA_coronal_training.append(GA[train_id[iImage]])
    			
    	slice_coronal_validation = []
    	GA_coronal_validation = []
    	for iImage in range(len(X[test_id])):
    		for iSlice in range(X[test_id[0]].transpose(0, 2, 1).shape[-1]):
    			if(percentOfForegroundInSlice(delineation[test_id[iImage]].transpose(0, 2, 1)[ :, :, iSlice]) > 0.5):
    				slice_coronal_validation.append(X[test_id[iImage]].transpose(0, 2, 1)[:, :, iSlice])
    				GA_coronal_validation.append(GA[test_id[iImage]])
    	
    	slice_coronal_training = np.array(slice_coronal_training)
    	GA_coronal_training = np.array(GA_coronal_training)
    	slice_coronal_validation = np.array(slice_coronal_validation)
    	GA_coronal_validation = np.array(GA_coronal_validation)
    	   	
    	## Instantiate the models
    	model_axial = ResNet34SpatialAttention_2D([image_size[0], image_size[1]] +[1])
    	model_sagittal = ResNet34SpatialAttention_2D([image_size[1], image_size[2]] +[1])
    	model_coronal = ResNet34SpatialAttention_2D([image_size[0], image_size[2]] +[1])


    		
    		
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
    	training_sagittal_generator = Generator_2D_weight(slice_sagittal_training, GA_sagittal_training, weight_sagittal , batch_size = 32)
    	training_coronal_generator = Generator_2D_dataAugmentation_weight(slice_coronal_training, GA_coronal_training, weight_coronal, batch_size = 32)
    	
    	validation_axial_generator = Generator_2D(slice_axial_validation, GA_axial_validation, batch_size = 32)
    	validation_sagittal_generator = Generator_2D(slice_sagittal_validation, GA_sagittal_validation, batch_size = 32)
    	validation_coronal_generator = Generator_2D(slice_coronal_validation, GA_coronal_validation, batch_size = 32)
    	
    	model_axial.compile(optimizer=optimizer, loss = 'mse', metrics=[tf.keras.metrics.MeanAbsoluteError()])
    	model_sagittal.compile(optimizer=optimizer, loss = 'mse', metrics=[tf.keras.metrics.MeanAbsoluteError()])
    	model_coronal.compile(optimizer=optimizer, loss = 'mse', metrics=[tf.keras.metrics.MeanAbsoluteError()])
    	
#     	print(model_axial.summary())
    	
    	start_training_time = time.time()
    	print("#### training axial ####")	
    	model_axial.fit(x = training_axial_generator, epochs = 120, verbose = 2, validation_data = validation_axial_generator)
    	print("#### training sagittal ####")
    	model_sagittal.fit(x = training_sagittal_generator, epochs = 120, verbose = 2, validation_data = validation_sagittal_generator)
    	print("#### training coronal ####")
    	model_coronal.fit(x = training_coronal_generator, epochs = 120, verbose = 2, validation_data = validation_coronal_generator)
    	end_training_time = time.time()	
    	print('training time: ' + str(end_training_time - start_training_time))
    	
    	model_axial.save_weights('/home/axel/dev/fetus_GA_prediction/data/output/deepEnsembleResNetSpatialAttention_slices2D/exp1/model_axial_' + str(iFold) + ".h5")
    	model_sagittal.save_weights('/home/axel/dev/fetus_GA_prediction/data/output/deepEnsembleResNetSpatialAttention_slices2D/exp1/model_sagittal_' + str(iFold) + ".h5")
    	model_coronal.save_weights('/home/axel/dev/fetus_GA_prediction/data/output/deepEnsembleResNetSpatialAttention_slices2D/exp1/model_coronal_' + str(iFold) + ".h5")
    
    	start_execution_time = time.time()  	
    	for iSubject in range(len(test_id)):
    		
    		current_axial_slices = []
    		for iSlice in range(image_size[2]):
    			if(percentOfForegroundInSlice(delineation[test_id[iSubject]][ :, :, iSlice]) > 0.5):
    				current_axial_slices.append(X[test_id[iSubject]][ :, :, iSlice])
    		
    		current_sagittal_slices = []
    		for iSlice in range(image_size[0]):
    			if(percentOfForegroundInSlice(delineation[test_id[iSubject]].transpose(1, 2, 0)[ :, :, iSlice]) > 0.5):
    				current_sagittal_slices.append(X[test_id[iSubject]].transpose(1, 2, 0)[ :, :, iSlice])
    		
    		current_coronal_slices = []
    		for iSlice in range(image_size[1]):
    			if(percentOfForegroundInSlice(delineation[test_id[iSubject]].transpose(0, 2, 1)[ :, :, iSlice]) > 0.5):
    				current_coronal_slices.append(X[test_id[iSubject]].transpose(0, 2, 1)[ :, :, iSlice])


    		current_axial_slices = np.array(current_axial_slices)
    		current_sagittal_slices = np.array(current_sagittal_slices)
    		current_coronal_slices = np.array(current_coronal_slices)
    		
    		predictions_axial = model_axial.predict(np.expand_dims(current_axial_slices, -1))
    		predictions_sagittal = model_sagittal.predict(np.expand_dims(current_sagittal_slices, -1))
    		predictions_coronal = model_coronal.predict(np.expand_dims(current_coronal_slices, -1))
    		

    		predictions = np.concatenate((predictions_axial, predictions_sagittal, predictions_coronal))

    		
    		GA_predictions[test_id[iSubject]] = np.mean(predictions)
    		uncertainty[test_id[iSubject]] = np.var(predictions)
    		
    	print("mean absolute error per fold")
    	print(mean_absolute_error(GA[test_id], GA_predictions[test_id]))    	

    	print("mean error per fold")
    	print(ME(GA[test_id], GA_predictions[test_id]))   
    	   	
    	end_execution_time = time.time()
    	print('executation time:' + str((end_execution_time - start_execution_time)/len(test_id)))

    	del model_axial
    	del model_sagittal
    	del model_coronal

    	K.clear_session()
    	gc.collect()
    
    print("Mean absolute error whole cohort")
    print(mean_absolute_error(GA, GA_predictions))

    print("Mean error whole cohort")
    print(ME(GA, GA_predictions))
    	
    np.save('/home/axel/dev/fetus_GA_prediction/data/output/deepEnsembleResNetSpatialAttention_slices2D/exp1/GA.npy', GA)
    np.save('/home/axel/dev/fetus_GA_prediction/data/output/deepEnsembleResNetSpatialAttention_slices2D/exp1/GA_prediction.npy', GA_predictions)
    np.save('/home/axel/dev/fetus_GA_prediction/data/output/deepEnsembleResNetSpatialAttention_slices2D/exp1/uncertainty.npy', uncertainty)
   
if __name__ == '__main__':
    main()
