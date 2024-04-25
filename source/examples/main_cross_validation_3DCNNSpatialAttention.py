import sys
sys.path.append('/home/axel/dev/fetus_GA_prediction/source')
import os
import pandas
import numpy as np
from utils import get_nii_data, get_nii_affine, save_image

import tensorflow as tf
from model.model import CNN_3D_spatialAttention
from model.generator_array import Generator3D, Generator3D_augment, Generator3D_augment_weight
from metrics.metrics import MAE, ME
from tensorflow.keras import optimizers
from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model
from tensorflow.keras import backend as K
# from keras.callbacks import ModelCheckpoint
from sklearn.metrics import mean_absolute_error

# from model.dataio import import_data_filename, write_nii

import time
import pickle
import gc

from sklearn.neighbors import KernelDensity


# from viewer import view3plane
# Just disables the warning and gpu info, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
# check the gpu connection
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())


# from tensorflow.keras import mixed_precision
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_global_policy(policy)

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
    
    X = np.zeros((n_subject, image_size[0], image_size[1], image_size[2]), dtype=np.float32)
    affines = []
    
    for n in range(n_subject):
        X[n] =  get_nii_data(img_dir + '/' + subject_names[n] + '.nii' )
        affines.append(get_nii_affine(img_dir + '/' + subject_names[n] + '.nii'))

    print('..data size:' + str(X.shape), flush=True)
       
    GA_predictions = np.zeros((n_subject, 1), dtype=np.float32)  
    
    nFold = 10
    foldSize = int(n_subject/nFold)
    
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
    	
    	# Compute weights for the training samples
    	kde = KernelDensity(kernel = 'gaussian', bandwidth = 1).fit(GA[train_id][:, np.newaxis])	
    	log_dens = kde.score_samples(GA[train_id][:, np.newaxis])    	
    	dens = np.exp(log_dens)
    	
    	normalized_dens = (dens - np.min(dens)) / (np.max(dens) - np.min(dens))

    	weight = 1 - normalized_dens
    	
    	eps = 1e-4
    	weight[weight < eps] = eps
    	
    	weight = weight / np.mean(weight)

          		
    	optimizer = optimizers.Adam(learning_rate=1e-3)
    	

    	model = CNN_3D_spatialAttention(image_size+[1])  
    	training_generator = Generator3D_augment_weight(X[train_id], GA[train_id], weight, batch_size = 16)
    	validation_generator = Generator3D(X[test_id], GA[test_id], batch_size = 6)

    	model.compile(optimizer=optimizer, loss = 'mse', metrics=[tf.keras.metrics.MeanAbsoluteError()])
    	
    	start_training_time = time.time()		
    	model.fit(x = training_generator, epochs = 120, verbose = 2, validation_data = validation_generator)	
    	end_training_time = time.time()
    	
    	print('training time: ' + str(end_training_time - start_training_time))
    	
    	model.save_weights('/home/axel/dev/fetus_GA_prediction/data/output/3DCNNSpatialAttention/exp1/model_' + str(iFold) + ".h5")
    	
    	start_execution_time = time.time()
    	for iSubject in range(len(test_id)):
    		GA_predictions[test_id[iSubject]] = model.predict(X[test_id[iSubject], :, :,:][np.newaxis, :, :, :, np.newaxis])	
    	end_execution_time = time.time()
    	print('executation time:' + str((end_execution_time - start_execution_time)/len(test_id)))
    	
    	print("mean absolute error per fold")
    	print(mean_absolute_error(GA[test_id], GA_predictions[test_id]))
    	
    	print("mean error per fold")
    	print(ME(GA[test_id], GA_predictions[test_id]))
    	

    	#Clean the memory at the end of the current fold 	
    	del model
    	del training_generator
    	del validation_generator

    	K.clear_session()
    	gc.collect()
    
    print("Mean absolute error whole cohort")
    print(mean_absolute_error(GA, GA_predictions))
    
    print("Mean error whole cohort")
    print(ME(GA, GA_predictions))
    	
    np.save('/home/axel/dev/fetus_GA_prediction/data/output/3DCNNSpatialAttention/exp1/GA.npy', GA)
    np.save('/home/axel/dev/fetus_GA_prediction/data/output/3DCNNSpatialAttention/exp1/GA_prediction.npy', GA_predictions)
#     
if __name__ == '__main__':
    main()
