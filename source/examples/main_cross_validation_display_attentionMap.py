import sys
sys.path.append('/home/axel/dev/fetus_GA_prediction/source')
import os
import pandas
import numpy as np
from utils import get_nii_data, get_nii_affine, save_image, percentOfForegroundInSlice

import tensorflow as tf
from model.model import ResNet34SpatialAttention_2D
from model.generator_array import Generator_2D
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
import cv2
# from viewer import view3plane
# Just disables the warning and gpu info, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
# check the gpu connection
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())



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
    	
    	   	
    	## Instantiate the models
    	model_axial =  ResNet34SpatialAttention_2D([image_size[0], image_size[1]] +[1])
    	model_sagittal =  ResNet34SpatialAttention_2D([image_size[1], image_size[2]] +[1])
    	model_coronal =  ResNet34SpatialAttention_2D([image_size[0], image_size[2]] +[1])

    	## Load the models
    	model_axial.load_weights('/home/axel/dev/fetus_GA_prediction/data/output/deepEnsembleResNetSpatialAttention_slices2D/exp2/model_axial_' + str(iFold) + ".h5")
    	model_sagittal.load_weights('/home/axel/dev/fetus_GA_prediction/data/output/deepEnsembleResNetSpatialAttention_slices2D/exp2/model_sagittal_' + str(iFold) + ".h5")
    	model_coronal.load_weights('/home/axel/dev/fetus_GA_prediction/data/output/deepEnsembleResNetSpatialAttention_slices2D/exp2/model_coronal_' + str(iFold) + ".h5")
    
    	start_execution_time = time.time()
    	
    	for iLayer in range(1,9):
    		
    		nLayer = str(iLayer)
    		get_layer_output_axial = tf.keras.backend.function([model_axial.layers[0].input], [model_axial.get_layer('Attention_map_' + nLayer).output])
    		get_layer_output_sagittal = tf.keras.backend.function([model_sagittal.layers[0].input], [model_sagittal.get_layer('Attention_map_' + nLayer).output])
    		get_layer_output_coronal = tf.keras.backend.function([model_coronal.layers[0].input], [model_coronal.get_layer('Attention_map_' + nLayer).output])
    	
    		
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

    		
    			#Create folers for current subject
    			pathAxialAttentionMaps = '/home/axel/dev/fetus_GA_prediction/data/output/deepEnsembleResNetSpatialAttention_slices2D/exp2/attentionMaps/axial/layer_' + nLayer + '/' + subject_names[test_id[iSubject]]
    			pathSagittalAttentionMaps = '/home/axel/dev/fetus_GA_prediction/data/output/deepEnsembleResNetSpatialAttention_slices2D/exp2/attentionMaps/sagittal/layer_' + nLayer + '/' + subject_names[test_id[iSubject]]
    			pathCoronalAttentionMaps = '/home/axel/dev/fetus_GA_prediction/data/output/deepEnsembleResNetSpatialAttention_slices2D/exp2/attentionMaps/coronal/layer_' + nLayer + '/' + subject_names[test_id[iSubject]]
    			
    			os.makedirs(pathAxialAttentionMaps, exist_ok = True)
    			os.makedirs(pathSagittalAttentionMaps, exist_ok = True)
    			os.makedirs(pathCoronalAttentionMaps, exist_ok = True)
    			  			    			
    			if(iLayer == 1):
    				pathAxialSlices = '/home/axel/dev/fetus_GA_prediction/data/output/deepEnsembleResNetSpatialAttention_slices2D/exp2/attentionMaps/axial/slices/' + subject_names[test_id[iSubject]]
    				pathSagittalSlices = '/home/axel/dev/fetus_GA_prediction/data/output/deepEnsembleResNetSpatialAttention_slices2D/exp2/attentionMaps/sagittal/slices/' + subject_names[test_id[iSubject]]
    				pathCoronalSlices = '/home/axel/dev/fetus_GA_prediction/data/output/deepEnsembleResNetSpatialAttention_slices2D/exp2/attentionMaps/coronal/slices/' + subject_names[test_id[iSubject]]
    			
    				os.makedirs(pathAxialSlices, exist_ok = True)
    				os.makedirs(pathSagittalSlices, exist_ok = True)
    				os.makedirs(pathCoronalSlices, exist_ok = True)
    			
    			
#     			pathAxialPrediction = '/home/axel/dev/fetus_GA_prediction/data/output/deepEnsembleResNetSpatialAttention_slices2D/exp1/attentionMaps/axial/prediction/' + subject_names[test_id[iSubject]]		
#     			pathSagittalPrediction = '/home/axel/dev/fetus_GA_prediction/data/output/deepEnsembleResNetSpatialAttention_slices2D/exp1/attentionMaps/sagittal/prediction/' + subject_names[test_id[iSubject]]    			
#     			pathCoronalPrediction = '/home/axel/dev/fetus_GA_prediction/data/output/deepEnsembleResNetSpatialAttention_slices2D/exp1/attentionMaps/coronal/prediction/' + subject_names[test_id[iSubject]]
#     		
#     			os.makedirs(pathAxialPrediction, exist_ok = True)			
#     			os.makedirs(pathSagittalPrediction, exist_ok = True)
#     			os.makedirs(pathCoronalPrediction, exist_ok = True)
#     		
#     			np.savetxt(pathAxialPrediction + '/' + subject_names[test_id[iSubject]] + '_predictedGA.txt', predictions_axial)
#     			np.savetxt(pathSagittalPrediction + '/' + subject_names[test_id[iSubject]] + '_predictedGA.txt', predictions_sagittal)
#     			np.savetxt(pathCoronalPrediction + '/' + subject_names[test_id[iSubject]] + '_predictedGA.txt', predictions_coronal)
#     		
#     			np.savetxt(pathAxialPrediction + '/' + subject_names[test_id[iSubject]] + '_GA.txt', [GA[test_id[iSubject]]])
#     			np.savetxt(pathSagittalPrediction + '/' + subject_names[test_id[iSubject]] + '_GA.txt', [GA[test_id[iSubject]]])
#     			np.savetxt(pathCoronalPrediction + '/' + subject_names[test_id[iSubject]] + '_GA.txt', [GA[test_id[iSubject]]])
    	
    				
    			for iSlice in range(len(current_axial_slices)):
    		
    				attentionMap1 = get_layer_output_axial([np.expand_dims(current_axial_slices[iSlice], 0)])[0]		
    				save_image(cv2.resize(np.array(attentionMap1[0,:,:,0]), (128, 128), interpolation = cv2.INTER_CUBIC), affines[test_id[iSubject]], pathAxialAttentionMaps + '/attentionMap_' + str(iSlice) + '.nii')
    				if(iLayer == 1):
    					save_image(current_axial_slices[iSlice], affines[test_id[iSubject]], pathAxialSlices + '/slice_' + str(iSlice) + '.nii')
    		
    			for iSlice in range(len(current_sagittal_slices)):
    		
    				attentionMap1 = get_layer_output_sagittal([np.expand_dims(current_sagittal_slices[iSlice], 0)])[0]	
    				save_image(cv2.resize(np.array(attentionMap1[0,:,:,0]), (128,128)), affines[test_id[iSubject]], pathSagittalAttentionMaps + '/attentionMap_' + str(iSlice) + '.nii')
    				if(iLayer == 1):
    					save_image(current_sagittal_slices[iSlice], affines[test_id[iSubject]], pathSagittalSlices + '/slice_' + str(iSlice) + '.nii')

    			for iSlice in range(len(current_coronal_slices)):
    		
    				attentionMap1 = get_layer_output_coronal([np.expand_dims(current_coronal_slices[iSlice], 0)])[0]
    				save_image(cv2.resize(attentionMap1[0,:,:,0], (128, 128), interpolation = cv2.INTER_CUBIC), affines[test_id[iSubject]], pathCoronalAttentionMaps + '/attentionMap_' + str(iSlice) + '.nii')
    				if(iLayer == 1):
    					save_image(current_coronal_slices[iSlice], affines[test_id[iSubject]], pathCoronalSlices + '/slice_' + str(iSlice) + '.nii')

    		
    	end_execution_time = time.time()
    	print('executation time:' + str((end_execution_time - start_execution_time)/len(test_id)))

    	del model_axial
    	del model_sagittal
    	del model_coronal

	    	
    	K.clear_session()
    	gc.collect()

if __name__ == '__main__':
    main()
