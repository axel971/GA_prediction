
from tensorflow.keras.layers import Conv3D, Conv1D, MaxPool3D, concatenate, Input, Dropout, PReLU,  Conv3DTranspose, BatchNormalization, SpatialDropout3D, SpatialDropout2D, Flatten, Dense, Conv2D, MaxPooling2D, MaxPooling3D, Add, GlobalAveragePooling2D, GlobalAveragePooling3D, Lambda, GlobalMaxPooling2D, Activation, Multiply, Reshape, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import Zeros
# from keras_contrib.layers import CRF
from model.ConcreteDropout.spatialConcreteDropout import SpatialConcreteDropout
from model.ConcreteDropout.concreteDropout import ConcreteDropout
import tensorflow.keras as K
import tensorflow as tf
from tensorflow_addons.activations import mish


def CNN_3D(input_shape):

	inputs = Input(shape = input_shape)


	x = Conv3D(16, kernel_size = (3, 3, 3),  padding = 'same')(inputs) #128
	x = BatchNormalization()(x)
	x = mish(x)	
	x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)

	x = Conv3D(32, kernel_size = (3, 3, 3), padding = 'same')(x) #64
	x = BatchNormalization()(x)
	x = mish(x)
	x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)


	x = Conv3D(64, kernel_size = (3, 3, 3), padding = 'same')(x) #32
	x = BatchNormalization()(x)
	x = mish(x)
	x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)


	x = Conv3D(128, kernel_size = (3, 3, 3), padding = 'same')(x) #16
	x = BatchNormalization()(x)
	x = mish(x)
	x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding ='same')(x)

	x = Conv3D(256, kernel_size = (3, 3, 3), padding = 'same')(x) #8
# 	x = SpatialDropout3D(0.2)(x)
	x = BatchNormalization()(x)
	x = mish(x)
	x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding ='same')(x)
	
	x = Conv3D(512, kernel_size = (3, 3, 3), padding = 'same')(x) #4
	x = BatchNormalization()(x)
	x = mish(x)
	
	x = GlobalAveragePooling3D()(x)
    
	x = Flatten()(x)
	x = Dense(1, activation = 'linear')(x)

	model = Model(inputs = inputs, outputs = x, name = 'CNN_3D')
	
	return model

	
def CNN_3D_spatialAttention(input_shape):

	inputs = Input(shape = input_shape)

	x = Conv3D(16, kernel_size = (3, 3, 3),  padding = 'same')(inputs) #128
	x = BatchNormalization()(x)
	x = spatialAttention3DModule(x, "0", (7, 7, 7))
	x = mish(x)	
	x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)

	x = Conv3D(32, kernel_size = (3, 3, 3), padding = 'same')(x) #64
	x = BatchNormalization()(x)
	x = spatialAttention3DModule(x, "1", (7, 7, 7))
	x = mish(x)
	x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)

	x = Conv3D(64, kernel_size = (3, 3, 3), padding = 'same')(x) #32
	x = BatchNormalization()(x)
	x = spatialAttention3DModule(x, "2", (7, 7, 7))
	x = mish(x)
	x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)

	x = Conv3D(128, kernel_size = (3, 3, 3), padding = 'same')(x) #16
	x = BatchNormalization()(x)
	x = spatialAttention3DModule(x, "3", (5, 5, 5))
	x = mish(x)
	x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding ='same')(x)

	x = Conv3D(256, kernel_size = (3, 3, 3), padding = 'same')(x) #8
	x = BatchNormalization()(x)
	x = spatialAttention3DModule(x, "4", (3, 3, 3))
	x = mish(x)
	x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding ='same')(x)
	
	x = Conv3D(512, kernel_size = (3, 3, 3), padding = 'same')(x) #4
	x = BatchNormalization()(x)
	x = mish(x)
	
	x = GlobalAveragePooling3D()(x)
    
	x = Flatten()(x)
	x = Dense(1, activation = 'linear')(x)

	model = Model(inputs = inputs, outputs = x, name = 'CNN_3D')
	
	return model
	
    
##### ResNet 50

def ResNet50_2D(input_shape):

    input_layer = Input(shape = input_shape)
    
    x = Conv2D(filters = 64, kernel_size=(7, 7),  padding='same', strides=2)(input_layer)
    x = MaxPooling2D(pool_size = (3,3), strides = (2,2), padding='same')(x)
    
    for iBlock in range(3):
    	if(iBlock != 0):
    		x = Res50_block_2D(x, stride_first_layer = 1, filter_size_1 = 64, kernel_size_1 = (1, 1), filter_size_2 = 64, kernel_size_2 = (3, 3), filter_size_3 = 256, kernel_size_3 = (1, 1))
    	else:
    		x = Res50_BottleNeckBlock_2D(x, stride_first_layer = 1, filter_size_1 = 64, kernel_size_1 = (1, 1), filter_size_2 = 64, kernel_size_2 = (3, 3), filter_size_3 = 256, kernel_size_3 = (1, 1))
    
    for iBlock in range(8):
    	if(iBlock != 0):
    		x = Res50_block_2D(x, stride_first_layer = 1, filter_size_1 = 128, kernel_size_1 = (1, 1), filter_size_2 = 128, kernel_size_2 = (3, 3), filter_size_3 = 512, kernel_size_3 = (1, 1))
    	else:
    		x = Res50_BottleNeckBlock_2D(x, stride_first_layer = 2, filter_size_1 = 128, kernel_size_1 = (1, 1), filter_size_2 = 128, kernel_size_2 = (3, 3), filter_size_3 = 512, kernel_size_3 = (1, 1))
    
    for iBlock in range(36):
    	if(iBlock != 0):
    		x = Res50_block_2D(x, stride_first_layer = 1, filter_size_1 = 256, kernel_size_1 = (1, 1), filter_size_2 = 256, kernel_size_2 = (3, 3), filter_size_3 = 1024, kernel_size_3 = (1, 1))
    	else:
    		x = Res50_BottleNeckBlock_2D(x, stride_first_layer = 2, filter_size_1 = 256, kernel_size_1 = (1, 1), filter_size_2 = 256, kernel_size_2 = (3, 3), filter_size_3 = 1024, kernel_size_3 = (1, 1))
     		
    for iBlock in range(3):
    	if(iBlock != 0):		
    		x = Res50_block_2D(x, stride_first_layer = 1, filter_size_1 = 512, kernel_size_1 = (1, 1), filter_size_2 = 512, kernel_size_2 = (3, 3), filter_size_3 = 2048, kernel_size_3 = (1, 1))
    	else:
    		x = Res50_BottleNeckBlock_2D(x, stride_first_layer = 2, filter_size_1 = 512, kernel_size_1 = (1, 1), filter_size_2 = 512, kernel_size_2 = (3, 3), filter_size_3 = 2048, kernel_size_3 = (1, 1))
    
     
    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    x = Dense(1, activation = 'linear')(x)
	
    model = Model(inputs = input_layer, outputs = x)
    
    return model

    
def Res50_BottleNeckBlock_2D(x, stride_first_layer = 1, filter_size_1 = 8, kernel_size_1 = (3, 3), filter_size_2 = 8, kernel_size_2 = (3, 3), filter_size_3 = 8, kernel_size_3 = (3, 3)):
 
    x1 = Conv2D(filters=filter_size_1, kernel_size=kernel_size_1, padding='same', strides = stride_first_layer)(x)   
    x1 = BatchNormalization()(x1)
    x1 = mish(x1)

    x1 = Conv2D(filters=filter_size_2, kernel_size=kernel_size_2, padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = mish(x1)

    x1 = Conv2D(filters=filter_size_3, kernel_size=kernel_size_3, padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = mish(x1)

    # Skip connection
    x = Conv2D(filters = filter_size_3, kernel_size= (1,1), padding='same', strides= stride_first_layer)(x)
    x = BatchNormalization()(x)
    x = mish(x)
    
    x = Add()([x1, x])
    
    return x

def Res50_block_2D(x, stride_first_layer = 1, filter_size_1 = 8, kernel_size_1 = (3, 3), filter_size_2 = 8, kernel_size_2 = (3, 3), filter_size_3 = 8, kernel_size_3 = (3, 3)):
 
    x1 = Conv2D(filters=filter_size_1, kernel_size=kernel_size_1, padding='same', strides = stride_first_layer)(x)   
    x1 = BatchNormalization()(x1)
    x1 = mish(x1)

    x1 = Conv2D(filters=filter_size_2, kernel_size=kernel_size_2, padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = mish(x1)

    x1 = Conv2D(filters=filter_size_3, kernel_size=kernel_size_3, padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = mish(x1)
    
    x = Add()([x1, x])
    
    return x
    

# ResNet34
def ResNet34_2D(input_shape):

    input_layer = Input(shape = input_shape)
    
    x = Conv2D(filters=64, kernel_size=(7, 7),  padding='same', strides=2)(input_layer)
    x = MaxPooling2D(pool_size = (3,3), strides = (2,2), padding='same')(x)
    
    for iBlock in range(3):
    	x = Res34_block_2D(x, stride_first_layer = 1, filter_size_1 = 64, kernel_size_1 = (3, 3), filter_size_2 = 64, kernel_size_2 = (3, 3))
    
    for iBlock in range(4):
    	if(iBlock != 0):
    		x = Res34_block_2D(x, stride_first_layer = 1, filter_size_1 = 128, kernel_size_1 = (3, 3), filter_size_2 = 128, kernel_size_2 = (3, 3))
    	else:
    		x = Res34_BottleNeckBlock_2D(x, stride_first_layer = 2, filter_size_1 = 128, kernel_size_1 = (3, 3), filter_size_2 = 128, kernel_size_2 = (3, 3))
    		
    for iBlock in range(6):
    	if(iBlock != 0):
    		x = Res34_block_2D(x, stride_first_layer = 1, filter_size_1 = 256, kernel_size_1 = (3, 3), filter_size_2 = 256, kernel_size_2 = (3, 3))
    	else:
    		x = Res34_BottleNeckBlock_2D(x, stride_first_layer = 2, filter_size_1 = 256, kernel_size_1 = (3, 3), filter_size_2 = 256, kernel_size_2 = (3, 3))
     		
    for iBlock in range(3):
    	if(iBlock != 0):		
    		x = Res34_block_2D(x, stride_first_layer = 1, filter_size_1 = 512, kernel_size_1 = (3, 3), filter_size_2 = 512, kernel_size_2 = (3, 3))
    	else:
    		x = Res34_BottleNeckBlock_2D(x, stride_first_layer = 2, filter_size_1 = 512, kernel_size_1 = (3, 3), filter_size_2 = 512, kernel_size_2 = (3, 3))
    
     
    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    x = Dense(1, activation = 'linear')(x)
	
    model = Model(inputs = input_layer, outputs = x)
    
    return model
    

def Res34_block_2D(x, stride_first_layer = 1, filter_size_1 = 8, kernel_size_1 = (3, 3), filter_size_2 = 8, kernel_size_2 = (3, 3)):
    
    x1 = Conv2D(filters=filter_size_1, kernel_size=kernel_size_1, padding='same', strides = stride_first_layer)(x)
    x1 = BatchNormalization()(x1)
    x1 = mish(x1)
    
    x1 = Conv2D(filters=filter_size_2, kernel_size=kernel_size_2, padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = mish(x1)

   
    x = Add()([x1, x])
    
    return x
    


def Res34_BottleNeckBlock_2D(x, stride_first_layer = 1, filter_size_1 = 8, kernel_size_1 = (3, 3), filter_size_2 = 8, kernel_size_2 = (3, 3)):
    
    x1 = Conv2D(filters=filter_size_1, kernel_size=kernel_size_1, padding='same', strides = stride_first_layer)(x)
    x1 = BatchNormalization()(x1)
    x1 = mish(x1)

    x1 = Conv2D(filters=filter_size_2, kernel_size=kernel_size_2, padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = mish(x1)

   
    x = Conv2D(filters = filter_size_2, kernel_size= (1,1), padding='same', strides= stride_first_layer)(x)
    x = BatchNormalization()(x)
    x = mish(x)
    
    x = Add()([x1, x])
    
    return x
    
# ResNet34 dropout   
def ResNet34_dropout_2D(input_shape, dt_ratio):

    input_layer = Input(shape = input_shape)
    
    x = Conv2D(filters=64, kernel_size=(7, 7),  padding='same', strides=2)(input_layer)
    x = MaxPooling2D(pool_size = (3,3), strides = (2,2), padding='same')(x)
    
    for iBlock in range(3):
    	if(iBlock == 2):
    		x = Res34_block_dropout_2D(x, stride_first_layer = 1, filter_size_1 = 64, kernel_size_1 = (3, 3), filter_size_2 = 64, kernel_size_2 = (3, 3), dropout_ratio = dt_ratio)
    	else:
    		x = Res34_block_2D(x, stride_first_layer = 1, filter_size_1 = 64, kernel_size_1 = (3, 3), filter_size_2 = 64, kernel_size_2 = (3, 3))
    
    for iBlock in range(4):
    	if(iBlock == 0):
    		x = Res34_BottleNeckBlock_2D(x, stride_first_layer = 2, filter_size_1 = 128, kernel_size_1 = (3, 3), filter_size_2 = 128, kernel_size_2 = (3, 3))
    	if(iBlock == 3):
    		x = Res34_block_dropout_2D(x, stride_first_layer = 1, filter_size_1 = 128, kernel_size_1 = (3, 3), filter_size_2 = 128, kernel_size_2 = (3, 3), dropout_ratio = dt_ratio)
    	else:
    		x = Res34_block_2D(x, stride_first_layer = 1, filter_size_1 = 128, kernel_size_1 = (3, 3), filter_size_2 = 128, kernel_size_2 = (3, 3))
	
    for iBlock in range(6):
    	if(iBlock == 0):
    		x = Res34_BottleNeckBlock_2D(x, stride_first_layer = 2, filter_size_1 = 256, kernel_size_1 = (3, 3), filter_size_2 = 256, kernel_size_2 = (3, 3))
    	if(iBlock == 5):
    		x = Res34_block_dropout_2D(x, stride_first_layer = 1, filter_size_1 = 256, kernel_size_1 = (3, 3), filter_size_2 = 256, kernel_size_2 = (3, 3), dropout_ratio = dt_ratio)
    	else:
    		x = Res34_block_2D(x, stride_first_layer = 1, filter_size_1 = 256, kernel_size_1 = (3, 3), filter_size_2 = 256, kernel_size_2 = (3, 3))
    		
     		
    for iBlock in range(3):
    	if(iBlock == 0):
    		x = Res34_BottleNeckBlock_2D(x, stride_first_layer = 2, filter_size_1 = 512, kernel_size_1 = (3, 3), filter_size_2 = 512, kernel_size_2 = (3, 3))
    	if(iBlock == 2):		
    		x = Res34_block_dropout_2D(x, stride_first_layer = 1, filter_size_1 = 512, kernel_size_1 = (3, 3), filter_size_2 = 512, kernel_size_2 = (3, 3), dropout_ratio = dt_ratio)
    	else:
    		x = Res34_block_2D(x, stride_first_layer = 1, filter_size_1 = 512, kernel_size_1 = (3, 3), filter_size_2 = 512, kernel_size_2 = (3, 3))
    		   
    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    x = Dense(1, activation = 'linear')(x)
	
    model = Model(inputs = input_layer, outputs = x)
    
    return model

def Res34_block_dropout_2D(x, stride_first_layer = 1, filter_size_1 = 8, kernel_size_1 = (3, 3), filter_size_2 = 8, kernel_size_2 = (3, 3), dropout_ratio = 0.2):
    
    x1 = Conv2D(filters=filter_size_1, kernel_size=kernel_size_1, padding='same', strides = stride_first_layer)(x)
    x1 = SpatialDropout2D(dropout_ratio)(x1)
    x1 = BatchNormalization()(x1)
    x1 = mish(x1)

    
    x1 = Conv2D(filters=filter_size_2, kernel_size=kernel_size_2, padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = mish(x1)

   
    x = Add()([x1, x])
    
    return x
    
# Bayesian ResNet34  
def Bayesian_ResNet34_2D(input_shape, _dropout_ratio):

    input_layer = Input(shape = input_shape)
    
    x = Conv2D(filters=64, kernel_size=(7, 7),  padding='same', strides=2)(input_layer)
    x = MaxPooling2D(pool_size = (3,3), strides = (2,2), padding='same')(x)
    
    for iBlock in range(3):
    	x = Res34_block_2D(x, stride_first_layer = 1, filter_size_1 = 64, kernel_size_1 = (3, 3), filter_size_2 = 64, kernel_size_2 = (3, 3))
    		   
 
    for iBlock in range(4):
    	if(iBlock == 0):
    		x = Res34_BottleNeckBlock_2D(x, stride_first_layer = 2, filter_size_1 = 128, kernel_size_1 = (3, 3), filter_size_2 = 128, kernel_size_2 = (3, 3))
    	if(iBlock == 3):
    		x = Bayesian_Res34_block_2D(x, stride_first_layer = 1, filter_size_1 = 128, kernel_size_1 = (3, 3), filter_size_2 = 128, kernel_size_2 = (3, 3), dropout_ratio = _dropout_ratio)
    	else:
    		x = Res34_block_2D(x, stride_first_layer = 1, filter_size_1 = 128, kernel_size_1 = (3, 3), filter_size_2 = 128, kernel_size_2 = (3, 3))
    		
    for iBlock in range(6):
    	if(iBlock == 0):
    		x = Res34_BottleNeckBlock_2D(x, stride_first_layer = 2, filter_size_1 = 256, kernel_size_1 = (3, 3), filter_size_2 = 256, kernel_size_2 = (3, 3))
    	if(iBlock == 5):
    		x = Bayesian_Res34_block_2D(x, stride_first_layer = 1, filter_size_1 = 256, kernel_size_1 = (3, 3), filter_size_2 = 256, kernel_size_2 = (3, 3), dropout_ratio = _dropout_ratio)
    	else:
    		x = Res34_block_2D(x, stride_first_layer = 1, filter_size_1 = 256, kernel_size_1 = (3, 3), filter_size_2 = 256, kernel_size_2 = (3, 3))
     		
    for iBlock in range(3):
    	if(iBlock == 0):
    		x = Res34_BottleNeckBlock_2D(x, stride_first_layer = 2, filter_size_1 = 512, kernel_size_1 = (3, 3), filter_size_2 = 512, kernel_size_2 = (3, 3))
    	if(iBlock == 2):		
    		x = Bayesian_Res34_block_2D(x, stride_first_layer = 1, filter_size_1 = 512, kernel_size_1 = (3, 3), filter_size_2 = 512, kernel_size_2 = (3, 3), dropout_ratio = _dropout_ratio)
    	else:
    		x = Res34_block_2D(x, stride_first_layer = 1, filter_size_1 = 512, kernel_size_1 = (3, 3), filter_size_2 = 512, kernel_size_2 = (3, 3))
     
    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    x = Dense(1, activation = 'linear')(x)
	
    model = Model(inputs = input_layer, outputs = x)
    
    return model

def Bayesian_Res34_block_2D(x, stride_first_layer = 1, filter_size_1 = 8, kernel_size_1 = (3, 3), filter_size_2 = 8, kernel_size_2 = (3, 3), dropout_ratio = 0.2):
    
    x1 = Conv2D(filters=filter_size_1, kernel_size=kernel_size_1, padding='same', strides = stride_first_layer)(x)
    x1 = SpatialDropout2D(dropout_ratio)(x1, training = True)
    x1 = BatchNormalization()(x1)
    x1 = mish(x1)
    
    x1 = Conv2D(filters=filter_size_2, kernel_size=kernel_size_2, padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = mish(x1)

   
    x = Add()([x1, x])
    
    return x

def Bayesian_Res34_BottleNeckBlock_2D(x, stride_first_layer = 1, filter_size_1 = 8, kernel_size_1 = (3, 3), filter_size_2 = 8, kernel_size_2 = (3, 3), dropout_ratio = 0.2):
    
    x1 = Conv2D(filters=filter_size_1, kernel_size=kernel_size_1, padding='same', strides = stride_first_layer)(x)
    x1 = SpatialDropout2D(dropout_ratio)(x1, training = True)
    x1 = BatchNormalization()(x1)
    x1 = mish(x1)

    x1 = Conv2D(filters=filter_size_2, kernel_size=kernel_size_2, padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = mish(x1)

   
    x = Conv2D(filters = filter_size_2, kernel_size= (1,1), padding='same', strides= stride_first_layer)(x)
    x = BatchNormalization()(x)
    x = mish(x)
    
    x = Add()([x1, x])
    
    return x
    
#ResNeXt
def ResNeXt_2D(input_shape):

    input_layer = Input(shape = input_shape)
    
    x = Conv2D(filters=64, kernel_size=(7, 7),  padding='same', strides=2)(input_layer)
    x = MaxPooling2D(pool_size = (3,3), strides = (2,2), padding='same')(x)
    
    for iBlock in range(3):
    	if(iBlock != 0):
    		x = ResXt_block_2D(x, stride_first_layer = 1, filter_size_1 = 128, kernel_size_1 = (1, 1), filter_size_2 = 128, kernel_size_2 = (3, 3), filter_size_3 = 256, kernel_size_3 = (1, 1))
    	else:
    		x = ResXt_bottleNeckblock_2D(x, stride_first_layer = 1, filter_size_1 = 128, kernel_size_1 = (1, 1), filter_size_2 = 128, kernel_size_2 = (3, 3), filter_size_3 = 256, kernel_size_3 = (1, 1))
	   
    for iBlock in range(4):
    	if(iBlock != 0):
    		x = ResXt_block_2D(x, stride_first_layer = 1, filter_size_1 = 256, kernel_size_1 = (1, 1), filter_size_2 = 256, kernel_size_2 = (3, 3), filter_size_3 = 256, kernel_size_3 = (1, 1))
    	else:
    		x = ResXt_bottleNeckblock_2D(x, stride_first_layer = 2, filter_size_1 = 256, kernel_size_1 = (1, 1), filter_size_2 = 256, kernel_size_2 = (3, 3), filter_size_3 = 256, kernel_size_3 = (1, 1))
    
    for iBlock in range(6):
    	if(iBlock != 0):
    		x = ResXt_block_2D(x, stride_first_layer = 1, filter_size_1 = 512, kernel_size_1 = (1, 1), filter_size_2 = 512, kernel_size_2 = (3, 3), filter_size_3 = 1024, kernel_size_3 = (1, 1))
    	else:
    		x = ResXt_bottleNeckblock_2D(x, stride_first_layer = 2, filter_size_1 = 512, kernel_size_1 = (1, 1), filter_size_2 = 512, kernel_size_2 = (3, 3), filter_size_3 = 1024, kernel_size_3 = (1, 1))
     		
    for iBlock in range(3):
    	if(iBlock != 0):		
    		x = ResXt_block_2D(x, stride_first_layer = 1, filter_size_1 = 1024, kernel_size_1 = (1, 1), filter_size_2 = 1024, kernel_size_2 = (3, 3), filter_size_3 = 2048, kernel_size_3 = (1, 1))
    	else:
    		x = ResXt_bottleNeckblock_2D(x, stride_first_layer = 2, filter_size_1 = 1024, kernel_size_1 = (1, 1), filter_size_2 = 1024, kernel_size_2 = (3, 3), filter_size_3 = 2048, kernel_size_3 = (1, 1))
    
     
    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    x = Dense(1, activation = 'linear')(x)
	
    model = Model(inputs = input_layer, outputs = x)
    
    return model

    
def ResXt_block_2D(x, stride_first_layer = 1, filter_size_1 = 8, kernel_size_1 = (3, 3), filter_size_2 = 8, kernel_size_2 = (3, 3), filter_size_3 = 8, kernel_size_3 = (3, 3)):
 
    x1 = Conv2D(filters=filter_size_1, kernel_size=kernel_size_1, padding='same', strides = stride_first_layer)(x)   
    x1 = BatchNormalization()(x1)
    x1 = mish(x1)

    x1 = grouped_convolution(x1, filter_size_2, kernel_size_2, 1, 32)
    x1 = BatchNormalization()(x1)
    x1 = mish(x1)

    x1 = Conv2D(filters=filter_size_3, kernel_size=kernel_size_3, padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = mish(x1)
    
    x = Add()([x1, x])
    
    return x

    
def ResXt_bottleNeckblock_2D(x, stride_first_layer = 1, filter_size_1 = 8, kernel_size_1 = (3, 3), filter_size_2 = 8, kernel_size_2 = (3, 3), filter_size_3 = 8, kernel_size_3 = (3, 3)):
 
    x1 = Conv2D(filters=filter_size_1, kernel_size=kernel_size_1, padding='same', strides = stride_first_layer)(x)   
    x1 = BatchNormalization()(x1)
    x1 = mish(x1)

    x1 = grouped_convolution(x1, filter_size_2, kernel_size_2, 1, 32)
    x1 = BatchNormalization()(x1)
    x1 = mish(x1)

    x1 = Conv2D(filters=filter_size_3, kernel_size=kernel_size_3, padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = mish(x1)

    # Skip connection
    x = Conv2D(filters = filter_size_3, kernel_size= (1,1), padding='same', strides= stride_first_layer)(x)
    x = BatchNormalization()(x)
    x = mish(x)
    
    x = Add()([x1, x])
    
    return x


def grouped_convolution(x, filter_size, kernel_size, stride, cardinality):

	d = filter_size // cardinality
	groups = []
	
	for iCardinality in range(cardinality):
		group = Lambda(lambda z: z[:, :, :, iCardinality * d: iCardinality * d + d])(x)
		groups.append(Conv2D(filters=d, kernel_size = kernel_size, padding='same', strides = stride)(group))
	
	x = concatenate(groups)
		
	return x
	
	
#Attention module 	
def channelAttentionModule(x, ratio):

	#Get the size of the channel dimension for the input layer
	channel = x.shape[-1]
	
	#Compute average pooling in the channel dimension
	avgPool = GlobalAveragePooling2D(keepdims = "True")(x)
	
	#Compute max pooling in the channel dimension
	maxPool = GlobalMaxPooling2D(keepdims = "True")(x)
	
	#Build the shared multi-layer perceptron
	shared_MLP1 = Dense(channel // ratio)
	shared_MLP2 = mish
	shared_MLP3 = Dense(channel)
	
	#Apply the shared multi-layer perceptron in the avgPool and maxPool
	avgPool = shared_MLP3(shared_MLP2(shared_MLP1(avgPool)))
# 	avgPool = BatchNormalization()(avgPool)
	
	maxPool = shared_MLP3(shared_MLP2(shared_MLP1(maxPool)))
# 	maxPool = BatchNormalization()(maxPool)
	
	feature = Add()([avgPool, maxPool])
	feature = Activation("sigmoid")(feature)
	
	return Multiply()([feature, x])

def channelAttention3DModule(x, ratio):

	#Get the size of the channel dimension for the input layer
	channel = x.shape[-1]
	
	#Compute average pooling in the channel dimension
	avgPool = GlobalAveragePooling3D(keepdims = "True")(x)
	
	#Compute max pooling in the channel dimension
	maxPool = GlobalMaxPooling3D(keepdims = "True")(x)
	
	#Build the shared multi-layer perceptron
	shared_MLP1 = Dense(channel // ratio)
	shared_MLP2 = mish
	shared_MLP3 = Dense(channel)
	
	#Apply the shared multi-layer perceptron in the avgPool and maxPool
	avgPool = shared_MLP3(shared_MLP2(shared_MLP1(avgPool)))
# 	avgPool = BatchNormalization()(avgPool)
	
	maxPool = shared_MLP3(shared_MLP2(shared_MLP1(maxPool)))
# 	maxPool = BatchNormalization()(maxPool)
	
	feature = Add()([avgPool, maxPool])
	feature = Activation("sigmoid")(feature)
	
	return Multiply()([feature, x])
	
	
def ECA(x, k_size = 3):
	
	#Compute average pooling in the channel dimension
	avgPool = GlobalAveragePooling2D(keepdims = "True")(x)
	
	feature = Conv1D(filters = 1, kernel_size = k_size, strides = 1, padding = "same", use_bias = False)(avgPool)
	feature = Activation("sigmoid")(feature)
	
	return Multiply()([feature, x])
	
def spatialAttentionModule(x, nModule, kernel_size_spatialAttention):
	
	#Compute the average pooling in the channel dimension
	avgPool = tf.reduce_mean(x, axis = -1, keepdims = True)
	
	#Compute the max pooling in the channel dimension
	maxPool = tf.reduce_max(x, axis = -1, keepdims = True)
	
	concat = Concatenate(axis = -1)([avgPool, maxPool])
	
	feature = Conv2D(filters = 1, kernel_size = kernel_size_spatialAttention, strides = 1, padding = "same", use_bias=False)(concat)
	feature = BatchNormalization()(feature)
	feature = Activation("sigmoid", name = "Attention_map_" + str(nModule))(feature)
	
	return Multiply()([feature, x])
	
def spatialAttention3DModule(x, nModule, kernel_size_spatialAttention):
	
	#Compute the average pooling in the channel dimension
	avgPool = tf.reduce_mean(x, axis = -1, keepdims = True)
	
	#Compute the max pooling in the channel dimension
	maxPool = tf.reduce_max(x, axis = -1, keepdims = True)
	
	concat = Concatenate(axis = -1)([avgPool, maxPool])
	
	feature = Conv3D(filters = 1, kernel_size = kernel_size_spatialAttention, strides = 1, padding = "same", use_bias=False)(concat)
	feature = BatchNormalization()(feature)
	feature = Activation("sigmoid", name = "Attention_map_" + str(nModule))(feature)
	
	return Multiply()([feature, x])
	
	
def cbam_block(x, nModule = 0, ratio = 2, kernel_size_spatialAttention = (7,7)):

	x = channelAttentionModule(x, ratio)
	x = spatialAttentionModule(x, nModule, kernel_size_spatialAttention)
	
	return x
	
def cbam_parallel_block(x, nModule = 0, ratio = 2, kernel_size_spatialAttention = (7,7)):
	
	#### Compute channel attention ######
	
	#Get the size of the channel dimension for the input layer
	channel = x.shape[-1]
	
	#Compute average pooling in the channel dimension
	avgPool_channelAttention = GlobalAveragePooling2D(keepdims = "True")(x)
	
	#Compute max pooling in the channel dimension
	maxPool_channelAttention = GlobalMaxPooling2D(keepdims = "True")(x)
	
	#Build the shared multi-layer perceptron
	shared_MLP1 = Dense(channel // ratio)
	shared_MLP2 = mish
	shared_MLP3 = Dense(channel)
	
	#Apply the shared multi-layer perceptron in the avgPool and maxPool
	avgPool_channelAttention = shared_MLP3(shared_MLP2(shared_MLP1(avgPool_channelAttention)))
# 	avgPool = BatchNormalization()(avgPool)
	
	maxPool_channelAttention = shared_MLP3(shared_MLP2(shared_MLP1(maxPool_channelAttention)))
# 	maxPool = BatchNormalization()(maxPool)
	
	feature_channelAttention = Add()([avgPool_channelAttention, maxPool_channelAttention])
	feature_channelAttention = Activation("sigmoid")(feature_channelAttention)
	
	
	### Compute the attention mechanism ###
	
	#Compute the average pooling in the channel dimension
	avgPool_attentionlAttention = tf.reduce_mean(x, axis = -1, keepdims = True)
	
	#Compute the max pooling in the channel dimension
	maxPool_attentionlAttention = tf.reduce_max(x, axis = -1, keepdims = True)
	
	concat = Concatenate(axis = -1)([avgPool_attentionlAttention, maxPool_attentionlAttention])
	
	feature_attentionlAttention = Conv2D(filters = 1, kernel_size = kernel_size_spatialAttention, strides = 1, padding = "same", use_bias=False)(concat)
	feature_attentionlAttention = BatchNormalization()(feature_attentionlAttention)
	feature_attentionlAttention = Activation("sigmoid", name = "Attention_map_" + str(nModule))(feature_attentionlAttention)

	### Apply the channel and attention mechanism ####
	
	x = Multiply()([feature_channelAttention, x])
	x = Multiply()([feature_attentionlAttention, x])
	
	return x
	

# ResNet34 + CBAM attention module	
def ResNet34Attention_2D(input_shape):

    input_layer = Input(shape = input_shape)
    
    nModule = 1
    
    x = Conv2D(filters=64, kernel_size=(7, 7),  padding='same', strides=2)(input_layer)
    x = cbam_block(x, nModule, ratio = 2, kernel_size_spatialAttention = (7,7))
    x = MaxPooling2D(pool_size = (3,3), strides = (2,2), padding='same')(x)
    

    for iBlock in range(3):
    	nModule += 1
    	x = Res34Attention_block_2D(x, stride_first_layer = 1, filter_size_1 = 64, kernel_size_1 = (3, 3), filter_size_2 = 64, kernel_size_2 = (3, 3), nMod = nModule, ratio = 2, kernel_size_spatialAttention = (7,7))
    
    for iBlock in range(4):
    	nModule += 1
    	if(iBlock != 0):
    		x = Res34Attention_block_2D(x, stride_first_layer = 1, filter_size_1 = 128, kernel_size_1 = (3, 3), filter_size_2 = 128, kernel_size_2 = (3, 3), nMod = nModule, ratio = 2, kernel_size_spatialAttention = (5, 5))
    	else:
    		x = Res34Attention_BottleNeckBlock_2D(x, stride_first_layer = 2, filter_size_1 = 128, kernel_size_1 = (3, 3), filter_size_2 = 128, kernel_size_2 = (3, 3), nMod = nModule, ratio = 2, kernel_size_spatialAttention = (5, 5))
    		
    for iBlock in range(6):
    	nModule += 1
    	if(iBlock != 0):
    		x = Res34Attention_block_2D(x, stride_first_layer = 1, filter_size_1 = 256, kernel_size_1 = (3, 3), filter_size_2 = 256, kernel_size_2 = (3, 3), nMod = nModule, ratio = 2, kernel_size_spatialAttention = (3, 3))
    	else:
    		x = Res34Attention_BottleNeckBlock_2D(x, stride_first_layer = 2, filter_size_1 = 256, kernel_size_1 = (3, 3), filter_size_2 = 256, kernel_size_2 = (3, 3), nMod = nModule, ratio = 2, kernel_size_spatialAttention = (3, 3))
     		
    for iBlock in range(3):
    	nModule += 1
    	if(iBlock != 0):		
    		x = Res34_block_2D(x, stride_first_layer = 1, filter_size_1 = 512, kernel_size_1 = (3, 3), filter_size_2 = 512, kernel_size_2 = (3, 3))
    	else:
    		x = Res34_BottleNeckBlock_2D(x, stride_first_layer = 2, filter_size_1 = 512, kernel_size_1 = (3, 3), filter_size_2 = 512, kernel_size_2 = (3, 3))
    
     
    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    x = Dense(1, activation = 'linear')(x)
	
    model = Model(inputs = input_layer, outputs = x)
    
    return model
    

def Res34Attention_block_2D(x, stride_first_layer = 1, filter_size_1 = 8, kernel_size_1 = (3, 3), filter_size_2 = 8, kernel_size_2 = (3, 3), nMod = 0, ratio = 2, kernel_size_spatialAttention = (7,7)):
    
    x1 = Conv2D(filters=filter_size_1, kernel_size=kernel_size_1, padding='same', strides = stride_first_layer)(x)
    x1 = BatchNormalization()(x1)
    x1 = mish(x1)
    
    x1 = Conv2D(filters=filter_size_2, kernel_size=kernel_size_2, padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = cbam_block(x1, nMod, ratio, kernel_size_spatialAttention) #Attention computation
    x1 = mish(x1)
    
   
    x = Add()([x1, x])
    
    return x
    

def Res34Attention_BottleNeckBlock_2D(x, stride_first_layer = 1, filter_size_1 = 8, kernel_size_1 = (3, 3), filter_size_2 = 8, kernel_size_2 = (3, 3), nMod = 0, ratio = 2, kernel_size_spatialAttention = (7,7)):
    
    x1 = Conv2D(filters=filter_size_1, kernel_size=kernel_size_1, padding='same', strides = stride_first_layer)(x)
    x1 = BatchNormalization()(x1)
    x1 = mish(x1)

    x1 = Conv2D(filters=filter_size_2, kernel_size=kernel_size_2, padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = cbam_block(x1, nMod, ratio, kernel_size_spatialAttention) #Attention computation
    x1 = mish(x1)
    


   
    x = Conv2D(filters = filter_size_2, kernel_size= (1,1), padding='same', strides= stride_first_layer)(x)
    x = BatchNormalization()(x)
    x = mish(x)
    
    x = Add()([x1, x])
    
    return x
	
	
def ResNet34SpatialAttention_2D(input_shape):

    input_layer = Input(shape = input_shape)
    
    nModule = 1
    
    x = Conv2D(filters=64, kernel_size=(7, 7),  padding='same', strides=2)(input_layer)
    x = spatialAttentionModule(x, nModule, kernel_size_spatialAttention = (7,7))
    x = MaxPooling2D(pool_size = (3,3), strides = (2,2), padding='same')(x)
    

    
    for iBlock in range(3):
    	nModule += 1
    	x = Res34SpatialAttention_block_2D(x, stride_first_layer = 1, filter_size_1 = 64, kernel_size_1 = (3, 3), filter_size_2 = 64, kernel_size_2 = (3, 3), nMod = nModule, kernel_size_spatialAttention = (7,7))
    
    for iBlock in range(4):
    	nModule += 1
    	if(iBlock != 0):
    		x = Res34SpatialAttention_block_2D(x, stride_first_layer = 1, filter_size_1 = 128, kernel_size_1 = (3, 3), filter_size_2 = 128, kernel_size_2 = (3, 3), nMod = nModule, kernel_size_spatialAttention = (5,5))
    	else:
    		x = Res34SpatialAttention_BottleNeckBlock_2D(x, stride_first_layer = 2, filter_size_1 = 128, kernel_size_1 = (3, 3), filter_size_2 = 128, kernel_size_2 = (3, 3), nMod = nModule, kernel_size_spatialAttention = (5,5))
    		
    for iBlock in range(6):
    	nModule += 1
    	if(iBlock != 0):
    		x = Res34SpatialAttention_block_2D(x, stride_first_layer = 1, filter_size_1 = 256, kernel_size_1 = (3, 3), filter_size_2 = 256, kernel_size_2 = (3, 3), nMod = nModule, kernel_size_spatialAttention = (3,3))
    	else:
    		x = Res34SpatialAttention_BottleNeckBlock_2D(x, stride_first_layer = 2, filter_size_1 = 256, kernel_size_1 = (3, 3), filter_size_2 = 256, kernel_size_2 = (3, 3), nMod = nModule, kernel_size_spatialAttention = (3,3))
     		
    for iBlock in range(3):
    	nModule += 1
    	if(iBlock != 0):		
    		x = Res34_block_2D(x, stride_first_layer = 1, filter_size_1 = 512, kernel_size_1 = (3, 3), filter_size_2 = 512, kernel_size_2 = (3, 3))
    	else:
    		x = Res34_BottleNeckBlock_2D(x, stride_first_layer = 2, filter_size_1 = 512, kernel_size_1 = (3, 3), filter_size_2 = 512, kernel_size_2 = (3, 3))
        
    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    x = Dense(1, activation = 'linear')(x)
	
    model = Model(inputs = input_layer, outputs = x)
    
    return model
    

def Res34SpatialAttention_block_2D(x, stride_first_layer = 1, filter_size_1 = 8, kernel_size_1 = (3, 3), filter_size_2 = 8, kernel_size_2 = (3, 3), nMod = 0, kernel_size_spatialAttention = (7,7)):
    
    x1 = Conv2D(filters=filter_size_1, kernel_size=kernel_size_1, padding='same', strides = stride_first_layer)(x)
    x1 = BatchNormalization()(x1)
    x1 = mish(x1)
    
    x1 = Conv2D(filters=filter_size_2, kernel_size=kernel_size_2, padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = spatialAttentionModule(x1, nMod, kernel_size_spatialAttention) #Attention computation
    x1 = mish(x1)
    


   
    x = Add()([x1, x])
    
    return x
    

def Res34SpatialAttention_BottleNeckBlock_2D(x, stride_first_layer = 1, filter_size_1 = 8, kernel_size_1 = (3, 3), filter_size_2 = 8, kernel_size_2 = (3, 3), nMod = 0, kernel_size_spatialAttention = (7,7)):
    
    x1 = Conv2D(filters=filter_size_1, kernel_size=kernel_size_1, padding='same', strides = stride_first_layer)(x)
    x1 = BatchNormalization()(x1)
    x1 = mish(x1)

    x1 = Conv2D(filters=filter_size_2, kernel_size=kernel_size_2, padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = spatialAttentionModule(x1, nMod, kernel_size_spatialAttention) #Attention computation
    x1 = mish(x1)
    


   
    x = Conv2D(filters = filter_size_2, kernel_size= (1,1), padding='same', strides= stride_first_layer)(x)
    x = BatchNormalization()(x)
    x = mish(x)
    
    x = Add()([x1, x])
    
    return x	

def Res34ChannelAttention_block_2D(x, stride_first_layer = 1, filter_size_1 = 8, kernel_size_1 = (3, 3), filter_size_2 = 8, kernel_size_2 = (3, 3), ratio = 1):
    
    x1 = Conv2D(filters=filter_size_1, kernel_size=kernel_size_1, padding='same', strides = stride_first_layer)(x)
    x1 = BatchNormalization()(x1)
    x1 = mish(x1)
    
    x1 = Conv2D(filters=filter_size_2, kernel_size=kernel_size_2, padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = channelAttentionModule(x1, ratio) #Attention computation
    x1 = mish(x1)
    
   
    x = Add()([x1, x])
    
    return x
    

def Res34ChannelAttention_BottleNeckBlock_2D(x, stride_first_layer = 1, filter_size_1 = 8, kernel_size_1 = (3, 3), filter_size_2 = 8, kernel_size_2 = (3, 3), ratio = 1):
    
    x1 = Conv2D(filters=filter_size_1, kernel_size=kernel_size_1, padding='same', strides = stride_first_layer)(x)
    x1 = BatchNormalization()(x1)
    x1 = mish(x1)

    x1 = Conv2D(filters=filter_size_2, kernel_size=kernel_size_2, padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = channelAttentionModule(x1, ratio) #Attention computation
    x1 = mish(x1)
    


   
    x = Conv2D(filters = filter_size_2, kernel_size= (1,1), padding='same', strides= stride_first_layer)(x)
    x = BatchNormalization()(x)
    x = mish(x)
    
    x = Add()([x1, x])
    
    return x
    
def Res34ChannelAttention_dropout_BottleNeckBlock_2D(x, stride_first_layer = 1, filter_size_1 = 8, kernel_size_1 = (3, 3), filter_size_2 = 8, kernel_size_2 = (3, 3), ratio = 1):
    
    x1 = Conv2D(filters=filter_size_1, kernel_size=kernel_size_1, padding='same', strides = stride_first_layer)(x)
    x1 = BatchNormalization()(x1)
    x1 = mish(x1)
    x1 = SpatialDropout2D(0.2)(x1)

    x1 = Conv2D(filters=filter_size_2, kernel_size=kernel_size_2, padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = channelAttentionModule(x1, ratio) #Attention computation
    x1 = mish(x1)
    


   
    x = Conv2D(filters = filter_size_2, kernel_size= (1,1), padding='same', strides= stride_first_layer)(x)
    x = BatchNormalization()(x)
    x = mish(x)
    
    x = Add()([x1, x])
    
    return x



		
		