
import numpy as np
import tensorflow.keras
import imgaug.augmenters as iaa
import torchio as tio
import random

class Generator3D(tensorflow.keras.utils.Sequence):
    'Generates data for Keras, based on array data X and Y'

    def __init__(self, X, Y, batch_size=32):
        'Initialization'
        self.batch_size = batch_size
        self.image_size = np.asarray(X.shape[1:])
        self.n_subject = X.shape[0]

        self.indices_images = random.sample(range(self.n_subject), self.n_subject)
        
        
        self.X = X
        self.Y = Y
        
        self.step_by_epoch = 400
        self.current_epoch = 0
    	

    def __len__(self):
        'Denotes the number of step per epoch'
        return self.step_by_epoch


    def __getitem__(self, index):
        'Generate one batch of data'
          
        batch_image_indices = random.sample(self.indices_images, self.batch_size)     
            
        return np.expand_dims(self.X[batch_image_indices, :, :, :], -1), self.Y[batch_image_indices]
            
            



class Generator3D_augment(tensorflow.keras.utils.Sequence):
    'Generates data for Keras, based on array data X and Y'

    def __init__(self, X, Y, batch_size=32):
        'Initialization'
        self.batch_size = batch_size
        self.image_size = np.asarray(X.shape[1:])
        self.n_subject = X.shape[0]

        self.indices_images = random.sample(range(self.n_subject), self.n_subject)
           
        self.X = X
        self.Y = Y
        
        self.step_by_epoch = 400
        self.current_epoch = 0
        
        self.data_augmentation_transform = tio.RandomFlip(axes = 0, flip_probability = 0.5)
    	

    def __len__(self):
        'Denotes the number of step per epoch'
        return self.step_by_epoch


    def __getitem__(self, index):
        'Generate one batch of data'

        # Built arrays to stock the delineations and images for one bacth
        X = np.zeros((self.batch_size, self.image_size[0], self.image_size[1], self.image_size[2], 1), dtype=np.float32)
          
        batch_image_indices = random.sample(self.indices_images, self.batch_size)
        
        
        for batch_count in range(self.batch_size):
        
            X[batch_count, :, :, :, 0] = self.data_augmentation_transform(np.expand_dims(self.X[batch_image_indices[batch_count]], 0))[0]
            
                        
        return X, self.Y[batch_image_indices]
                       


class Generator3D_augment_weight(tensorflow.keras.utils.Sequence):
    'Generates data for Keras, based on array data X and Y'

    def __init__(self, X, Y, weights, batch_size=32):
        'Initialization'
        self.batch_size = batch_size
        self.image_size = np.asarray(X.shape[1:])
        self.n_subject = X.shape[0]

        self.indices_images = random.sample(range(self.n_subject), self.n_subject)
           
        self.X = X
        self.Y = Y
        self.weights = weights
        
        self.step_by_epoch = 400
        self.current_epoch = 0
        
        self.data_augmentation_transform = tio.RandomFlip(axes = 0, flip_probability = 0.5)
    	

    def __len__(self):
        'Denotes the number of step per epoch'
        return self.step_by_epoch


    def __getitem__(self, index):
        'Generate one batch of data'

        # Built arrays to stock the delineations and images for one bacth
        X = np.zeros((self.batch_size, self.image_size[0], self.image_size[1], self.image_size[2], 1), dtype=np.float32)
          
        batch_image_indices = random.sample(self.indices_images, self.batch_size)
        
        
        for batch_count in range(self.batch_size):
        
            X[batch_count, :, :, :, 0] = self.data_augmentation_transform(np.expand_dims(self.X[batch_image_indices[batch_count]], 0))[0]
            
                        
        return X, self.Y[batch_image_indices], self.weights[batch_image_indices]
                       
               

class Generator_2D(tensorflow.keras.utils.Sequence):
    'Generates data for Keras, based on array data X and Y'

    def __init__(self, X, Y, batch_size=32):
        'Initialization'
        self.batch_size = batch_size
        self.image_size = np.asarray(X.shape[1:])
        self.n_subject = X.shape[0]
        self.indices_images = random.sample(range(self.n_subject), self.n_subject)
                    
        self.X = X
        self.Y = Y
        
        self.step_by_epoch = 400
        self.current_epoch = 0
        
#         self.data_augmentation_transform = iaa.Sometimes(0.70, iaa.OneOf([iaa.Fliplr(1.0), iaa.Flipud(1.0), iaa.Affine(translate_px={"x": [-5, 5], "y": [-5, 5]}, rotate=(-5, 5), mode="constant", cval=0)]))
#         self.data_augmentation_transform = (tio.OneOf({tio.RandomAffine(scales = (1,1), degrees=(5,5,5), translation = (5, 5, 5)):0.4, tio.RandomFlip(axes = 0, flip_probability = 1): 0.4, tio.RandomMotion(): 0.2}, p = 0.80) )
    	

    def __len__(self):
        'Denotes the number of step per epoch'
        return self.step_by_epoch


    def __getitem__(self, index):
        'Generate one batch of data'

        # Built arrays to stock the delineations and images for one bacth         
        #batch_image_indices = np.random.choice(self.indices_images, self.batch_size)
        batch_image_indices = random.sample(self.indices_images, self.batch_size)
                   
        return np.expand_dims(self.X[batch_image_indices, :, :], -1), self.Y[batch_image_indices]
        
        

    def on_epoch_end(self):
        'Shuffle the image and patch indexes after each epoch'
        self.current_epoch += 1
#         np.random.shuffle(self.indices_images)


class Generator_2D_dataAugmentation(tensorflow.keras.utils.Sequence):
    'Generates data for Keras, based on array data X and Y'

    def __init__(self, X, Y, batch_size=32):
        'Initialization'
        self.batch_size = batch_size
        self.image_size = np.asarray(X.shape[1:])
        self.n_subject = X.shape[0]
        self.indices_images = random.sample(range(self.n_subject), self.n_subject)
        
            
        self.X = X
        self.Y = Y
        
        self.step_by_epoch = 400
        self.current_epoch = 0
        
        self.data_augmentation_transform = iaa.Flipud(0.5)
        
        
    def __len__(self):
        'Denotes the number of step per epoch'
        return self.step_by_epoch


    def __getitem__(self, index):
        'Generate one batch of data'

        # Built arrays to stock the delineations and images for one bacth         
        #batch_image_indices = np.random.choice(self.indices_images, self.batch_size)
        batch_image_indices = random.sample(self.indices_images, self.batch_size)
                   
        return np.expand_dims(self.data_augmentation_transform.augment_images(self.X[batch_image_indices, :, :]), -1), self.Y[batch_image_indices]
        

    def on_epoch_end(self):
        'Shuffle the image and patch indexes after each epoch'
        self.current_epoch += 1
#         np.random.shuffle(self.indices_images)
  

class Generator_2D_weight(tensorflow.keras.utils.Sequence):
    'Generates data for Keras, based on array data X and Y'

    def __init__(self, X, Y, weights, batch_size=32):
        'Initialization'
        self.batch_size = batch_size
        self.image_size = np.asarray(X.shape[1:])
        self.n_subject = X.shape[0]
        self.indices_images = random.sample(range(self.n_subject), self.n_subject)
                    
        self.X = X
        self.Y = Y
        self.weights = weights
        
        self.step_by_epoch = 400
        self.current_epoch = 0
        
#         self.data_augmentation_transform = iaa.Sometimes(0.70, iaa.OneOf([iaa.Fliplr(1.0), iaa.Flipud(1.0), iaa.Affine(translate_px={"x": [-5, 5], "y": [-5, 5]}, rotate=(-5, 5), mode="constant", cval=0)]))
#         self.data_augmentation_transform = (tio.OneOf({tio.RandomAffine(scales = (1,1), degrees=(5,5,5), translation = (5, 5, 5)):0.4, tio.RandomFlip(axes = 0, flip_probability = 1): 0.4, tio.RandomMotion(): 0.2}, p = 0.80) )
    	

    def __len__(self):
        'Denotes the number of step per epoch'
        return self.step_by_epoch


    def __getitem__(self, index):
        'Generate one batch of data'

        # Built arrays to stock the delineations and images for one bacth         
        #batch_image_indices = np.random.choice(self.indices_images, self.batch_size)
        batch_image_indices = random.sample(self.indices_images, self.batch_size)
                   
        return np.expand_dims(self.X[batch_image_indices, :, :], -1), self.Y[batch_image_indices], self.weights[batch_image_indices]
        
        

    def on_epoch_end(self):
        'Shuffle the image and patch indexes after each epoch'
        self.current_epoch += 1
#         np.random.shuffle(self.indices_images)


class Generator_2D_dataAugmentation_weight(tensorflow.keras.utils.Sequence):
    'Generates data for Keras, based on array data X and Y'

    def __init__(self, X, Y, weights, batch_size=32):
        'Initialization'
        self.batch_size = batch_size
        self.image_size = np.asarray(X.shape[1:])
        self.n_subject = X.shape[0]
        self.indices_images = random.sample(range(self.n_subject), self.n_subject)
        
            
        self.X = X
        self.Y = Y
        self.weights = weights
        
        self.step_by_epoch = 400
        self.current_epoch = 0
        
        self.data_augmentation_transform = iaa.Flipud(0.5)
        
        
    def __len__(self):
        'Denotes the number of step per epoch'
        return self.step_by_epoch


    def __getitem__(self, index):
        'Generate one batch of data'

        # Built arrays to stock the delineations and images for one bacth         
        #batch_image_indices = np.random.choice(self.indices_images, self.batch_size)
        batch_image_indices = random.sample(self.indices_images, self.batch_size)
                   
        return np.expand_dims(self.data_augmentation_transform.augment_images(self.X[batch_image_indices, :, :]), -1), self.Y[batch_image_indices], self.weights[batch_image_indices]
        

    def on_epoch_end(self):
        'Shuffle the image and patch indexes after each epoch'
        self.current_epoch += 1
#         np.random.shuffle(self.indices_images)
  
