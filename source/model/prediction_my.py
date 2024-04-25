import sys
sys.path.append('/home/axel/dev/fetus_GA_prediction/source')

import numpy as np


    	
class BayesianPredict(object):
    'run the model on test data'

    def __init__(self, model, T	):
        'Initialization'
        self.model = model
        self.T = T

    def __run__(self, X):
    
    	Y0 = np.zeros(self.T, dtype=np.float32)
    			  			  	
    	for iPass in range(self.T):
    	    	
    		Y0[iPass] = self.model(X, training = False)
   		    	
    	return np.mean(Y0), np.std(Y0)
    	

class BayesianPredictMulti(object):
    'run the model on test data'

    def __init__(self, model, T	):
        'Initialization'
        self.model = model
        self.T = T

    def __run__(self, X):
    
    	Y0 = np.zeros((X.shape[0], self.T), dtype=np.float32)
    			  			  	
    	for iPass in range(self.T):
    	    	
    		Y0[ :, iPass] = np.squeeze(self.model.predict(X), axis = 1)
   		    	
    	return np.mean(Y0, axis = 1, keepdims = False), np.std(Y0, axis = 1, keepdims = False)