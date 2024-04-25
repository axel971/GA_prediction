import numpy as np

def MAE(y_pred, y_true):
	
	return np.mean(np.absolute(y_pred - y_true))

def ME(y_pred, y_true):
	
	return np.mean(y_pred - y_true)