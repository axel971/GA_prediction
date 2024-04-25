
import numpy as np

# Get paths for each files
path_preedicted_GA = '/home/axel/dev/fetus_GA_prediction/data/output/2D_Bayesian_with_one_slice/exp1/GA_prediction.npy'
path_true_GA =  '/home/axel/dev/fetus_GA_prediction/data/output/2D_Bayesian_with_one_slice/exp1/GA.npy'
path_uncertainty =  '/home/axel/dev/fetus_GA_prediction/data/output/2D_Bayesian_with_one_slice/exp1/GA_uncertainty.npy'



# Open the files
predicted_GA = np.load(path_preedicted_GA)
true_GA = np.load(path_true_GA)
uncertainty = np.load(path_uncertainty)

print(uncertainty)
# Compute the wilxocon test
#print(wilcox.test(as.numeric(data1[ , 2]), as.numeric(data2[ , 2]), exact = T, paired = T, alternative = "greater"))
#print(wilcox.test(as.numeric(data1[ , 7]), as.numeric(data2[, 7]), exact = T, paired = T, alternative = "less"))