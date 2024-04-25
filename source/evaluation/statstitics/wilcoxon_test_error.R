library(reticulate)
np = import("numpy")


# Get paths for each files
path_predicted_GA_exp1_DLM1 = '/home/axel/dev/fetus_GA_prediction/data/output/2D_Bayesian_uncertaintyGuided/exp1/GA_prediction.npy'
path_predicted_GA_exp2_DLM1 =  '/home/axel/dev/fetus_GA_prediction/data/output/2D_Bayesian_uncertaintyGuided/exp2/GA_prediction.npy'
path_predicted_GA_exp3_DLM1 = '/home/axel/dev/fetus_GA_prediction/data/output/2D_Bayesian_uncertaintyGuided/exp3/GA_prediction.npy'
path_predicted_GA_exp4_DLM1 = '/home/axel/dev/fetus_GA_prediction/data/output/2D_Bayesian_uncertaintyGuided/exp4/GA_prediction.npy'
path_predicted_GA_exp5_DLM1 =  '/home/axel/dev/fetus_GA_prediction/data/output/2D_Bayesian_uncertaintyGuided/exp5/GA_prediction.npy'

path_true_GA_exp1_DLM1 = '/home/axel/dev/fetus_GA_prediction/data/output/2D_Bayesian_uncertaintyGuided/exp1/GA.npy'
path_true_GA_exp2_DLM1 =  '/home/axel/dev/fetus_GA_prediction/data/output/2D_Bayesian_uncertaintyGuided/exp2/GA.npy'
path_true_GA_exp3_DLM1 = '/home/axel/dev/fetus_GA_prediction/data/output/2D_Bayesian_uncertaintyGuided/exp3/GA.npy'
path_true_GA_exp4_DLM1 = '/home/axel/dev/fetus_GA_prediction/data/output/2D_Bayesian_uncertaintyGuided/exp4/GA.npy'
path_true_GA_exp5_DLM1 = '/home/axel/dev/fetus_GA_prediction/data/output/2D_Bayesian_uncertaintyGuided/exp5/GA.npy'

path_predicted_GA_exp1_DLM2 = '/home/axel/dev/fetus_GA_prediction/data/output/3D/exp1/GA_prediction.npy'
path_predicted_GA_exp2_DLM2 =  '/home/axel/dev/fetus_GA_prediction/data/output/3D/exp2/GA_prediction.npy'
path_predicted_GA_exp3_DLM2 = '/home/axel/dev/fetus_GA_prediction/data/output/3D/exp3/GA_prediction.npy'
path_predicted_GA_exp4_DLM2 = '/home/axel/dev/fetus_GA_prediction/data/output/3D/exp4/GA_prediction.npy'
path_predicted_GA_exp5_DLM2 =  '/home/axel/dev/fetus_GA_prediction/data/output/3D/exp5/GA_prediction.npy'

path_true_GA_exp1_DLM2 = '/home/axel/dev/fetus_GA_prediction/data/output/3D/exp1/GA.npy'
path_true_GA_exp2_DLM2 =  '/home/axel/dev/fetus_GA_prediction/data/output/3D/exp2/GA.npy'
path_true_GA_exp3_DLM2 = '/home/axel/dev/fetus_GA_prediction/data/output/3D/exp3/GA.npy'
path_true_GA_exp4_DLM2 = '/home/axel/dev/fetus_GA_prediction/data/output/3D/exp4/GA.npy'
path_true_GA_exp5_DLM2 = '/home/axel/dev/fetus_GA_prediction/data/output/3D/exp5/GA.npy'

#Load the python files
predicted_GA_exp1_DLM1 = as.numeric(np$load(path_predicted_GA_exp1_DLM1))
predicted_GA_exp2_DLM1 = as.numeric(np$load(path_predicted_GA_exp2_DLM1))
predicted_GA_exp3_DLM1 = as.numeric(np$load(path_predicted_GA_exp3_DLM1))
predicted_GA_exp4_DLM1 = as.numeric(np$load(path_predicted_GA_exp4_DLM1))
predicted_GA_exp5_DLM1 = as.numeric(np$load(path_predicted_GA_exp5_DLM1))

true_GA_exp1_DLM1 = as.numeric(np$load(path_true_GA_exp1_DLM1))
true_GA_exp2_DLM1 = as.numeric(np$load(path_true_GA_exp2_DLM1))
true_GA_exp3_DLM1 = as.numeric(np$load(path_true_GA_exp3_DLM1))
true_GA_exp4_DLM1 = as.numeric(np$load(path_true_GA_exp4_DLM1))
true_GA_exp5_DLM1 = as.numeric(np$load(path_true_GA_exp5_DLM1))

predicted_GA_exp1_DLM2 = as.numeric(np$load(path_predicted_GA_exp1_DLM2))
predicted_GA_exp2_DLM2 = as.numeric(np$load(path_predicted_GA_exp2_DLM2))
predicted_GA_exp3_DLM2 = as.numeric(np$load(path_predicted_GA_exp3_DLM2))
predicted_GA_exp4_DLM2 = as.numeric(np$load(path_predicted_GA_exp4_DLM2))
predicted_GA_exp5_DLM2 = as.numeric(np$load(path_predicted_GA_exp5_DLM2))

true_GA_exp1_DLM2 = as.numeric(np$load(path_true_GA_exp1_DLM2))
true_GA_exp2_DLM2 = as.numeric(np$load(path_true_GA_exp2_DLM2))
true_GA_exp3_DLM2 = as.numeric(np$load(path_true_GA_exp3_DLM2))
true_GA_exp4_DLM2 = as.numeric(np$load(path_true_GA_exp4_DLM2))
true_GA_exp5_DLM2 = as.numeric(np$load(path_true_GA_exp5_DLM2))

# Compute the mean of the true and predicted GA by rows
dataPredictedGA_DLM1 = cbind(predicted_GA_exp1_DLM1, predicted_GA_exp2_DLM1, predicted_GA_exp3_DLM1, predicted_GA_exp4_DLM1, predicted_GA_exp5_DLM1) 
dataTrueGA_DLM1 = cbind(true_GA_exp1_DLM1, true_GA_exp2_DLM1, true_GA_exp3_DLM1, true_GA_exp4_DLM1, true_GA_exp5_DLM1)
meanPredictedGA_DLM1 = rowMeans(dataPredictedGA_DLM1)
meanTrueGA_DLM1 = rowMeans(dataTrueGA_DLM1)

dataPredictedGA_DLM2 = cbind(predicted_GA_exp1_DLM2, predicted_GA_exp2_DLM2, predicted_GA_exp3_DLM2, predicted_GA_exp4_DLM2, predicted_GA_exp5_DLM2) 
dataTrueGA_DLM2 = cbind(true_GA_exp1_DLM2, true_GA_exp2_DLM2, true_GA_exp3_DLM2, true_GA_exp4_DLM2, true_GA_exp5_DLM2)
meanPredictedGA_DLM2 = rowMeans(dataPredictedGA_DLM2)
meanTrueGA_DLM2 = rowMeans(dataTrueGA_DLM2)

# Compute the error
error_DLM1 = meanTrueGA_DLM1 - meanPredictedGA_DLM1
error_DLM2 = meanTrueGA_DLM2 - meanPredictedGA_DLM2

# Compute the wilxocon test
print(wilcox.test(as.numeric(error_DLM1), as.numeric(error_DLM2), exact = T, paired = T, alternative = "two.side"))