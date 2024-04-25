library(reticulate)
np = import("numpy")

# Get paths for each files
path_predicted_GA_exp1 = '/home/axel/dev/fetus_GA_prediction/data/output/2D_with_one_slice/exp1/GA_prediction.npy'
path_predicted_GA_exp2 =  '/home/axel/dev/fetus_GA_prediction/data/output/2D_with_one_slice/exp2/GA_prediction.npy'
path_predicted_GA_exp3 = '/home/axel/dev/fetus_GA_prediction/data/output/2D_with_one_slice/exp3/GA_prediction.npy'
path_predicted_GA_exp4 = '/home/axel/dev/fetus_GA_prediction/data/output/2D_with_one_slice/exp4/GA_prediction.npy'
path_predicted_GA_exp5 =  '/home/axel/dev/fetus_GA_prediction/data/output/2D_with_one_slice/exp5/GA_prediction.npy'

path_true_GA_exp1 = '/home/axel/dev/fetus_GA_prediction/data/output/2D_with_one_slice/exp1/GA.npy'
path_true_GA_exp2 =  '/home/axel/dev/fetus_GA_prediction/data/output/2D_with_one_slice/exp2/GA.npy'
path_true_GA_exp3 = '/home/axel/dev/fetus_GA_prediction/data/output/2D_with_one_slice/exp3/GA.npy'
path_true_GA_exp4 = '/home/axel/dev/fetus_GA_prediction/data/output/2D_with_one_slice/exp4/GA.npy'
path_true_GA_exp5 =  '/home/axel/dev/fetus_GA_prediction/data/output/2D_with_one_slice/exp5/GA.npy'

#Load the python files
predicted_GA_exp1 = as.numeric(np$load(path_predicted_GA_exp1))
predicted_GA_exp2 = as.numeric(np$load(path_predicted_GA_exp2))
predicted_GA_exp3 = as.numeric(np$load(path_predicted_GA_exp3))
predicted_GA_exp4 = as.numeric(np$load(path_predicted_GA_exp4))
predicted_GA_exp5 = as.numeric(np$load(path_predicted_GA_exp5))

true_GA_exp1 = as.numeric(np$load(path_true_GA_exp1))
true_GA_exp2 = as.numeric(np$load(path_true_GA_exp2))
true_GA_exp3 = as.numeric(np$load(path_true_GA_exp3))
true_GA_exp4 = as.numeric(np$load(path_true_GA_exp4))
true_GA_exp5 = as.numeric(np$load(path_true_GA_exp5))

# Computeerror
error_exp1 = true_GA_exp1 - predicted_GA_exp1
error_exp2 = true_GA_exp2 - predicted_GA_exp2
error_exp3 = true_GA_exp3 - predicted_GA_exp3
error_exp4 = true_GA_exp4 - predicted_GA_exp4
error_exp5 = true_GA_exp5 - predicted_GA_exp5



# Combine the data in one matrix
data = cbind(error_exp1, error_exp2, error_exp3, error_exp4, error_exp5)

# Compute the friedman test
print(friedman.test(data), exact = T, paired = T)
