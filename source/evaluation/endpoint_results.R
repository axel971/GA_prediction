
library(reticulate)
np = import("numpy")

# Get paths for each files
path_preedicted_GA = '/home/axel/dev/fetus_GA_prediction/data/output/deepEnsembleResNet_slices2D/exp1/GA_prediction.npy'
path_true_GA =  '/home/axel/dev/fetus_GA_prediction/data/output/deepEnsembleResNet_slices2D/exp1/GA.npy'


# Open the files
predicted_GA = as.numeric(np$load(path_preedicted_GA))
true_GA = as.numeric(np$load(path_true_GA))

# Compute the absolute error and error from true and preedicted GA
abs_error = abs(true_GA - predicted_GA)
error = (true_GA - predicted_GA)

#compute endpoints

MAE = mean(abs_error)
ME = mean(error)

SDAE = sd(abs_error)
SDE = sd(error)

correlation = cor(true_GA, predicted_GA)

#Display
print("MAE")
print(MAE)

print("ME")
print(ME)

print("SDAE")
print(SDAE)

print("SDE")
print(SDE)

print("correlation")
print(correlation)

#print("Min GA")
#print(min(true_GA))

#print("Max GA")
#print(max(true_GA))

#print("Mean GA")
#print(mean(true_GA))

#print("SD GA")
#print(sd(true_GA))

