library(kdensity)
library(reticulate)
np = import("numpy")

# Get paths for each files
path_preedicted_GA_CHD = '/home/axel/dev/fetus_GA_prediction/data/output/deepEnsembleResNet_slices2D_wholeCohort/exp5/GA_prediction.npy'
path_true_GA_CHD =  '/home/axel/dev/fetus_GA_prediction/data/output/deepEnsembleResNet_slices2D_wholeCohort/exp5/GA.npy'

path_preedicted_GA = '/home/axel/dev/fetus_GA_prediction/data/output/deepEnsembleResNet_slices2D/exp5/GA_prediction.npy'
path_true_GA =  '/home/axel/dev/fetus_GA_prediction/data/output/deepEnsembleResNet_slices2D/exp5/GA.npy'

# Open the files
predicted_GA_CHD = as.numeric(np$load(path_preedicted_GA_CHD))
true_GA_CHD = as.numeric(np$load(path_true_GA_CHD))

predicted_GA = as.numeric(np$load(path_preedicted_GA))
true_GA = as.numeric(np$load(path_true_GA))

# Compute the absolute error and error from true and preedicted GA
abs_error_CHD = abs(true_GA_CHD - predicted_GA_CHD)
error_CHD = (true_GA_CHD - predicted_GA_CHD)

#compute endpoints

MAE_CHD = mean(abs_error_CHD)
ME_CHD = mean(error_CHD)

SDAE_CHD = sd(abs_error_CHD)
SDE_CHD = sd(error_CHD)

correlation_CHD = cor(true_GA_CHD, predicted_GA_CHD)

#Display
print("MAE")
print(MAE_CHD)

print("ME")
print(ME_CHD)

print("SDAE")
print(SDAE_CHD)

print("SDE")
print(SDE_CHD)

print("correlation")
print(correlation_CHD)


fit = kdensity(true_GA_CHD, bw = 1, kernel = "gaussian")

weights = fit(true_GA)

index_resampled = sample(1:length(true_GA), length(true_GA_CHD), replace =FALSE, prob = weights)

tiff("/home/axel/dev/fetus_GA_prediction/source/evaluation/hist_healthy_with_CHD_distribution.tiff")
hist(true_GA[index_resampled], freq = FALSE)
dev.off()

tiff("/home/axel/dev/fetus_GA_prediction/source/evaluation/hist_CHD.tiff")
hist(true_GA_CHD, freq = FALSE)
lines(seq(min((true_GA_CHD)), max(true_GA_CHD), length = 1000), fit(seq(min((true_GA_CHD)), max(true_GA_CHD), length = 1000)), col = 2)
dev.off()

print("############# Endpoint values of the model with healthy cohort GA following the same distribution than CHD cohort GA ###########")
# Compute the absolute error and error from true and preedicted GA
abs_error = abs(true_GA[index_resampled] - predicted_GA[index_resampled])
error = (true_GA[index_resampled] - predicted_GA[index_resampled])

#compute endpoints

MAE = mean(abs_error)
ME = mean(error)

SDAE = sd(abs_error)
SDE = sd(error)

correlation_CHD = cor(true_GA, predicted_GA)

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
print(correlation_CHD)

