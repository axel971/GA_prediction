
library(reticulate)
np = import("numpy")

# Get paths for each files
path_preedicted_GA = '/home/axel/dev/fetus_GA_prediction/data/output/deepEnsembleResNet_slices2D/exp5/GA_prediction.npy'
path_true_GA =  '/home/axel/dev/fetus_GA_prediction/data/output/deepEnsembleResNet_slices2D/exp5/GA.npy'
path_uncertainty =  '/home/axel/dev/fetus_GA_prediction/data/output/deepEnsembleResNet_slices2D/exp5/uncertainty.npy'



# Open the files
predicted_GA = as.numeric(np$load(path_preedicted_GA))
true_GA = as.numeric(np$load(path_true_GA))
uncertainty = as.numeric(np$load(path_uncertainty))

# Compute the error prediction
error = abs(predicted_GA - true_GA)


# Compute Kolmogorv-Smirnov test to test the normality of the data
#print(shapiro.test(uncertainty))

# Compute the corellation test
#print(cor.test(error, uncertainty, method= "pearson"))

print(mean(uncertainty))

tiff("/home/axel/dev/fetus_GA_prediction/data/output/deepEnsembleResNet_slices2D/exp5/figure/error_vs_uncertainty_deepEnsembleResNet_slices2D_exp5.tiff")
plot( error, uncertainty)
dev.off()
