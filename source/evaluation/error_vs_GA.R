
library(reticulate)
np = import("numpy")

# Get paths for each files
path_preedicted_GA = '/home/axel/dev/fetus_GA_prediction/data/output/3DCNN/exp1/GA_prediction.npy'
path_true_GA = '/home/axel/dev/fetus_GA_prediction/data/output/3DCNN/exp1/GA.npy'


# Open the files
predicted_GA = as.numeric(np$load(path_preedicted_GA))
true_GA = as.numeric(np$load(path_true_GA))

# Compute the error prediction
error =  true_GA - predicted_GA

tiff("/home/axel/dev/fetus_GA_prediction/data/output/3DCNN/exp1/figure/GA_vs_error_deepEnsembleResNet_slices2D_ME_exp1.tiff")
plot(true_GA, error)
dev.off()
