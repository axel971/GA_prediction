
library(reticulate)
np = import("numpy")

# Get paths for each files
path_true_GA = '/home/axel/dev/fetus_GA_prediction/data/output/deepEnsembleResNet_slices2D/exp1/GA.npy'

# Open the files
true_GA = as.numeric(np$load(path_true_GA))



tiff("/home/axel/dev/fetus_GA_prediction/source/evaluation/GA_histogram_healthy_cohort.tiff")
hist(true_GA, breaks = seq(15,40,1))
dev.off()
