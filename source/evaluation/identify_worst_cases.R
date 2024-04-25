library("xlsx")
library(reticulate)
np = import("numpy")

# Get paths for each files
path_preedicted_GA = '/home/axel/dev/fetus_GA_prediction/data/output/deepEnsembleResNet_slices2D/exp1/GA_prediction.npy'
path_true_GA =  '/home/axel/dev/fetus_GA_prediction/data/output/deepEnsembleResNet_slices2D/exp1/GA.npy'
path_uncertainty =  '/home/axel/dev/fetus_GA_prediction/data/output/deepEnsembleResNet_slices2D/exp1/uncertainty.npy'
path_subject_list = '/home/axel/dev/fetus_GA_prediction/data/subjectList.xlsx'

# Open the files
predicted_GA = as.numeric(np$load(path_preedicted_GA))
uncertainty = as.numeric(np$load(path_uncertainty))
true_GA = as.numeric(np$load(path_true_GA))
subject_list = read.xlsx(path_subject_list, 1, header = TRUE)

# Compute the error prediction
error = abs(predicted_GA - true_GA)

print("subjects with large error")
print(subject_list[error > 3. , 1])
print("error > 3")
print(error[error >  3.])
print("true GA and predicted GA for subject with large error")
print(true_GA[error >  3.])
print(predicted_GA[error > 3.])
print("Subject with large uncertainty")
print(subject_list[uncertainty > 3. & error < 3., 1])