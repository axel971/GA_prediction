#!/bin/bash

module load singularity

echo "deep Ensemble learning ResNet slices2D (Whole cohort training)"

singularity exec --nv -w ../../../ubuntu_container/ python3 ../examples/main_cross_validation_testingWholeCohort_deepEnsembleResNet_slices2D.py


