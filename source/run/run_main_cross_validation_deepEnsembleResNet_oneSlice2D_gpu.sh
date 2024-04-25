#!/bin/bash

module load singularity

echo "deep Ensemble ResNet one slice 2D"

singularity exec --nv -w ../../../ubuntu_container/ python3 ../examples/main_cross_validation_deepEnsembleResNet_oneSlice2D.py


