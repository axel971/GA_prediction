#!/bin/bash

module load singularity

echo "deep Ensemble learning ResNet with Attention"

singularity exec --nv -w ../../../ubuntu_container/ python3 ../examples/main_cross_validation_display_attentionMap.py


