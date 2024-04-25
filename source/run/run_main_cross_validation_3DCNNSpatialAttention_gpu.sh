#!/bin/bash

module load singularity

echo "3D CNN spatial attention"

singularity exec --nv -w ../../../ubuntu_container/ python3 ../examples/main_cross_validation_3DCNNSpatialAttention.py


