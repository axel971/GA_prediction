#!/bin/bash

module load singularity
# sbatch -p gpua --gres=gpu run_main_cross_validation_3DCNN_gpu.sh 
echo "3D CNN"

singularity exec --nv -w ../../../ubuntu_container/ python3 ../examples/main_cross_validation_3DCNN.py


