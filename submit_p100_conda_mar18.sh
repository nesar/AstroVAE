#!/bin/sh
#SBATCH -p cp100

echo [$SECONDS] setting up environment
#export KERAS_BACKEND=tensorflow

srun -p cp100 python Cl_denoiseP4_mar18.py
srun -p cp100 python Cl_VAEpredictP4_mar18.py

echo [$SECONDS] End job 


