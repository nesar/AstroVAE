#!/bin/sh
#SBATCH -N 16

echo [$SECONDS] setting up environment
#export KERAS_BACKEND=tensorflow

srun -N 16 python Cl_extendedP10.py 

echo [$SECONDS] End job 


