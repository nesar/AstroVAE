#!/bin/sh
#SBATCH -p cp100

echo [$SECONDS] setting up environment
#export KERAS_BACKEND=tensorflow


#srun -p cp100 python Cl_linear.py
srun -p cp100 python Cl_extendedP10.py 
#srun -p cp100 python Cl_predictP10GPy.py 

echo [$SECONDS] End job 


