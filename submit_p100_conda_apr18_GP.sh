#!/bin/sh
#SBATCH -p cp100

echo [$SECONDS] setting up environment
#export KERAS_BACKEND=tensorflow


#srun -p cp100 python Cl_linear.py
#srun -p cp100 python Cl_VAEpredictGPy.py
#srun -p cp100 python Cl_extendedP10.py 
#srun -p cp100 python Cl_predictP10GPy.py 
#srun -p cp100 python Cl_denoiseP4_apr18.py
srun -p cp100 python Cl_VAEpredictP4_apr18.py

echo [$SECONDS] End job 


