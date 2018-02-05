#!/bin/sh
#SBATCH -p cp100

echo [$SECONDS] setting up environment

export PATH="/cosmo/software/anaconda3/bin:$PATH"
export PYTHONPATH=/cosmo/software/anaconda3/lib/python3.6/site-packages/:$HOME/.conda/envs/nes_keras/lib/python3.6/site-packages/

mkdir slurm-$SLURM_JOBID
cd slurm-$SLURM_JOBID

#cp ../classification2label.py ./
#cp ../ConvNetLensJPG_noAugment.py ./
 
srun -p cp100 python Cl_VAEpredict.py
#srun -p cp100 python ConvNetLensJPG_noAugment.py

echo [$SECONDS] job completed

