#!/bin/sh
#SBATCH -n 1 -N 1

echo [$SECONDS] setting up environment

export PATH="/cosmo/software/anaconda3/bin:$PATH"
export PYTHONPATH=/cosmo/software/anaconda3/lib/python3.6/site-packages/:$HOME/.conda/envs/nes_keras/lib/python3.6/site-packages/

python Cl_denoiseVAE.py
python Cl_VAEpredict.py

echo [$SECONDS] job completed
