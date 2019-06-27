#!/bin/sh
#SBATCH -N 1

echo [$SECONDS] setting up environment

export KERAS_BACKEND=tensorflow

echo [$SLURM_JOBID]

source /cosmo/spack/share/spack/setup-env.sh

spack load cuda@10.0.130 
spack load cudnn@6.0
spack load mpich@3.2.1 

#export PATH="/homes/nramachandra/miniconda2/bin:$PATH"
echo ". /homes/nramachandra/miniconda2/etc/profile.d/conda.sh" >> ~/.bashrc

conda activate mlEnv

#python Cl_CVAE.py 

echo [$SECONDS] End job 

