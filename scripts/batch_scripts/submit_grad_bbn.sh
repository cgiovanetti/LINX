#!/bin/bash

#SBATCH --job-name=train
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --time=48:00:00
#SBATCH --account=iaifi_lab

export TF_CPP_MIN_LOG_LEVEL="2"

# Activate env
mamba activate linx2

# Go to dir and train
cd /n/holystore01/LABS/iaifi_lab/Users/smsharma/constraints_from_GPs/LINX/scripts
python -u run_svi.py --bbn_only=False --n_steps_svi=1000 --n_warmup_mcmc=500 --n_samples_mcmc=1200 --n_chains=2