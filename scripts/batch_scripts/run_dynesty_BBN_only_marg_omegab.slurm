#!/bin/bash

#SBATCH -N 1   # node count
#SBATCH --ntasks-per-node=36
#SBATCH --job-name=BBN_only_marg_omegab
#SBATCH -t 00:30:00
#SBATCH --mem=50GB
#SBATCH --mail-user=###
#SBATCH --mail-type=all

module purge
module load anaconda3/2020.07
module load gsl/intel/2.6 
# depending on cluster configuration, openmpi may
# need to be loaded for multiprocess
module load openmpi/intel/4.0.5

# ensure JAX runs only on CPU and does not look for a GPU
export JAX_PLATFORMS="cpu"

# NYU Greene cluster requires the use of singularity containers
# clik installation must also be sourced (modify filepath for your installation)
singularity exec --nv \
	    --overlay /scratch/cg3566/python_class/python_class.ext3:ro \
	    /scratch/work/public/singularity/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif\
	    /bin/bash -c "source /ext3/env.sh; \
        source /scratch/cg3566/jaxBBNsampling/planck/code/plc_3.0/plc-3.01/bin/clik_profile.sh; \
        python BBN_only_marg_omegab.py --sampler dynesty --method static --network key_PRIMAT_2023 --save_name BBN_only_marg_omegab --n_cpus 36 --n_live 1000 --dlogz 0.5"
