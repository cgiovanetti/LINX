#!/bin/bash

#SBATCH --nodes=4   # be sure to match n_cpus argument below to this request   
#SBATCH --ntasks-per-node=48
#SBATCH --cpus-per-task=1  
#SBATCH --job-name=CMB_BBN_marg_nuisance_omegab_Neff_multinode
#SBATCH -t 72:00:00
#SBATCH --mem=90GB
#SBATCH --mail-user=XXX
#SBATCH --mail-type=all

export JAX_PLATFORMS="cpu"

# extra setup was required to use schwimmbad on NYU's Greene cluster, hence the call to
# an additional bash script
srun /scratch/cg3566/jaxBBNsampling/run-schwimmbad_clik.bash python -u CMB_BBN_marg_nuisance_omegab_Neff.py --sampler dynesty --method static --network key_PRIMAT_2023 --save_name CMB_BBN_marg_nuisance_omegab_Neff --n_cpus 192 --n_live 500 --dlogz 0.5
