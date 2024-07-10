import os

import multiprocessing
from multiprocessing import Pool
from schwimmbad import MPIPool
import argparse

import numpy as np
from dynesty import DynamicNestedSampler, NestedSampler
from dynesty.utils import resample_equal
import clik
from classy import Class

from scipy.special import ndtri
from scipy.stats import truncnorm

import sys
file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(file_dir+'/..')

from linx.nuclear import NuclearRates
from linx.abundances import AbundanceModel
from linx.background import BackgroundModel
import linx.const as const

import jax
import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True) # need this to enable float64

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--n_cpus", action="store", default=4, type=int)  # Number of CPUs to parallelize over
    parser.add_argument("--n_live", action="store", default=500, type=int)  # Number of live points
    parser.add_argument("--dlogz", action="store", default=0.1, type=float)  # Convergence criterion; fraction of evidence estimated left
    parser.add_argument("--sampler", action="store", default="dynesty", type=str)  # Which sampler to use; only dynesty so far
    parser.add_argument("--method", action="store", default="static", type=str)  # Which dynesty method to use; `static` or `dynamic`
    parser.add_argument("--save_dir", action="store", default="./samples/", type=str)  # Where to save samples
    parser.add_argument("--save_name", action="store", default="test", type=str)  # What to name the saved samples
    parser.add_argument("--network", action="store", default="key_PRIMAT_2022", type=str) # Which network to use

    return parser.parse_args()
args = parse_args()

thermo_model_DNeff = BackgroundModel()
GetAbundances = AbundanceModel(NuclearRates(nuclear_net=args.network))

# no need to warm up if using schwimmbad

# set up clik
plik = clik.clik('/scratch/cg3566/jaxBBNsampling/planck/baseline/plc_3.0/hi_l/plik/plik_rd12_HM_v22b_TTTEEE.clik')
commander = clik.clik('/scratch/cg3566/jaxBBNsampling/planck/baseline/plc_3.0/low_l/commander/commander_dx12_v3_2_29.clik')
simall = clik.clik('/scratch/cg3566/jaxBBNsampling/planck/baseline/plc_3.0/low_l/simall/simall_100x143_offlike5_EE_Aplanck_B.clik')

# how many parameters are you sampling?
n_CMB = 8
num_reactions_used = len(GetAbundances.nuclear_net.reactions)
n_nuisance_varying = 21 
n_dim = n_CMB + n_nuisance_varying + num_reactions_used

def lnlike(theta, n_dim):
    """ n-d Gaussian Likelihood function
        `theta` is the parameter array, and any other args can be specified through `logl_args` in the sampler.
    """

    omega_b = theta[3]*1e-2
    omb_to_eta = const.Omegabh2_to_eta0 
    eta = omega_b*omb_to_eta
    eta_fac = eta/(const.Omegabh2*omb_to_eta) # eta = eta_fac * eta0

    t_vec_ref, a_vec_ref, rho_g_vec, rho_nu_vec, rho_NP_vec, P_NP_vec, Neff_vec = thermo_model_DNeff(theta[6])
    N_eff = Neff_vec[-1]

    Yn, Yp, Yd, Yt, YHe3, Ya, YLi7, YBe7 = GetAbundances(
        jnp.array(rho_g_vec), 
        jnp.array(rho_nu_vec), 
        jnp.array(rho_NP_vec), 
        jnp.array(P_NP_vec),
        t_vec=jnp.array(t_vec_ref), 
        a_vec=jnp.array(a_vec_ref), 
        eta_fac = jnp.asarray(eta_fac), 
        tau_n_fac = jnp.asarray(theta[7]), 
        nuclear_rates_q = jnp.asarray(theta[n_CMB + n_nuisance_varying:])
    )

    YHe = 4 * Ya
    YHe_BBN = 0.2449 # Aver 2015
    YHe_BBN_sigma = 0.004

    DH = Yd/Yp * 1e5 # rescale
    DH_BBN = 2.527 # Cooke 2018
    DH_BBN_sigma = 0.030 

    ll_BBN = -.5*(((YHe-YHe_BBN)/YHe_BBN_sigma)**2 + ((DH-DH_BBN)/DH_BBN_sigma)**2)

    # Now, do the CMB side of things
    # call class with the parameters specified in theta
    params = {
        'YHe': YHe, # set YHe to input from LINX
        'output': 'tCl pCl lCl',
        'l_max_scalars': max(plik.get_lmax()), # make sure you get lmax from the hi_l likelihood
        'lensing': 'yes',
        'N_ncdm': 1,
        'N_ur' : N_eff - 1.0132, # see explanatory.ini
        'T_ncdm' : 0.71611,
        'm_ncdm' : 0.06,
        'A_s': theta[0]*1e-9,
        'tau_reio': theta[5]*1e-2, 
        'n_s': theta[1], 
        'h': theta[2], 
        'omega_b': theta[3]*1e-2, 
        'omega_cdm': theta[4]} 

    # these are the nuisance parameters that must stay at their central values (see recommended_priors)
    cib_index =  -1.3
    EE_dust_levels = [0.055, 0.040, 0.094, 0.086,  0.21, 0.70, -2.4]
    galf_TE_index = -2.4
    A_cnoise_factors = [1, 1, 1, 1]
    A_sbpx_factors = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    p_calib_factors = [1.021, 0.966, 1.040]
    A_pol = 1
    # combine with the nuisance parameters that get varied
    nuisance_params = np.concatenate(([theta[n_CMB]],[cib_index],theta[n_CMB+1:n_CMB+1+11],EE_dust_levels,theta[n_CMB+12:n_CMB+12+6], \
        [galf_TE_index], A_cnoise_factors, A_sbpx_factors,theta[n_CMB+12+6:n_CMB+12+6+2],p_calib_factors,[A_pol],[theta[n_CMB+12+6+2]]))

    try:
        cosmo = Class()
        cosmo.set(params)
        cosmo.compute()
        class_clstt = cosmo.lensed_cl(max(plik.get_lmax()))['tt'] # raw_cl would require processing to compare to clik. clik gives lensed cls since you measure lensed cls.
        class_clste = cosmo.lensed_cl(max(plik.get_lmax()))['te']
        class_clsee = cosmo.lensed_cl(max(plik.get_lmax()))['ee']

        class_cls_lowltt = cosmo.lensed_cl(max(commander.get_lmax()))['tt']
        class_cls_lowlee = cosmo.lensed_cl(max(simall.get_lmax()))['ee']

        # feed the class cls into clik and get out a log likelihood
        # need to scale outputs by T_CMB !!

        # order must be TT EE BB TE TB EB (nothing for the ones for which there is no data)
        # nuisance parameters go at the end
        ll_plik = plik(np.concatenate((class_clstt*(2.7255e6)**2,class_clsee*(2.7255e6)**2,class_clste*(2.7255e6)**2,nuisance_params)))[0] 
        ll_commander = commander(np.concatenate((class_cls_lowltt*(2.7255e6)**2,[theta[n_CMB + n_nuisance_varying - 1]])))[0]
        ll_simall = simall(np.concatenate((class_cls_lowlee*(2.7255e6)**2,[theta[n_CMB + n_nuisance_varying - 1]])))[0] # last nuisance parameter is A_planck
    except:
        ll_plik = -1e30
        ll_simall = -1e30
        ll_commander = -1e30
    return ll_BBN + ll_plik + ll_commander + ll_simall 

def prior_cube(cube):
    """ Prior cube, specifying a bijection between the unit cube and the prior volume. Assumed uniform.
    """


    # uniform priors on CMB parameters
    # note there is some scaling on input A_s, Omega_b
    theta_min = np.array([1, .85, .5, 1.9, 0.09, 2.5, -7.0]) # lower bound of prior
    theta_max = np.array([3, 1., .8, 2.5, 0.15, 9, 7.0])  # Upper bound of prior
    
    cube[:n_CMB-1] = cube[:n_CMB-1] * (theta_max - theta_min) + theta_min

    # gaussian prior on tau_n_fac
    cube[n_CMB-1] = 1. + 0.000682 * ndtri(cube[n_CMB-1])

    # order: 'A_cib_217' : 47.2,
            # 'xi_sz_cib' : 0.42,
            # 'A_sz' : 7.23,
            # 'ps_A_100_100': 251.0,
            # 'ps_A_143_143' : 47.4, 
            # 'ps_A_143_217' : 47.3, 
            # 'ps_A_217_217' : 119.8,
            # 'ksz_norm' : 0.01,
    # central values also included above for reference
    # note recommended_priors does not give a value for 'ps_A_143_217', but it does give
    # a prior on r_cs_143_217 of [0,1], so we give the same prior range as 'ps_A_143_143'
    theta_nuisance_uniform_min = np.array([1e-10,1e-10,1e-10,1e-10,1e-10,1e-10,1e-10,1e-10])
    theta_nuisance_uniform_max = np.array([80,1,10,360,270,270,450,10])

    cube[n_CMB:n_CMB + len(theta_nuisance_uniform_max)] = cube[n_CMB:n_CMB + len(theta_nuisance_uniform_max)] * \
                (theta_nuisance_uniform_max - theta_nuisance_uniform_min) + theta_nuisance_uniform_min

    # From plik_recommended_priors.txt
    nuisance_central_values = {

                        # from recommended priors
                        'gal545_A_100' : 8.6, 
                        'gal545_A_143': 10.6, 
                        'gal545_A_143_217': 23.5, 
                        'gal545_A_217': 91.9,

                        # from recommended priors
                        'galf_TE_A_100' : 0.13, 
                        'galf_TE_A_100_143':  0.13, 
                        'galf_TE_A_100_217': 0.46, 
                        'galf_TE_A_143': 0.207, 
                        'galf_TE_A_143_217' : 0.69, 
                        'galf_TE_A_217': 1.938, 

                        # from recommended priors
                        'calib_100T' : 1.0002,
                        'calib_217T' : 0.99805, 

                        # from the wiki
                        'A_planck' : 1
    }

    nuisance_sigmas = {          
                        'gal545_A_100' : 2, 
                        'gal545_A_143': 2, 
                        'gal545_A_143_217': 8.5, 
                        'gal545_A_217': 20,

                        'galf_TE_A_100' : 0.042, 
                        'galf_TE_A_100_143': 0.036, 
                        'galf_TE_A_100_217': 0.09, 
                        'galf_TE_A_143': 0.072, 
                        'galf_TE_A_143_217' : 0.09, 
                        'galf_TE_A_217': 0.54, 

                        'calib_100T' : 0.0007, 
                        'calib_217T' : 0.00065, 
                        
                        'A_planck' : 0.0025
    }

    nuisance_upper = {
                        
                        'gal545_A_100' : 50., 
                        'gal545_A_143': 50., 
                        'gal545_A_143_217': 100., 
                        'gal545_A_217': 400.,

                        'galf_TE_A_100' : 10., 
                        'galf_TE_A_100_143': 10., 
                        'galf_TE_A_100_217': 10., 
                        'galf_TE_A_143': 10., 
                        'galf_TE_A_143_217' : 10., 
                        'galf_TE_A_217': 10., 

                        'calib_100T' : 1.02, 
                        'calib_217T' : 1.05, 
                        
                        'A_planck' : 1.1
    }

    nuisance_lower = {
                        
                        'gal545_A_100' : 1e-10, 
                        'gal545_A_143': 1e-10, 
                        'gal545_A_143_217': 1e-10, 
                        'gal545_A_217': 1e-10,

                        'galf_TE_A_100' : 1e-10, 
                        'galf_TE_A_100_143': 1e-10, 
                        'galf_TE_A_100_217': 1e-10, 
                        'galf_TE_A_143': 1e-10, 
                        'galf_TE_A_143_217' : 1e-10, 
                        'galf_TE_A_217': 1e-10, 

                        'calib_100T' : .98, 
                        'calib_217T' : .95, 
                        
                        'A_planck' : 0.9
    }
    
    # # Gaussian priors for rate uncertainties, mean 0 and sigma 1 for each
    mu_Apl = np.array(list(nuisance_central_values.values()))
    sigma_Apl = np.array(list(nuisance_sigmas.values()))

    # Truncate bounds to avoid breaking anything in combination with CLASS
    a = np.array(list(nuisance_lower.values()))  # Lower bound of truncation
    b = np.array(list(nuisance_upper.values()))  # Upper bound of truncation

    # Convert real-world bounds to Z-scores relative to the non-truncated distribution
    # (This is just how the truncnorm function likes its arguments)
    a_std = (a - mu_Apl) / sigma_Apl
    b_std = (b - mu_Apl) / sigma_Apl

    # Generalize for truncated normal distribution
    cube[n_CMB + len(theta_nuisance_uniform_max):n_CMB + n_nuisance_varying] = truncnorm.ppf(cube[n_CMB + len(theta_nuisance_uniform_max):n_CMB + n_nuisance_varying], a_std, b_std, loc=mu_Apl, scale=sigma_Apl)

    mu_Apl = jnp.zeros(num_reactions_used)
    sigma_Apl = jnp.ones(num_reactions_used)

    cube[n_CMB + n_nuisance_varying:] = mu_Apl + sigma_Apl * ndtri(cube[n_CMB + n_nuisance_varying:])

    return cube

def run_dynesty(lnlike, prior_cube, nlive=500, dlogz=0.5, n_cpus=36, method="dynamic"):

    with MPIPool() as pool: # number of processes is now specified outside the script
        # safety
        if not pool.is_master():
            pool.wait()
            sys.exit(0)

        if method == "dynamic":
            # https://dynesty.readthedocs.io/en/stable/quickstart.html#checkpointing
            checkpoint_file = str(args.network) + "_marg_nuisance.save"
            sampler = DynamicNestedSampler(lnlike, prior_cube, n_dim, pool=pool, queue_size=n_cpus, logl_args=(n_dim,))
            sampler.run_nested(dlogz_init=dlogz, nlive_init=nlive, nlive_batch=nlive,checkpoint_file=checkpoint_file)

            # to resume from checkpoint
            # sampler = DynamicNestedSampler.restore('nuisance.save', pool = pool)
            # sampler.run_nested(resume=True)
        elif method == "static":
            checkpoint_file = str(args.network) + "_marg_nuisance.save"
            sampler = NestedSampler(lnlike, prior_cube, n_dim, nlive=nlive, pool=pool, queue_size=n_cpus, logl_args=(n_dim,))
            sampler.run_nested(dlogz=dlogz,checkpoint_file=checkpoint_file)

            # sampler = NestedSampler.restore(checkpoint_file, pool = pool)
            # new_checkpoint = str(args.network) + "_marg_nuisance_alt1.save"
            # sampler.run_nested(resume=True,checkpoint_file=new_checkpoint)
        else:
            raise NotImplementedError

    # Draw posterior samples
    weights = np.exp(sampler.results["logwt"] - sampler.results["logz"][-1])
    samples_equal_weights = resample_equal(sampler.results.samples, weights)

    return samples_equal_weights, sampler

if __name__ == "__main__":

    args = parse_args()

    save_dir = args.save_dir
    # Create save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Sample using chosen sampler

    if args.sampler == "dynesty":

        samples, sampler = run_dynesty(lnlike, prior_cube, args.n_live, args.dlogz, args.n_cpus, args.method)

        # Save various sampling results
        logl_ary = sampler.results['logl']
        logz_ary = sampler.results['logz']
        logzerr_ary = sampler.results['logzerr']
        dkl_ary = sampler.results['information']

        results_dict = {"samples":samples, "logl":logl_ary, "logz":logz_ary,  "logzerr":logzerr_ary, "dkl":dkl_ary}

    else:
        raise NotImplementedError("Sampler {} not implemented.".format(args.sampler))
    
    # Print mean of samples to make sure we get np.arange(n_dim) as we've defined the test likelihood
    print("Mean of samples: {}".format(np.mean(samples, axis=0)))

    # Save samples
    np.savez(f"{save_dir}/{args.sampler}_{args.network}_{args.save_name}_samples.npz", **results_dict)
