import os

import multiprocessing
from multiprocessing import Pool
import argparse

import numpy as np
from dynesty import DynamicNestedSampler, NestedSampler
from dynesty.utils import resample_equal

from scipy.special import ndtri

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
    parser.add_argument("--network", action="store", default="key_PRIMAT_2023", type=str) # Which network to use

    return parser.parse_args()

args = parse_args()


GetAbundances = AbundanceModel(NuclearRates(nuclear_net=args.network))

# warm-up call--need this with multiprocess to make sure it uses the compiled 
# code for the scans
def warmup_process(q):
    # all floats must be wrapped in jnp.asarray()
    Yn, Yp, Yd, Yt, YHe3, Ya, YLi7, YBe7 = GetAbundances(
                            jnp.array(rho_g_vec), 
                            jnp.array(rho_nu_vec), 
                            jnp.array(rho_NP_vec), 
                            jnp.array(P_NP_vec), 
                            t_vec=jnp.array(t_vec_ref), 
                            a_vec=jnp.array(a_vec_ref), 
                            eta_fac = jnp.asarray(0.5), 
                            tau_n_fac = jnp.asarray(0.3)
                            )

# warm up on each core
if __name__ == "__main__":
    num_cores = 36
    multiprocessing.set_start_method('spawn')
    q = multiprocessing.Queue()
    processes = [multiprocessing.Process(target=warmup_process,args=(q,)) for _ in range(num_cores)]

    for p in processes:
        p.start()
    for p in processes:
        p.join()
        print("warmed up on {0}".format(p))

# how many parameters are you sampling?
n_CMB = 2
num_reactions_used = len(GetAbundances.nuclear_net.reactions)
n_dim = n_CMB + num_reactions_used

thermo_model_DNeff = BackgroundModel()
t_vec_ref, a_vec_ref, rho_g_vec, rho_nu_vec, rho_NP_vec, P_NP_vec, Neff_vec = thermo_model_DNeff(0.)

def lnlike(theta, n_dim):
    """ n-d Gaussian Likelihood function
        `theta` is the parameter array, and any other args can be specified through `logl_args` in the sampler.
    """
    omega_b = theta[0]*1e-2
    omb_to_eta = const.Omegabh2_to_eta0
    eta = omega_b*omb_to_eta
    eta_fac = eta/(const.Omegabh2*omb_to_eta) # eta = eta_fac * eta0

    Yn, Yp, Yd, Yt, YHe3, Ya, YLi7, YBe7 = GetAbundances(
        jnp.array(rho_g_vec), 
        jnp.array(rho_nu_vec), 
        jnp.zeros_like(rho_g_vec),
        jnp.zeros_like(rho_g_vec), 
        t_vec=jnp.array(t_vec_ref), 
        a_vec=jnp.array(a_vec_ref), 
        eta_fac = jnp.asarray(eta_fac), 
        tau_n_fac = jnp.asarray(theta[1]), 
        nuclear_rates_q = jnp.asarray(theta[n_CMB:])
        )

    YHe = 4 * Ya
    YHe_BBN = 0.2449 # Aver 2015
    YHe_BBN_sigma = 0.004

    DH = Yd/Yp * 1e5 # rescale
    DH_BBN = 2.527 # Cooke 2018
    DH_BBN_sigma = 0.030 

    ll_BBN = -.5*(((YHe-YHe_BBN)/YHe_BBN_sigma)**2 + ((DH-DH_BBN)/DH_BBN_sigma)**2)
    return ll_BBN

def prior_cube(cube):
    """ Prior cube, specifying a bijection between the unit cube and the prior volume. Assumed uniform.
    """

    #omega_b 
    theta_min = np.array([1.9])
    theta_max = np.array([2.5])

    # uniform priors on omega_b
    cube[:len(theta_min)] = cube[:len(theta_min)] * (theta_max - theta_min) + theta_min

    # gaussian prior
    cube[n_CMB-1] = 1. + 0.000682 * ndtri(cube[n_CMB-1])
    
    # Gaussian priors for rate uncertainties, mean 0 and sigma 1 for each
    mu_Apl = jnp.zeros(num_reactions_used)
    sigma_Apl = jnp.ones(num_reactions_used)

    cube[n_CMB:] = mu_Apl + sigma_Apl * ndtri(cube[n_CMB:])

    return cube

def run_dynesty(lnlike, prior_cube, nlive=500, dlogz=0.5, n_cpus=36, method="dynamic"):

    with Pool(processes=n_cpus) as pool:

        if method == "dynamic":
            sampler = DynamicNestedSampler(lnlike, prior_cube, n_dim, pool=pool, queue_size=n_cpus, logl_args=(n_dim,))
            sampler.run_nested(dlogz_init=dlogz, nlive_init=nlive, nlive_batch=nlive)
        elif method == "static":
            sampler = NestedSampler(lnlike, prior_cube, n_dim, nlive=nlive, pool=pool, queue_size=n_cpus, logl_args=(n_dim,))
            sampler.run_nested(dlogz=dlogz)
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
