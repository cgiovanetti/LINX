import sys
sys.path.append('../../../cmb-emu-analysis/')
sys.path.append('../../LINX')

from absl import flags, logging
logging.set_verbosity(logging.INFO)

import numpy as np
import pickle

import jax
from jax import numpy as jnp
from jax import config
import diffrax as dfx
import optax

config.update("jax_enable_x64", True)

import numpyro
import numpyro.distributions as dist
from numpyro import optim
from numpyro.infer import SVI, Trace_ELBO, autoguide, MCMC, NUTS
import numpyro.infer.initialization as init
from numpyro.infer.reparam import NeuTraReparam
from numpyro.diagnostics import print_summary

numpyro.set_host_device_count(2)

import linx.const as const
from linx.nuclear import NuclearRates
from linx.background import BackgroundModel
from linx.abundances import AbundanceModel

try:
    from modules.cosmopower_jax import CosmoPowerJAX as CPJ
    from likelihoods.planck2018_lite.likelihood import PlanckLiteJax
except ImportError:
    logging.warning("Could not import CMB emulators. Only BBN-only run possible.")


def run(bbn_only=True, n_steps_svi=5, n_particles_svi=1, lr=1e-3, rng_svi=152, n_samples_mcmc=10, n_warmup_mcmc=5, n_chains=2, rng_mcmc=442, step_size=1e-2):
    """ Run SVI, neural transport reparameterization, and NUTS/HMC

    Args:
        bbn_only (bool, optional): Just BBN, no CMB. Defaults to True.
        n_steps_svi (int, optional): Number of SVI steps. Defaults to 5.
        n_particles_svi (int, optional): Number of SVI particles for ELBO estimate. Defaults to 1.
        lr (float, optional): Learning rate for SVI. Defaults to 1e-3.
        rng_svi (int, optional): SVI RNG. Defaults to 152.
        n_samples_mcmc (int, optional): Number of HMC samples. Defaults to 10.
        n_warmup_mcmc (int, optional): Number of HMC warmup steps. Defaults to 5.
        n_chains (int, optional): Number of chains. Defaults to 2.
        rng_mcmc (int, optional): HMC RNG. Defaults to 442.
        step_size (float, optional): HMC step size. Defaults to 1e-2.

    Returns:
        _type_: _description_
    """
    
    thermo_model_DNeff = BackgroundModel()

    if bbn_only:
        
        logging.info("BBN-only analysis. No CMB emulators loaded.")
        loglike_cmb = lambda x: 0.0

    else:

        logging.info("Joint BBN and CMB analysis. Loading CMB emulators...")

        emulator_custom_TT = CPJ(probe='custom_log', filename='cmb_neff_spt_TT_NN.pkl')
        emulator_custom_TE = CPJ(probe='custom_pca', filename='cmb_neff_spt_TE_PCAplusNN.pkl')
        emulator_custom_EE = CPJ(probe='custom_log', filename='cmb_neff_spt_EE_NN.pkl')

        ell = emulator_custom_TT.modes

        ellmin = int(ell[0]) 

        plite = PlanckLiteJax(year=2018, spectra="TTTEEE", ellmin=ellmin, use_low_ell_bins=True, data_directory='../../../cmb-emu-analysis/likelihoods/planck2018_lite/data')

        def loglike_cmb(params):
            """ Emulation + log-likelihood wrapper
            """

            ell = emulator_custom_TT.modes
            Dltt = emulator_custom_TT.predict(params) * ell * (ell + 1) / (2 * np.pi)
            Dlte = emulator_custom_TE.predict(params) * ell * (ell + 1) / (2 * np.pi)
            Dlee = emulator_custom_EE.predict(params) * ell * (ell + 1) / (2 * np.pi)

            return plite.loglike(Dltt, Dlte, Dlee)

    abundance_model_PRIMAT_2023 = AbundanceModel(NuclearRates(nuclear_net='key_PRIMAT_2023'))

    def loglike_bbn(params):
        """
        params should be vector of length 14, with params[0] = eta_fac (mean = 1, broad uniform prior(e.g. 0.1 - 10?)), 
        params[1] = tau_n_fac ~ N(mean = 1, sigma = const.sigma_tau_n / const.tau_n) 
        params[2:] = p_nuclear_process ~ N(0,1) corresponding to npdg, dpHe3g, ddHe3n, ddtp, tpag, tdan, taLi7g, He3ntp, He3dap, He3aBe7g, Be7nLi7p, Li7paa. 
        """

        Delt_Neff = params[0]
        eta_fac   = params[1]
        tau_n_fac = params[2]
        nuclear_rates_q = params[3:]

        t_vec_ref, a_vec_ref, rho_g_vec, rho_nu_vec, rho_NP_vec, P_NP_vec, Neff_vec = thermo_model_DNeff(Delt_Neff)

        vJax_res = abundance_model_PRIMAT_2023(
            jnp.array(rho_g_vec), 
            jnp.array(rho_nu_vec), 
            jnp.array(rho_NP_vec),
            jnp.array(P_NP_vec),
            t_vec=jnp.array(t_vec_ref),
            a_vec=jnp.array(a_vec_ref), 
            eta_fac = jnp.asarray(eta_fac), 
            tau_n_fac = jnp.asarray(tau_n_fac), 
            nuclear_rates_q = nuclear_rates_q,
            rtol=1e-4, atol=1e-9, solver=dfx.Kvaerno3(), max_steps=8092 * 12
            # rtol=1e-7, atol=1e-9, solver=dfx.ImplicitEuler(), max_steps=8092 * 2
        )

        YHe = 4 * vJax_res[5]
        YHe_BBN = 0.2449  # Aver 2015
        YHe_BBN_sigma = 0.004

        DH = vJax_res[2] / vJax_res[1] * 1e5  # Just avoid small numbers
        DH_BBN = 2.527  # Cooke 2018
        DH_BBN_sigma = 0.030 

        ll_BBN = -.5*(((YHe-YHe_BBN)/YHe_BBN_sigma)**2 + ((DH-DH_BBN)/DH_BBN_sigma)**2)
        return ll_BBN, Neff_vec[-1]


    def model():

        eta_fac = numpyro.sample("eta_fac", dist.Uniform(0.85, 1.2))

        sigma_tau_fac = 0.000682

        tau_fac = numpyro.sample("tau_fac", dist.TwoSidedTruncatedDistribution(dist.Normal(1.0, sigma_tau_fac), low=1.0 - 5. * sigma_tau_fac, high=1.0 + 5. * sigma_tau_fac))
        nuclear_rates_q = numpyro.sample("nuclear_rates_q", dist.Normal(0., 1.0).expand([12]))

        DeltaNeff = numpyro.sample("DNeff", dist.Uniform(-7, 7))
        params = jnp.array([DeltaNeff, eta_fac, tau_fac, *nuclear_rates_q])
            
        log_like, Neff = loglike_bbn(params)
        Neff = numpyro.deterministic("Neff", Neff)

        ombh2 = numpyro.deterministic("ombh2", eta_fac * const.Omegabh2)
        ombh2 = eta_fac * const.Omegabh2

        if not bbn_only:

            omch2 = numpyro.sample('omch2', dist.Uniform(0.08, 0.2))
            h = numpyro.sample('h', dist.Uniform(0.4, 1.0))
            ns = numpyro.sample('ns', dist.Uniform(0.88, 1.06))
            logA = numpyro.sample('logA', dist.Uniform(2.5, 3.5))
            tau = numpyro.sample('tau', dist.TruncatedDistribution(dist.Normal(0.0506, 0.0086), low=1e-4, high=0.15))

            log_like = log_like + loglike_cmb(jnp.array([ombh2, omch2, h, tau, ns, logA, Neff,]))

        numpyro.factor('log_like', log_like)

    # def model(dim=20):
    #     y = numpyro.sample("y", dist.Normal(0, 3))
    #     numpyro.sample("x", dist.Normal(jnp.zeros(dim - 1), jnp.exp(y / 2)))

    ## SVI

    logging.info("Running SVI...")

    guide = autoguide.AutoIAFNormal(model, num_flows=4, nonlinearity=jax.example_libraries.stax.Elu, skip_connections=False, hidden_dims=[128, 128],)

    optimizer = optim.optax_to_numpyro(optax.adam(lr,))
    svi = SVI(model, guide, optimizer, Trace_ELBO(num_particles=n_particles_svi),)
    svi_results = svi.run(jax.random.PRNGKey(rng_svi), n_steps_svi)

    ## Neural Transport Reparameterization

    logging.info("Doing neural transport reparam...")

    neutra = NeuTraReparam(guide, svi_results.params)
    model_neutra = neutra.reparam(model)

    ## NUTS/HMC

    logging.info("Running MCMC...")

    nuts_kernel = NUTS(model_neutra, max_tree_depth=4, dense_mass=True, step_size=step_size, init_strategy=init.init_to_mean)
    mcmc = MCMC(nuts_kernel, num_warmup=n_warmup_mcmc, num_samples=n_samples_mcmc, num_chains=n_chains, chain_method='parallel')
    mcmc.run(jax.random.PRNGKey(rng_mcmc))

    ## Postprocessing

    logging.info("Postprocessing samples...")
    zs = mcmc.get_samples(group_by_chain=True)["auto_shared_latent"]

    logging.info("Transform samples into unwarped space...")
    samples = neutra.transform_sample(zs)

    print_summary(samples)

    ## Save samples

    logging.info("Saving samples...")
    
    # Make filename based on flags
    save_filename = f"samples_svi_mcmc_bbn_only_{bbn_only}_n_steps_svi_{n_steps_svi}_n_particles_svi_{n_particles_svi}_lr_{lr}_rng_svi_{rng_svi}_n_samples_mcmc_{n_samples_mcmc}_n_warmup_mcmc_{n_warmup_mcmc}_n_chains_{n_chains}_rng_mcmc_{rng_mcmc}_step_size_{step_size}.pkl"
    with open(save_filename, "wb") as f:
        pickle.dump(samples, f)

    # # Test opening the file
    # with open(save_filename, "rb") as f:
    #     samples = pickle.load(f)

    logging.info("Done! Have a great day.")


if __name__ == "__main__":
    FLAGS = flags.FLAGS

    flags.DEFINE_boolean('bbn_only', True, 'Run BBN-only analysis')
    flags.DEFINE_integer('n_steps_svi', 2000, 'Number of SVI steps')
    flags.DEFINE_integer('n_particles_svi', 1, 'Number of particles for SVI')
    flags.DEFINE_float('lr', 1.e-3, 'Learning rate for SVI')
    flags.DEFINE_integer('rng_svi', 152, 'Random seed for SVI')
    flags.DEFINE_integer('n_samples_mcmc', 2000, 'Number of MCMC samples')
    flags.DEFINE_integer('n_warmup_mcmc', 300, 'Number of MCMC warmup steps')
    flags.DEFINE_integer('n_chains', 2, 'Number of MCMC chains')
    flags.DEFINE_integer('rng_mcmc', 442, 'Random seed for MCMC')
    flags.DEFINE_float('step_size', 1e-2, 'MCMC step size')

    # Parse flags
    FLAGS(sys.argv)
    
    logging.info("Number of available devices: %d" % jax.local_device_count())

    # Start training run
    run(bbn_only=FLAGS.bbn_only,
        n_steps_svi=FLAGS.n_steps_svi,
        n_particles_svi=FLAGS.n_particles_svi,
        lr=FLAGS.lr,
        rng_svi=FLAGS.rng_svi,
        n_samples_mcmc=FLAGS.n_samples_mcmc,
        n_warmup_mcmc=FLAGS.n_warmup_mcmc,
        n_chains=FLAGS.n_chains,
        rng_mcmc=FLAGS.rng_mcmc,
        step_size=FLAGS.step_size)
