import jax.numpy as jnp
import jax
from jax import jit, vmap
import sys

sys.path.append("../")
import linx.const as const 
from linx.nuclear import NuclearRates
from linx.background import BackgroundModel
from linx.abundances import AbundanceModel

thermo_model_DNeff = BackgroundModel()

(
    t_vec_ref, a_vec_ref, rho_g_vec, rho_nu_vec, rho_NP_vec, P_NP_vec, Neff_vec 
) = thermo_model_DNeff(jnp.asarray(0.))

network = 'key_PRIMAT_2023'
# network = 'key_PRIMAT_2018'
# network = 'key_PArthENoPE'
# network = 'key_YOF'
abundance_model = AbundanceModel(NuclearRates(nuclear_net=network))

Planck_omega_b_res = abundance_model(
    rho_g_vec, # photon energy density
    rho_nu_vec, # neutrino energy density
    rho_NP_vec, # energy density of extra species
    P_NP_vec, # pressure of extra species
    t_vec=t_vec_ref, # vector of times at which quantities are given
    a_vec=a_vec_ref # vector of scale factor at corresponding times
)

print('n:   ', Planck_omega_b_res[0])