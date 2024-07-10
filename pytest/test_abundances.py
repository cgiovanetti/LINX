import pytest
import os
import sys

import numpy as np

import jax 
from jax import numpy as jnp
from jax import vmap


file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(file_dir+'/..')
 
from linx.background import BackgroundModel
from linx.abundances import AbundanceModel
import linx.nuclear as nucl


thermo_model_DNeff = BackgroundModel()


abundance_model_PRIMAT_2018 = AbundanceModel(
    nucl.NuclearRates(nuclear_net='key_PRIMAT_2018')
)


@pytest.fixture
def sample_inputs():
    t_vec_ref, a_vec_ref, rho_g_vec, rho_nu_vec, rho_NP_vec, P_NP_vec, Neff_vec = thermo_model_DNeff(0.)
    
    return t_vec_ref, a_vec_ref, rho_g_vec, rho_nu_vec, rho_NP_vec, P_NP_vec, Neff_vec 

@pytest.fixture
def reference_values():
    return np.array([
        3.0444,    # Neff
        0.2471,    # Y_p
        2.4386,    # D_over_H_e5
        1.0385,    # He3_over_H_e5
        5.5503     # Li7_over_H_e10
    ])

# Test function
def test_abundance_model_PRIMAT_2018(sample_inputs, reference_values):
    t_vec_ref, a_vec_ref, rho_g_vec, rho_nu_vec, rho_NP_vec, P_NP_vec, Neff_vec = sample_inputs

    # Call the function
    res_raw = abundance_model_PRIMAT_2018(
        rho_g_vec, 
        rho_nu_vec, 
        rho_NP_vec,
        P_NP_vec,
        t_vec=t_vec_ref,
        a_vec=a_vec_ref, 
        rtol=1e-6,
        sampling_nTOp=150
    )

    # Extract variables from the result
    Neff = Neff_vec[-1]
    res = (Neff, ) + tuple(res_raw)

    Neff = res[0]
    Y_p = res[6] * 4.
    D_over_H_e5 = res[3] / res[2] * 1e5
    He3_over_H_e5 = (res[4] + res[5]) / res[2] * 1e5
    Li7_over_H_e10 = (res[7] + res[8]) / res[2] * 1e10

    # Combine results into an array for comparison
    results = np.array([Neff, Y_p, D_over_H_e5, He3_over_H_e5, Li7_over_H_e10])

    # Perform a single np.allclose comparison
    assert np.allclose(results, reference_values, rtol=1e-3)

