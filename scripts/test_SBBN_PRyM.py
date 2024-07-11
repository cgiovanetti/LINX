"""
Test function for LINX, small network. To run, do `python test_SBBN.py'. 
"""

import os
import sys

import time

import numpy as np
from numpy import random

import jax 
from jax import numpy as jnp
from jax import vmap


file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(file_dir+'/..')
 
from linx.background import BackgroundModel
from linx.abundances import AbundanceModel
import linx.nuclear as nucl

start_time = time.time()

thermo_model_DNeff = BackgroundModel()

(
    t_vec_ref, a_vec_ref, rho_g_vec, rho_nu_vec, rho_NP_vec, P_NP_vec, Neff_vec 
) = thermo_model_DNeff(jnp.asarray(0.))

abundance_model_PRIMAT_2018 = AbundanceModel(
    nucl.NuclearRates(nuclear_net='key_PRIMAT_2018')
)


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


Neff = Neff_vec[-1]
res = (Neff, ) + tuple(res_raw)

end_time = time.time() 


print(  
    "Neff: ", res[0],   "\n",
    "Yn: ", res[1], "Yp: ", res[2], 
    "\n",
    "Yd: ", res[3], "Yt: ", res[4], 
    "\n",
    "YHe3:", res[5], "YHe4:", res[6], 
    "\n",
    "YLi7:", res[7], "YBe7:", res[8] 
) 

Neff = res[0] 
Y_p = res[6] * 4. 
D_over_H_e5 = res[3] / res[2] * 1e5 
He3_over_H_e5 = (res[4] + res[5]) / res[2] * 1e5 
Li7_over_H_e10 = (res[7] + res[8]) / res[2] * 1e10

Neff_ref = 3.04438852
Y_p_ref = 0.24713916
D_over_H_e5_ref = 2.43725492 
He3_over_H_e5_ref = 1.03852818
Li7_over_H_e10_ref = 5.55704265

pct_diff = np.array(
    [
        (Neff - Neff_ref) / Neff_ref,
        (Y_p - Y_p_ref) / Y_p_ref, 
        (D_over_H_e5 - D_over_H_e5_ref) / D_over_H_e5_ref , 
        (He3_over_H_e5 - He3_over_H_e5_ref) / He3_over_H_e5_ref, 
        (Li7_over_H_e10 - Li7_over_H_e10_ref) / Li7_over_H_e10_ref 
    ]
) * 100


print('  ') 
print('Results in readable form:')
print('Neff: ', Neff)
print('Y_p: ', Y_p)
print('D/H x 10^5: ', D_over_H_e5) 
print('He3/H x 10^5: ', He3_over_H_e5) 
print('Li7/H x 10^10: ', Li7_over_H_e10)

print('\n')
print('Percent Difference (LINX - ref) / ref: ')

print(f'Neff: {pct_diff[0]:.5f}%')
print(f'Y_p: {pct_diff[1]:.5f}%') 
print(f'D/H: {pct_diff[2]:.5f}%')
print(f'He3/H: {pct_diff[3]:.5f}%') 
print(f'Li7/H: {pct_diff[4]:.5f}%') 

if np.max(np.abs(pct_diff)) < 0.2: 
    print('PRyMordial comparison test passed! Agreement with reference < 0.2%.')
else: 
    print('Test failed.')

print('-------------------------------------------')

print('Total time taken by LINX: ', end_time - start_time, ' seconds')


start_time = time.time()

(
    t_vec_ref, a_vec_ref, rho_g_vec, rho_nu_vec, rho_NP_vec, P_NP_vec, Neff_vec
) = thermo_model_DNeff(jnp.asarray(3.))


res = abundance_model_PRIMAT_2018(
    jnp.array(rho_g_vec), 
    jnp.array(rho_nu_vec), 
    jnp.array(rho_NP_vec),
    jnp.array(P_NP_vec),
    t_vec=jnp.array(t_vec_ref),
    a_vec=jnp.array(a_vec_ref),
    rtol=1e-6,
    sampling_nTOp=150
)

end_time = time.time() 

print('Total time taken by compiled single run: ', end_time - start_time, ' seconds')


print('----------------------------')
print('Testing Differentiability...')
print('----------------------------')

def get_m2LL(params):

    Delt_Neff = params[0]
    eta_fac   = params[1]
    tau_n_fac = params[2]
    nuclear_rates_q = params[3:]


    (
        t_vec_ref, a_vec_ref, rho_g_vec, rho_nu_vec, rho_NP_vec, P_NP_vec, 
        Neff_vec
    ) = thermo_model_DNeff(Delt_Neff)

    P_NP_vec = rho_NP_vec / 3.

    res = abundance_model_PRIMAT_2018(
        jnp.array(rho_g_vec), 
        jnp.array(rho_nu_vec), 
        jnp.array(rho_NP_vec),
        jnp.array(P_NP_vec),
        t_vec=jnp.array(t_vec_ref),
        a_vec=jnp.array(a_vec_ref), 
        eta_fac = jnp.asarray(eta_fac), 
        tau_n_fac = jnp.asarray(tau_n_fac), 
        nuclear_rates_q = nuclear_rates_q
    )

    Y_p      = res[6] * 4
    D_over_H = res[3] / res[2]

    Y_p_obs_mu   = 0.245
    Y_p_obs_sig  = 0.003

    D_over_H_obs_mu  = 2.547e-5
    D_over_H_obs_sig = 2.5e-7

    m2LL = (Y_p - Y_p_obs_mu)**2 / Y_p_obs_sig**2 + (D_over_H - D_over_H_obs_mu)**2 / D_over_H_obs_sig**2

    return m2LL

start_time = time.time() 

value_and_grad_m2LL = jax.jit(jax.value_and_grad(get_m2LL))

res = value_and_grad_m2LL(2.*jnp.ones(15))

print(res)

if jnp.isnan(jnp.array(res[1])).any() or jnp.isnan(res[0]): 

    print('Differentiability test failed.')

else: 

    print('Differentiability test passed!')

end_time = time.time() 

print('Total time taken by LINX to compile: ', end_time - start_time, ' seconds')

start_time = time.time() 

value_and_grad_m2LL(4.*jnp.ones(15))

end_time = time.time() 

print('Total time per call with grad: ', end_time - start_time, ' seconds')


