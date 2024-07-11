"""
Test function for LINX, full network. To run, do `python test_SBBN.py'. 
"""

import os
import sys
import time

import jax
import numpy as np
import jax.numpy as jnp 
from jax import vmap


file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(file_dir+'/..')

from linx.background import BackgroundModel
from linx.abundances import AbundanceModel
from linx.nuclear import NuclearRates 
import linx.const as const 


start_time = time.time()

thermo_model_DNeff = BackgroundModel()

(
    t_vec_ref, a_vec_ref, rho_g_vec, rho_nu_vec, rho_NP_vec, P_NP_vec, Neff_vec 
) = thermo_model_DNeff(jnp.asarray(0.)) # need to pass as array for eqx.filter_jit

abundance_model_small_PRIMAT_2023 = AbundanceModel(
    NuclearRates(nuclear_net='small_PRIMAT_2023')
)
abundance_model_full_PRIMAT_2023 = AbundanceModel(
    NuclearRates(nuclear_net='full_PRIMAT_2023')
)


res_T_18 = abundance_model_small_PRIMAT_2023(
    rho_g_vec, 
    rho_nu_vec, 
    rho_NP_vec,
    P_NP_vec,
    t_vec=t_vec_ref,
    a_vec=a_vec_ref, 
    T_end=const.T_switch,
    sampling_nTOp=100, 
    rtol=1e-6
)

res_final = abundance_model_full_PRIMAT_2023(
    rho_g_vec, 
    rho_nu_vec, 
    rho_NP_vec,
    P_NP_vec,
    t_vec=t_vec_ref,
    a_vec=a_vec_ref, 
    Y_i=tuple(res_T_18)+(0., 0., 0., 0.),
    T_start=const.T_switch,
    sampling_nTOp=50,
    rtol=1e-6
)


Neff = Neff_vec[-1]
res = (Neff, ) + tuple(res_final)

end_time = time.time() 


print(  
    "Neff: ", res[0],   "\n",
    "Yn: ", res[1], "Yp: ", res[2], 
    "\n",
    "Yd: ", res[3], "Yt: ", res[4], 
    "\n",
    "YHe3:", res[5], "YHe4:", res[6], 
    "\n",
    "YLi7:", res[7], "YBe7:", res[8], 
    "\n", 
    "YHe6", res[9], "YLi6", res[10], 
    "\n", 
    "YLi8", res[11], "YB8", res[12]
) 

Neff = res[0] 
Y_p = res[6] * 4. 
D_over_H_e5 = res[3] / res[2] * 1e5 
He3_over_H_e5 = (res[4] + res[5]) / res[2] * 1e5 
Li7_over_H_e10 = (res[7] + res[8]) / res[2] * 1e10

# PRIMAT output
Neff_ref = 3.0435397
Y_p_ref = 0.24711702
D_over_H_e5_ref = 2.4385567
He3_over_H_e5_ref = 1.035961
Li7_over_H_e10_ref = 5.6299059

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
print('Percent Difference (LINX - PRIMAT) / PRIMAT: ')

print(f'Neff: {pct_diff[0]:.5f}%')
print(f'Y_p: {pct_diff[1]:.5f}%') 
print(f'D/H: {pct_diff[2]:.5f}%')
print(f'He3/H: {pct_diff[3]:.5f}%') 
print(f'Li7/H: {pct_diff[4]:.5f}%') 

if np.max(np.abs(pct_diff)) < 0.3: 
    print('PRIMAT comparison test passed! Agreement with reference < 0.3%.')
else: 
    print('Test failed.')

print('-------------------------------------------')

print('Total time taken by LINX: ', end_time - start_time, ' seconds')


start_time = time.time()

(
    t_vec_ref, a_vec_ref, rho_g_vec, rho_nu_vec, rho_NP_vec, P_NP_vec, Neff_vec 
) = thermo_model_DNeff(jnp.asarray(3.))



res_T_18 = abundance_model_small_PRIMAT_2023(
    rho_g_vec, 
    rho_nu_vec, 
    rho_NP_vec,
    P_NP_vec,
    t_vec=t_vec_ref,
    a_vec=a_vec_ref, 
    T_end=const.T_switch,
    sampling_nTOp=100, 
    rtol=1e-6
)

res_final = abundance_model_full_PRIMAT_2023(
    rho_g_vec, 
    rho_nu_vec, 
    rho_NP_vec,
    P_NP_vec,
    t_vec=t_vec_ref,
    a_vec=a_vec_ref, 
    Y_i=tuple(res_T_18)+(0., 0., 0., 0.),
    T_start=const.T_switch,
    sampling_nTOp=50,
    rtol=1e-6
)

end_time = time.time() 

print('Total time taken by compiled single run: ', end_time - start_time, ' seconds')


print('-------------------------------------------')
