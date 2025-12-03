import jax
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, Tsit5, PIDController, SaveAt

import time

def fun(rtol=1e-8, atol=1e-10,solver=Tsit5()): 
    T_EM_init = 8.6
    rho_extra_init = 830.

    Y0 = (0., T_EM_init)

    sol = diffeqsolve(
        ODETerm(dY), solver, args=(rho_extra_init),
        t0 = 0., t1=100., dt0=None, y0=Y0, 
        saveat=SaveAt(steps=True), 
        stepsize_controller = PIDController(
            rtol=rtol, atol=atol
        ), 
        max_steps=512
    )

    a_vec = jnp.exp(sol.ys[0])

    return (
        a_vec
    )

def dY(t, Y, args): 
    lna, T_g = Y
    rho_extra_init = args

    rho_EM = T_g**4
    rho_extra = rho_extra_init * 1. / jnp.exp(lna)**4 

    H = (rho_EM + rho_extra)**0.5
    drho_EM_dt = -3 * H * rho_EM
    dT_g_dt = drho_EM_dt / (4*T_g**3)

    return H, dT_g_dt

for i in range(5):
    start = time.time()
    a_vec = jax.block_until_ready(jax.jit(fun)())
    print(time.time() - start)
    