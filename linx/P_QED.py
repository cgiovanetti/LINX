import os 

import numpy as np

import jax.numpy as jnp 
import jax.lax as lax
from jax import grad, vmap

import linx.const as const 
import equinox as eqx


# high temperature behavior is not correct, probably because bounds need to
# change with T.
# Low temperature behavior is perfect.


def explicit_P0(T, me): # not needed, computed in thermo
    # 4.5 of https://arxiv.org/pdf/1911.04504

    prefac = T/jnp.pi**2

    p = jnp.linspace(0,50*T,num=3000) # this integral peaks at p close to T, integrating to 50*T is fine as long as resolution is very good
    Ee = jnp.sqrt(p**2 + me**2)
    integrand = p**2 * jnp.log( (1 + jnp.exp(-Ee/T))**2 / (1 - jnp.exp(-p/T)) ) 

    res = jnp.trapezoid(jnp.nan_to_num(integrand,nan=0),p) # p = 0 gives a nan, should be 0

    return prefac * res


def explicit_P2(T, me):
    # first compute 4.7 of https://arxiv.org/pdf/1911.04504, but ignoring the 
    # last term because it's itty bitty according to them

    e = jnp.sqrt(const.aFS * 4 * jnp.pi)
    prefac1 = - e**2 * T**2 / (12 * jnp.pi**2)
    prefac2 = - e**2/(8 * jnp.pi**4)

    p = jnp.linspace(0,50*T,num=2000) # this integral peaks at p close to T, integrating to 50*T is fine as long as resolution is good
    Ep = jnp.sqrt(p**2 + me**2)
    integrand = p**2/Ep * 2/(jnp.exp(Ep/T) + 1)

    res = jnp.trapezoid(integrand,p)

    return prefac1 * res + prefac2 * res**2

def explicit_P3(T, me):
    # compute 4.24 of https://arxiv.org/pdf/1911.04504
    
    e = jnp.sqrt(const.aFS * 4 * jnp.pi)
    prefac = e**3 * T/(12 * jnp.pi**4)

    p = jnp.linspace(0,50*T,num=2000) # this integral also peaks at p close to T, integrating to 50 is fine as long as resolution is good
    Ep = jnp.sqrt(p**2 + me**2)
    integrand = (p**2 + Ep**2)/Ep * 2/(jnp.exp(Ep/T) + 1)

    res = jnp.trapezoid(integrand,p)

    return prefac * res**(3./2)

def P_QED(T,me): # not needed, sums in thermo
    return explicit_P0(T, me) + explicit_P2(T, me) + explicit_P3(T, me)


dPdTQED_2 = grad(explicit_P2,argnums=0)
dPdTQED_3 = grad(explicit_P3,argnums=0)

# we don't need these computed explicitly, actually
d2PdT2QED_2 = grad(dPdTQED_2,argnums=0)
d2PdT2QED_3 = grad(dPdTQED_3,argnums=0)
