from functools import partial

import jax
from jax import lax
from jax import numpy as jnp 
from jax.scipy.special import zeta, gammaln, i0, i1, bernoulli
from jax.scipy.special import gamma as jax_gamma

euler_gamma = 0.57721566490153286061
zeta_3 = 1.202056903159594

# Number of terms for Li series. 
L = 60

# List of Bernoulli numbers B_n 
bernoulli_ary = bernoulli(L) 

def comb(N,k):
    """
    Combinatoric factor N! / (k! (N-k)!). 
    """
    return jnp.exp((gammaln(N+1) - gammaln(k+1) - gammaln(N-k+1) ))

def Bernoulli(n, x): 
    """
    The Bernoulli polynomial B_n(x), n < 60. See Wikipedia article. 
    """

    return lax.fori_loop(
        0, n+1, lambda i, val: val+bernoulli_ary[i] * comb(n,i) * x**(n-i), 0
    )

def gamma(x): 
    """
    Gamma function using the Lanczos approximation. Needed for imaginary
    arguments. 
    """
    def lanczos(x): 

        g = 7

        p_vals = jnp.array([
            0.99999999999980993,
            676.5203681218851,
            -1259.1392167224028,
            771.32342877765313,
            -176.61502916214059,
            12.507343278686905,
            -0.13857109526572012,
            9.9843695780195716e-6,
            1.5056327351493116e-7
        ])

        eps = 1e-06

        z = x
        z_fill = z - 1
        denom = z_fill + jnp.arange(1, len(p_vals))
        x_fill = p_vals[0] + jnp.sum(p_vals[1:] / denom)
        t = z_fill + g + 0.5

        result1 = (
            jnp.sqrt(2 * jnp.pi) * t ** (z_fill + 0.5) * jnp.exp(-t) * x_fill
        )
        result2 = jnp.where(
            jnp.abs(result1.imag) <= eps, result1.real, result1
        )
        return result2

    
    def reflect(x): 

        return jnp.pi / jnp.sin(jnp.pi * x) * lanczos(1. - x) 

    return lax.cond(x < 0.5, reflect, lanczos, x)


def Riemann_zeta(n): 
    """
    Riemann zeta function. Works for n > -60. Fixes results
    for negative arguments. 
    """

    return jnp.where(
        n > 0, zeta(n, 1), jnp.where(
            (n < 0) & (-n % 2 == 0), 0., 
            (-1.)**(-n) * bernoulli_ary[-n+1] / (-n + 1)
        )
    )

@partial(jax.jit, static_argnums=(0,))
def Li(n, z): 
    """
    Polylogarithm of order n and argument z. 
    """
    def _Li_z_small(z): 
    
        return lax.fori_loop(1, L, lambda j,val: val + z**j / j**n, 0)

    def _Li_z_intermed(z): 

        # Oddly enough, the fastest way to do this. 
        zeta_ary = jnp.concatenate((
            jnp.array([Riemann_zeta(n - m) for m in jnp.arange(n-1)]), 
            jnp.array([0.]), 
            jnp.array([Riemann_zeta(n - m) for m in jnp.arange(n, L)])
        ))

        zeta_series_term = jnp.sum(
            zeta_ary * jnp.concatenate(
                (jnp.array([1., jnp.log(z+0j)]), 
                jnp.log(z+0j)**jnp.arange(2, L))
            ) / jax_gamma(jnp.arange(L) + 1.)
        )

        H_n_m_1 = jnp.sum(1. / jnp.arange(1, n))
        harmonic_term = jnp.where(
            jnp.isclose(z - 1., 0), 0., 
            jnp.log(z+0j)**(n-1) / jax_gamma(n) * (
                H_n_m_1 - jnp.log(-jnp.log(
                    jnp.where(jnp.isclose(z - 1., 0), 2., z)+0j) + 0j
                )
            )
        )

        res = zeta_series_term + harmonic_term 

        return jnp.real(res)
    
    def _Li_z_large(z):

        recip_Li = lax.fori_loop(1, L, lambda j,val: val + (1/z)**j / j**n, 0)

        B_n = Bernoulli(n, jnp.log(z+0j)/(2 * jnp.pi * 1j))
 
        return jnp.real(
            - (-1)**n * recip_Li 
            - (2*jnp.pi*1j)**n / jax_gamma(n + 1) * B_n 
        )

    small_range = jnp.abs(z) <= 0.5

    intermed_range = (0.5 < jnp.abs(z)) & (jnp.abs(z) < 2)

    large_range = jnp.abs(z) > 2

    return jnp.where(
        small_range, _Li_z_small(jnp.where(small_range, z, 3.)), jnp.where(
            intermed_range, _Li_z_intermed(
                jnp.where(intermed_range, z, 3.)
            ), jnp.where(
                large_range, _Li_z_large(jnp.where(large_range, z, 3.)), 0.
            )
        ) 
    )

def K0(z): 
    """
    Modified Bessel function of the second kind of order 0. 
    See Zhang and Jin for algorithm. 
    """

    def K0_small(z):

        # n = 30 sufficient for abstol ~ 1e-12 and reltol ~ 1e-8
        int_ary = jnp.arange(1., 31)

        harmonic_ary = jnp.cumsum(1. / jnp.arange(1, 31))

        return -(jnp.log(z/2.) + euler_gamma) * i0(jnp.where(z < 600., z, 600.)) + jnp.sum(
            harmonic_ary * (z/2.)**(2.*int_ary) / jax_gamma(int_ary+1)**2.
        )

    def K0_large(z): 

        # n = 10 sufficient for abstol ~ 1e-12 and reltol ~ 1e-8

        int_ary = jnp.arange(1., 11)

        prod_term = jnp.cumprod(-(2.*int_ary - 1.) / (2.*int_ary) * (2.*int_ary - 1.)**2.) 

        res = 1. / 2. / z / i0(jnp.where(z < 600., z, 600.)) * (1. + jnp.sum((-1)**int_ary * prod_term / (2.*z)**(2.*int_ary)))

        return jnp.where(z < 600, res, 0.)

    return lax.cond(z < 9, K0_small, K0_large, z)

def K1(z): 
    """
    Modified Bessel function of the second kind of order 1. 
    """

    def K1_small(z): 

        return (1 / z - i1(z) * K0(z)) / i0(z)

    return jnp.where(z < 600., K1_small(jnp.where(z < 600. , z, 600.)), 0.)


def K2(z): 
    """
    Modified Bessel function of the second kind of order 2. 
    """

    return K0(z) + 2 / z * K1(z) 
