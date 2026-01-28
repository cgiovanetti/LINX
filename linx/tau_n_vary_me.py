import jax.numpy as jnp
import linx.const as const 
from jax import jit


def tau_n_fac_vary_me(me):
    """ Returns tau_n_fac during BBN given
    electron mass.

    Parameters
    ----------
    me : float
        Electron mass during BBN in MeV.

    Returns
    -------
    float
        The factor by which the neutron lifetime is scaled during
        BBN.
    """

    delta = const.mn - const.mp
    deltabar = delta/const.me
    deltabar_vary_me = delta/me

    # See https://arxiv.org/pdf/1801.08023 Eq (91)
    def f_int(deltabar):
        return jnp.sqrt(deltabar**2 - 1) * (-8 - 9*deltabar**2 + 2*deltabar**4)/60. + deltabar/4 * jnp.arccosh(deltabar)

    f_int_0 = f_int(deltabar)
    f_int_BBN = f_int(deltabar_vary_me)

    # new tau_n_fac = tau_n_BBN / const.tau_n
    return const.me**5/me**5 * f_int_0/f_int_BBN