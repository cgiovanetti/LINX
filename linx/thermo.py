import os 

import numpy as np

import jax.numpy as jnp 
import jax.lax as lax
from jax import grad, vmap

import linx.const as const 
from linx.special_funcs import Li, K1, K2

###########################################
#                 Cosmology               #
############################################


def Hubble(rho_tot):
    """
    The Hubble parameter in s^-1. 

    Parameters
    ----------
    rho_tot : float or array
        The total energy density in MeV^4. 

    Returns
    -------
    float or array
    """
    
    return (
        rho_tot * 8 * jnp.pi / (3 * const.Mpl**2)
    )**0.5 / const.hbar

def N_eff(rho_tot, rho_g):
    """ Neff parameter. 

    Parameters
    ----------
    rho_tot : float or array
        Total energy density of all fluids. 
    rho_g : float or array
        Energy density of photons. 

    Returns
    -------
    float or array

    """

    return 8./7. * (11./4.)**(4./3.) * (rho_tot-rho_g) / rho_g

def nB(a, eta_fac=1.):
    """ Number density of baryons in MeV^3. 

    Parameters
    ----------
    a : float or array
        Scale factor of interest. 
    eta_fac : float, optional
        Factor to rescale the central value of the baryon-to-photon ratio by.

    Returns
    -------
    float or array

    """
    
    n0B = eta_fac*const.n0CMB*const.eta0 # baryon density of today MeV^3
    return n0B/a**3 # MeV^3


####################################################################
#                 Generic Thermodynamic Variables                  #
####################################################################


def rho_massless_BE(T, mu, g):
    """
    Energy density of a massless particle with Bose-Einstein statistcs.

    Parameters
    ----------
    T : float
        Temperature of massless particle species in units of MeV.
    mu : float
        Chemical potential of massless particle species.
    g : float
        Degrees of freedom of massless particle species (1 for scalar, 
        2 for vector, etc.).

    Returns
    -------
    float
        Units of MeV^4. 
    """
    # epsilonbig = 1e12
    # epsilonsmall = 1e-12

    # T_non_negative = jnp.maximum(T, epsilonsmall) 
    # T_safer = lax.cond(
    #     T_non_negative < epsilonbig,
    #     lambda _: T_non_negative,
    #     lambda _: 0.0,
    #     None
    # )

    # T_safe_pow4 = T_safer**4

    # return g * 3 / jnp.pi**2 * T_safe_pow4 * Li(4, jnp.exp(mu/T_safer))
    
    return lax.cond(
        T > 0., 
        lambda T: g * 3 / jnp.pi**2 * Li(4, jnp.exp(mu/T)) * T**4, 
        lambda T: 0.,
        T
    )


def n_massless_BE(T, mu, g): 
    """
    Number density of a massless particle with Bose-Einstein statistcs.

    Parameters
    ----------
    T : float 
        Temperature of massless particle species in units of MeV.
    mu : float
        Chemical potential of massless particle species.
    g : float
        Degrees of freedom of massless particle species (1 for scalar, 
        2 for vector, etc.).

    Returns
    -------
    float
        Units of MeV^3. 
    """

    return lax.cond(
        T > 0., 
        lambda T: g / jnp.pi**2 * Li(3, jnp.exp(mu/T)) * T**3, 
        lambda T: 0., 
        T
    )

def p_massless_BE(T, mu, g): 
    """
    Pressure of a massless particle with Bose-Einstein statistcs.

    Parameters
    ----------
    T : float
        Temperature of massless particle species in units of MeV.
    mu : float
        Chemical potential of massless particle species.
    g : float
        Degrees of freedom of massless particle species (1 for scalar, 
        2 for vector, etc.).

    Returns
    -------
    float
        Units of MeV^4. 
    """

    return rho_massless_BE(T, mu, g) / 3. 

def rho_massless_FD(T, mu, g): 
    """
    Energy density of a massless particle with Fermi-Dirac statistcs.

    Parameters
    ----------
    T : float
        Temperature of massless particle species in units of MeV.
    mu : float
        Chemical potential of massless particle species.
    g : float
        Degrees of freedom of massless particle species (2 for Weyl fermion).

    Returns
    -------
    float
        Units of MeV^4. 
    """

    return lax.cond(
        T > 0., 
        lambda T: -g * 3 / jnp.pi**2 * Li(4, -jnp.exp(mu/T)) * T**4, 
        lambda T: 0., 
        T
    )
    # T = jnp.where(jnp.isnan(T**4),0,T)
    # value = jnp.where(jnp.isnan(g * 45 / jnp.pi**2 * T**4),0, T**4)
    # epsilonbig = 1e12
    # epsilonsmall = 1e-12

    # # Compute T^4 safely, avoid potential underflow or NaN by ensuring T is always above epsilon
    # T_non_negative = jnp.maximum(T, epsilonsmall) 
    # T_safer = lax.cond(
    #     T_non_negative < epsilonbig,
    #     lambda _: T_non_negative,
    #     lambda _: 0.0,
    #     None
    # )

    # T_safe_pow4 = T_safer**4
    # return g * 3 / jnp.pi**2 * T_safe_pow4 #* Li(4, -jnp.exp(mu/T))

def n_massless_FD(T, mu, g): 
    """
    Number density of a massless particle with Fermi-Dirac statistcs.

    Parameters
    ----------
    T : float
        Temperature of massless particle species in units of MeV.
    mu : float
        Chemical potential of massless particle species.
    g : float
        Degrees of freedom of massless particle species (2 for Weyl fermion).

    Returns
    -------
    float
        Units of MeV^3. 
    """

    return lax.cond(
        T > 0., 
        lambda T: -g / jnp.pi**2 * Li(3, -jnp.exp(mu/T)) * T**3, 
        lambda T: 0., 
        T
    )

def p_massless_FD(T, mu, g): 
    """
    Pressure of a massless particle with Fermi-Dirac statistcs.

    Parameters
    ----------
    T : float
        Temperature of massless particle species in units of MeV.
    mu : float
        Chemical potential of massless particle species.
    g : float
        Degrees of freedom of massless particle species (2 for Weyl fermion).

    Returns
    -------
    float
        Units of MeV^4. 
    """

    return rho_massless_FD(T, mu, g) / 3. 

def rho_massless_MB(T, mu, g): 
    """
    Energy density of a massless particle with Maxwell-Boltzmann statistcs.

    Parameters
    ----------
    T : float
        Temperature of massless particle species in units of MeV.
    mu : float
        Chemical potential of massless particle species.
    g : float
        Degrees of freedom of massless particle species.

    Returns
    -------
    float
        Units of MeV^4. 
    """

    return lax.cond(
        T > 0., 
        lambda T: g * 3 / jnp.pi**2 * jnp.exp(mu/T) * T**4, 
        lambda T: 0., 
        T
    )

def n_massless_MB(T, mu, g): 
    """
    Number density of a massless particle with Maxwell-Boltzmann statistcs.

    Parameters
    ----------
    T : float
        Temperature of massless particle species in units of MeV.
    mu : float
        Chemical potential of massless particle species.
    g : float
        Degrees of freedom of massless particle species.

    Returns
    -------
    float
        Units of MeV^3. 
    """

    return lax.cond(
        T > 0., 
        lambda T: g / jnp.pi**2 * jnp.exp(mu/T) * T**3, 
        lambda T: 0., 
        T
    )

def p_massless_MB(T, mu, g): 
    """
    Pressure of a massless particle with Maxwell-Boltzmann statistcs.

    Parameters
    ----------
    T : float
        Temperature of massless particle species in units of MeV.
    mu : float
        Chemical potential of massless particle species.
    g : float
        Degrees of freedom of massless particle species.

    Returns
    -------
    float
        Units of MeV^4. 
    """

    return rho_massless_MB(T, mu, g) / 3. 

# Parameters for series approximation of (massive) thermodynamic integrals.
# Series method is much faster than integral computation.
N_series_terms=20 # 20 Seems fine for mu < 0.7 T, << 1% error. 
rel_thres=30.

def rho_massive_BE(T, mu, m, g): 
    """
    Series approximation for energy density of a massive
    particle with Bose-Einstein statistcs.

    Parameters
    ----------
    T : float
        Temperature of massive particle species in MeV.
    mu : float
        Chemical potential of massive particle species in MeV. 
    m : float
        Mass of particle in MeV.
    g : float
        Degrees of freedom of massive particle species (1 for scalar, 
        3 for vector, etc.).

    Returns
    -------
    float
        Units of MeV^4. 
    """
    def res(i, val): 

        return val + jnp.exp(i * mu / T) * (
            (m / T)**3 / i * K1(i * m / T) 
            + 3 * (m / T)**2 / i**2 * K2(i * m / T)
        )

    return jnp.where(
        (T / m > rel_thres) | (T <= 0.), 
        rho_massless_BE(T, mu, g), 
        g / (2 * jnp.pi**2) * T**4 * lax.fori_loop(1, N_series_terms, res, 0)
    )

def n_massive_BE(T, mu, m, g): 
    """
    Series approximation for number density of a massive
    particle with Bose-Einstein statistcs.

    Parameters
    ----------
    T : float
        Temperature of massive particle species in MeV.
    mu : float
        Chemical potential of massive particle species in MeV. 
    m : float
        Mass of particle in MeV.
    g : float
        Degrees of freedom of massive particle species (1 for scalar, 
        3 for vector, etc.).

    Returns
    -------
    float
        Units of MeV^3. 
    """

    def res(i, val): 

        return val + jnp.exp(i * mu / T) * (m / T)**2 / i * K2(i * m / T)

    return jnp.where(
        (T / m > rel_thres) | (T <= 0.), 
        n_massless_BE(T, mu, g), 
        g / (2 * jnp.pi**2) * T**3 * lax.fori_loop(1, N_series_terms, res, 0)
    )


def p_massive_BE(T, mu, m, g): 
    """
    Series approximation for pressure of a massive
    particle with Bose-Einstein statistcs.

    Parameters
    ----------
    T : float
        Temperature of massive particle species in MeV.
    mu : float
        Chemical potential of massive particle species in MeV.
    m : float
        Mass of particle in MeV.
    g : float
        Degrees of freedom of massive particle species (1 for scalar, 
        3 for vector, etc.).

    Returns
    -------
    float
        Units of MeV^4. 
    """

    def res(i, val): 

        return val + jnp.exp(i * mu / T) * 3 * (m / T)**2 / i**2 * K2(i * m / T)

    return jnp.where(
        (T / m > rel_thres) | (T <= 0.), 
        p_massless_BE(T, mu, g), 
        g / (6 * jnp.pi**2) * T**4 * lax.fori_loop(1, N_series_terms, res, 0)
    )

def rho_massive_FD(T, mu, m, g):
    """
    Series approximation for energy density of a massive
    particle with Fermi-Dirac statistcs.

    Parameters
    ----------
    T : float
        Temperature of massive particle species in MeV.
    mu : float
        Chemical potential of massive particle species in MeV.
    m : float
        Mass of particle in MeV.
    g : float
        Degrees of freedom of massive particle species (2 for 
        Majorana Fermion, 4 for Dirac Fermion, etc.).

    Returns
    -------
    float
        Units of MeV^4. 
    """ 

    def res(i, val): 

        return val + (-1)**(i - 1) * jnp.exp(i * mu / T) * (
            (m / T)**3 / i * K1(i * m / T) 
            + 3 * (m / T)**2 / i**2 * K2(i * m / T)
        )


    return jnp.where(
        (T / m > rel_thres) | (T <= 0.), 
        rho_massless_FD(T, mu, g), 
        g / (2 * jnp.pi**2) * T**4 * lax.fori_loop(1, N_series_terms, res, 0)
    )

def n_massive_FD(T, mu, m, g): 
    """
    Series approximation for number density of a massive
    particle with Fermi-Dirac statistcs.

    Parameters
    ----------
    T : float
        Temperature of massive particle species in MeV.
    mu : float
        Chemical potential of massive particle species in MeV.
    m : float
        Mass of particle in MeV.
    g : float
        Degrees of freedom of massive particle species (2 for 
        Majorana Fermion, 4 for Dirac Fermion, etc.).

    Returns
    -------
    float
        Units of MeV^3. 
    """ 

    def res(i, val): 

        return val + (
            (-1)**(i - 1) * jnp.exp(i * mu / T) * (m / T)**2 / i * K2(i * m / T)
        )

    return jnp.where(
        (T / m > rel_thres) | (T <= 0.), 
        n_massless_FD(T, mu, g), 
        g / (2 * jnp.pi**2) * T**3 * lax.fori_loop(1, N_series_terms, res, 0)
    )

def p_massive_FD(T, mu, m, g): 
    """
    Series approximation for pressure of a massive
    particle with Fermi-Dirac statistcs.
    
    Parameters
    ----------
    T : float
        Temperature of massive particle species in MeV.
    mu : float
        Chemical potential of massive particle species in MeV.
    m : float
        Mass of particle in MeV.
    g : float
        Degrees of freedom of massive particle species (1 for 
        Majorana Fermion, 2 for Dirac Fermion, etc.).

    Returns
    -------
    float
        Units of MeV^4. 
    """ 

    def res(i, val): 

        return val + (
            (-1)**(i - 1) * jnp.exp(i * mu / T) 
            * 3 * (m / T)**2 / i**2 * K2(i * m / T)
        )

    return jnp.where(
        (T / m > rel_thres) | (T <= 0.), 
        p_massless_FD(T, mu, g), 
        g / (6 * jnp.pi**2) * T**4 * lax.fori_loop(1, N_series_terms, res, 0)
    )

def rho_massive_MB(T, mu, m, g): 
    """
    Series approximation for energy density of a massive
    particle with Maxwell-Boltzmann statistcs.

    Parameters
    ----------
    T : float
        Temperature of massive particle species in MeV.
    mu : float
        Chemical potential of massive particle species in MeV
    m : float
        Mass of particle in MeV.
    g : float
        Degrees of freedom of massive particle species.

    Returns
    -------
    float
        Units of MeV^4. 
    """ 

    return lax.cond(
        T > 0., 
        lambda T: g * m**2 * T * jnp.exp(mu/T) / (2 * jnp.pi**2) * (
            m * K1(m / T) + 3 * T * K2(m / T)
        ), 
        lambda T: 0., 
        T
    )

def n_massive_MB(T, mu, m, g): 
    """
    Series approximation for number density of a massive
    particle with Maxwell-Boltzmann statistcs.

    Parameters
    ----------
    T : float
        Temperature of massive particle species in MeV.
    mu : float
        Chemical potential of massive particle species in MeV.
    m : float
        Mass of particle in MeV.
    g : float
        Degrees of freedom of massive particle species.

    Returns
    -------
    float
        Units of MeV^3. 
    """ 

    return lax.cond(
        T > 0., 
        lambda T: g * m**2 * T * jnp.exp(mu/T) / (2 * jnp.pi**2) * K2(m / T), 
        lambda T: 0., 
        T
    )

def p_massive_MB(T, mu, m, g): 
    """
    Series approximation for pressure of a massive
    particle with Maxwell-Boltzmann statistcs.

    Parameters
    ----------
    T : float
        Temperature of massive particle species in units of MeV.
    mu : float
        Chemical potential of massive particle species.
    m : float
        Mass of particle in units of MeV.
    g : float
        Degrees of freedom of massive particle species.

    Returns
    -------
    float
        Units of MeV^4. 
    """ 

    return n_massive_MB(T, mu, m, g) * T

####################################################################
#               Electromagnetic Sector and Neutrinos               #
####################################################################

file_dir = os.path.dirname(__file__)

# QED Corrections
P_QED_tab = np.loadtxt(file_dir+"/data/background/"+"QED_P_int.txt")
dPdT_QED_tab = np.loadtxt(file_dir+"/data/background/"+"QED_dP_intdT.txt")
d2PdT2_QED_tab = np.loadtxt(file_dir+"/data/background/"+"QED_d2P_intdT2.txt")

# Effect of standard value of electron mass in scattering matrix elements
f_nue_scat_tab = np.loadtxt(file_dir+"/data/background/"+"nue_scatt.txt")
f_numu_scat_tab = np.loadtxt(file_dir+"/data/background/"+"numu_scatt.txt")

# Effect of standard value of electron mass in annihilation matrix elements
f_nue_ann_tab = np.loadtxt(file_dir+"/data/background/"+"nue_ann.txt")
f_numu_ann_tab = np.loadtxt(file_dir+"/data/background/"+"numu_ann.txt")

try:
    gpus = jax.devices('gpu')
    P_QED_tab = jax.device_put(
        P_QED_tab, device=gpus[0] 
    )
    dPdT_QED_tab = jax.device_put(
        dPdT_QED_tab, device=gpus[0]
    )
    d2PdT2_QED_tab  = jax.device_put(
        d2PdT2_QED_tab , device=gpus[0]
    )

    f_nue_scat_tab = jax.device_put(
        f_nue_scat_tab, device=gpus[0]
    )
    f_numu_scat_tab = jax.device_put(
        f_numu_scat_tab, device=gpus[0]
    )

    f_nue_ann_tab = jax.device_put(
        f_nue_ann_tab, device=gpus[0]
    )
    f_numu_ann_tab = jax.device_put(
        f_numu_ann_tab, device=gpus[0]
    )
except: 
    pass


######################
# Standard EM Sector #
######################

def rho_EM_std(T_g, mu=0, LO=True, NLO=True): 
    """
    Total energy density of EM-coupled SM fluids.

    Parameters
    ----------
    T_g : float
        Photon temperature in MeV.
    mu : float, optional
        Parameter added for syntax consistency--does not impact function
        behavior.  Defaults to 0.
    LO : bool
        True includes leading order QED corrections to the energy density.
        Defaults to 'True'.
    NLO : bool
        True includes next-to-leading order QED corrections to the 
        energy density.  Defaults to 'True'.
        
    Returns
    -------
    float
        Units of MeV^4. 
    """ 

    corr_QED = (
        -jnp.interp(
            T_g, jnp.flip(P_QED_tab[:,0]), 
            jnp.flip(LO*P_QED_tab[:,1]+NLO*P_QED_tab[:,2])
        ) 
        + T_g*jnp.interp(
            T_g, jnp.flip(dPdT_QED_tab[:,0]),
            jnp.flip(LO*dPdT_QED_tab[:,1]+NLO*dPdT_QED_tab[:,2])
        )
    )

    return (
        rho_massless_BE(T_g, 0., 2)  + rho_massive_FD(T_g, 0., const.me, 4) 
        + corr_QED
    )

rho_EM_std_v = vmap(rho_EM_std, in_axes=0)

def p_EM_std(T_g, mu=0, LO=True, NLO=True): 
    """
    Total pressure of EM-coupled SM fluids.

    Parameters
    ----------
    T_g : float 
        Temperature of massive particle species in MeV.
    mu : float, optional
        Parameter added for syntax consistency--does not impact function
        behavior.  Defaults to 0.
    LO : bool
        True includes leading order QED corrections to the pressure.
        Defaults to 'True'.
    NLO : bool
        True includes next-to-leading order QED corrections to the 
        pressure.  Defaults to 'True'.
        
    Returns
    -------
    float 
        Units of MeV^4. 
    """ 

    corr_QED = jnp.interp(
        T_g, jnp.flip(P_QED_tab[:,0]),
        jnp.flip(LO*P_QED_tab[:,1] + NLO*P_QED_tab[:,2])
    )

    return (
        p_massless_BE(T_g, 0., 2) + p_massive_FD(T_g, 0., const.me, 4) 
        + corr_QED
    )

p_EM_std_v = vmap(p_EM_std, in_axes=0)

def rho_plus_p_EM_std(T_g, mu=0, LO=True, NLO=True): 
    """
    Sum of energy densities and pressures of all EM-coupled SM fluids.

    Parameters
    ----------
    T_g : float 
        Photon temperature in MeV.
    mu : float, optional
        Parameter added for syntax consistency--does not impact function
        behavior.  Defaults to 0.
    LO : bool
        True includes leading order QED corrections to the energy density 
        and pressure.  Defaults to 'True'.
    NLO : bool
        True includes next-to-leading order QED corrections to the 
        energy density and pressure.  Defaults to 'True'.
        
    Returns
    -------
    float 
        Units of MeV^4. 
    """ 

    corr_QED = T_g * jnp.interp(
        T_g, jnp.flip(dPdT_QED_tab[:,0]),
        jnp.flip(LO*dPdT_QED_tab[:,1] + NLO*dPdT_QED_tab[:,2])
    )
    
    return (
        4/3 * rho_massless_BE(T_g, 0., 2) + rho_massive_FD(T_g, 0., 0.511, 4) 
        + p_massive_FD(T_g, 0., 0.511, 4) + corr_QED
    )

def T_g(rho_g):
    """
    Photon temperature from photon energy density. 

    Parameters
    ----------
    rho_g : float or array
        Energy density of photons. 
    
    Returns
    -------
    float or array
        Same units as (rho_g)**0.25
    """
    return (30*rho_g/(2*jnp.pi**2))**(1/4)

drho_EM_dT_g_std = grad(rho_EM_std, argnums=0)

############################
# Standard Neutrino Sector #
############################ 

def rho_nue_std(T_nue, mu_nue=0.): 
    """
    Total energy density of electron neutrinos.

    Parameters
    ----------
    T_nue : float 
        Electron neutrino temperature in MeV.
    mu_nue : float, optional
        Chemical potential of electron neutrinos in MeV.  Defaults to 0.
        
    Returns
    -------
    float
        Units of MeV^4
    """ 

    return rho_massless_FD(T_nue, mu_nue, 2)

def p_nue_std(T_nue, mu_nue=0.): 
    """
    Total pressure of electron neutrinos.

    Parameters
    ----------
    T_nue : float
        Electron neutrino temperature in MeV.
    mu_nue : float, optional
        Chemical potential of electron neutrinos in MeV.  Defaults to 0.
        
    Returns
    -------
    float
        Units of MeV^4. 
    """ 

    return rho_massless_FD(T_nue, mu_nue, 2) / 3

def n_nue_std(T_nue, mu_nue=0.): 
    """
    Total number density of electron neutrinos.

    Parameters
    ----------
    T_nue : float
        Electron neutrino temperature in MeV.
    mu_nue : float, optional
        Chemical potential of electron neutrinos in MeV.  Defaults to 0.
        
    Returns
    -------
    float
        Units of MeV^3. 
    """ 

    return n_massless_FD(T_nue, mu_nue, 2) 

def rho_numt_std(T_numt, mu_numt=0.): 
    """
    Total energy density of mu, tau neutrinos.

    Parameters
    ----------
    T_nue : float
        Mu, tau neutrino temperature in MeV.
    mu_nue : float, optional
        Chemical potential of mu,tau neutrinos in MeV. Defaults to 0.
        
    Returns
    -------
    float
        Units of MeV^4. 
    """ 

    return 2 * rho_massless_FD(T_numt, mu_numt, 2)

def p_numt_std(T_numt, mu_numt=0.): 
    """
    Total pressure of mu, tau neutrinos.

    Parameters
    ----------
    T_nue : float
        Mu, tau neutrino temperature in MeV.
    mu_nue : float, optional
        Chemical potential of mu,tau neutrinos in MeV.  Defaults to 0.
        
    Returns
    -------
    float
        Units of MeV^4. 
    """ 

    return 2 * rho_massless_FD(T_numt, mu_numt, 2) / 3

def n_numt_std(T_numt, mu_numt=0.): 
    """
    Total number density of mu, tau neutrinos.

    Parameters
    ----------
    T_nue : float
        Mu, tau neutrino temperature in MeV.
    mu_nue : float, optional
        Chemical potential of mu,tau neutrinos in MeV.  Defaults to 0.
        
    Returns
    -------
    float
        Units of MeV^4. 
    """ 

    return 2 * n_massless_FD(T_numt, mu_numt, 2)

def T_nu(rho_nu):
    """
    Neutrino temperature given an energy density.

    Parameters
    ----------
    rho_nu : float or array
        Neutrino energy density in MeV^4.
        
    Returns
    -------
    float or array
        Units of MeV. 
    """ 
    return (8./7.*30*rho_nu/(2*jnp.pi**2))**(1/4)

drho_nue_dT_nue_std  = grad(rho_nue_std, argnums=0)
drho_nue_dmu_nue_std = grad(rho_nue_std, argnums=1)
dn_nue_dT_nue_std    = grad(n_nue_std, argnums=0)
dn_nue_dmu_nue_std   = grad(n_nue_std, argnums=1)

drho_numt_dT_numt_std  = grad(rho_numt_std, argnums=0)
drho_numt_dmu_numt_std = grad(rho_numt_std, argnums=1)
dn_numt_dT_numt_std    = grad(n_numt_std, argnums=0)
dn_numt_dmu_numt_std   = grad(n_numt_std, argnums=1)


def collision_terms_std(
    T_g, T_nue, T_numt, mu_nue=0., mu_numt=0., 
    decoupled=False, use_FD=True, collision_me=True
): 
    """
    Energy and number density transfer rate between EM and neutrino sector 
    relevant for incomplete neutrino decoupling.

    Parameters
    ----------
    T_g : array_like 
        Photon temperature in MeV.
    T_nue : array_like
        Electron neutrino temperature in MeV.
    T_numt : array_like
        Mu, tau neutrino temperature in MeV.
    mu_nue : float, optional
        Chemical potential of electron neutrinos in MeV.
        Defaults to 0.
    mu_numt : float, optional
        Chemical potential of mu, tau neutrinos in MeV.
        Defaults to 0.
    decoupled : bool, optional
        Neutrinos are assumed to be completely decoupled if True. 
    use_FD : bool, optional
        Fermi-Dirac distribution used for neutrinos if True. 
    collision_me : bool, optional
        Finite electron mass if true. 
        
    Returns
    -------
    tuple
        (C_rho_nue, C_rho_numu, C_n_nue, C_n_numu) for the energy density 
        transfer rate (in MeV^4/s) to nu_e and (nu_mu, nu_tau), followed by 
        the number density transfer rate (in MeV^3/s) to nu_e and (nu_mu, 
        nu_tau)

    
    """ 

    f_n, f_a, f_s = lax.cond(
        decoupled, 
        lambda _: (0., 0., 0.), 
        lambda _: lax.cond(
            use_FD, lambda _: (0.852, 0.884, 0.829), 
            lambda _: (1., 1., 1.), 0.
        ), 
        0.
    )

    def G(T_1, mu_1, T_2, mu_2): 
        
        return (
            32 * f_a * (
                T_1**9 * jnp.exp(2 * mu_1 / T_1) 
                - T_2**9 * jnp.exp(2 * mu_2 / T_2)
            ) 
            + 56 * f_s * jnp.exp(2 * mu_1 / T_1) * jnp.exp(2 * mu_2 / T_2) *(   
                T_1**4 * T_2**4 * (T_1 - T_2)
            )
        )

    def G_nue_with_me(T_1, mu_1, T_2, mu_2):

        def interp_f(f_tab): 

            return jnp.interp(
                T_1, f_tab[:,0], f_tab[:,1], left=f_tab[0,1], right=f_tab[-1,1]
            )

        f_nue_ann  = lax.cond(
            collision_me, interp_f, lambda _: 1., f_nue_ann_tab
        )
        f_nue_scat = lax.cond(
            collision_me, interp_f, lambda _: 1., f_nue_scat_tab
        )
        
        return (
            32 * f_a * f_nue_ann * (
                T_1**9 * jnp.exp(2 * mu_1 / T_1) 
                - T_2**9 * jnp.exp(2 * mu_2 / T_2)
            ) 
            + 56 * f_s * f_nue_scat * (
                jnp.exp(2 * mu_1 / T_1) * jnp.exp(2 * mu_2 / T_2) 
                * T_1**4 * T_2**4 * (T_1 - T_2)
            )
        )

    def G_numt_with_me(T_1, mu_1, T_2, mu_2): 

        def interp_f(f_tab): 

            return jnp.interp(
                T_1, f_tab[:,0], f_tab[:,1], left=f_tab[0,1], right=f_tab[-1,1]
            )

        f_numt_ann  = lax.cond(
            collision_me, interp_f, lambda _: 1., f_numu_ann_tab
        )
        f_numt_scat = lax.cond(
            collision_me, interp_f, lambda _: 1., f_numu_scat_tab
        )
        
        return (
            32 * f_a * f_numt_ann * (
                T_1**9 * jnp.exp(2 * mu_1 / T_1) 
                - T_2**9 * jnp.exp(2 * mu_2 / T_2)
            ) 
            + 56 * f_s * f_numt_scat * (
                jnp.exp(2 * mu_1 / T_1) * jnp.exp(2 * mu_2 / T_2) 
                * T_1**4 * T_2**4 * (T_1 - T_2)
            )
        )

    geL  = const.geL
    geR  = const.geR
    gmuL = const.gmuL
    gmuR = const.gmuR

    # Units MeV^4 s^-1
    C_rho_nue = const.GF**2 / jnp.pi**5 * (
        4 * (geL**2 + geR**2) * G_nue_with_me(T_g, 0., T_nue, mu_nue) 
        + 2 * G(T_numt, mu_numt, T_nue, mu_nue)
    ) / const.hbar

    # Units MeV^4 s^-1
    C_rho_numu = const.GF**2 / jnp.pi**5 * (
        4 * (gmuL**2 + gmuR**2) * G_numt_with_me(T_g, 0., T_numt, mu_numt) 
        - G(T_numt, mu_numt, T_nue, mu_nue)
    ) / const.hbar 

    # Units MeV^3 s^-1
    C_n_nue = 8 * f_n * const.GF**2 / jnp.pi**5 * (
        4 * (geL**2 + geR**2) 
        * (T_g**8 - T_nue**8 * jnp.exp(2 * mu_nue / T_nue))
        + 2 * (
            T_numt**8 * jnp.exp(2 * mu_numt / T_numt) 
            - T_nue**8 * jnp.exp(2 * mu_nue / T_nue)
        )
    ) / const.hbar

    # Units MeV^3 s^-1
    C_n_numu = 8 * f_n * const.GF**2 / jnp.pi**5 * (
        4 * (gmuL**2 + gmuR**2) 
        * (T_g**8 - T_nue**8 * jnp.exp(2 * mu_numt / T_numt))
        - (
            T_numt**8 * jnp.exp(2 * mu_numt / T_numt) 
            - T_nue**8 * jnp.exp(2 * mu_nue / T_nue)
        )
    ) / const.hbar

    return C_rho_nue, C_rho_numu, C_n_nue, C_n_numu