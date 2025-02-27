import pytest
import jax.numpy as jnp
import numpy as np
from jax import grad

from linx.thermo import (
    Hubble, N_eff, T_g, rho_massless_BE, rho_massless_FD, 
    rho_plus_p_massless_BE, rho_plus_p_massless_FD,
    rho_EM_std, rho_plus_p_EM_std, drho_EM_dT_g_std,
    rho_nue_std, drho_nue_dT_nue_std
)
import linx.const as const


def test_hubble_function():
    """Test the Hubble parameter calculation."""
    # Test at a specific energy density
    rho_tot = 1.0  # MeV^4
    H = Hubble(rho_tot)
    
    # Check that Hubble parameter is positive
    assert H > 0
    
    # Check scaling with energy density (H ∝ sqrt(ρ))
    rho_tot_2 = 4.0  # MeV^4
    H_2 = Hubble(rho_tot_2)
    assert jnp.isclose(H_2 / H, 2.0, rtol=1e-5)


def test_n_eff_function():
    """Test the Neff parameter calculation."""
    # For standard model with 3 neutrino species, Neff should be close to 3
    rho_g = 1.0  # MeV^4
    rho_nu = 3 * (7/8) * (4/11)**(4/3) * rho_g  # Standard model neutrino energy density
    rho_tot = rho_g + rho_nu
    
    neff = N_eff(rho_tot, rho_g)
    
    # Check that Neff is close to 3 (within a few percent)
    assert jnp.isclose(neff, 3.0, rtol=0.05)


def test_temperature_conversions():
    """Test temperature conversion functions."""
    # Convert energy density to temperature
    rho = rho_massless_BE(1.0, 0.0, 2)  # Energy density for T=1 MeV
    temp = T_g(rho)
    
    # Temperature should be close to 1 MeV
    assert jnp.isclose(temp, 1.0, rtol=1e-5)


def test_massless_energy_densities():
    """Test energy density calculations for massless particles."""
    T = 1.0  # MeV
    
    # Bose-Einstein statistics (e.g., photons)
    rho_be = rho_massless_BE(T, 0.0, 2)  # g=2 for photons
    
    # Fermi-Dirac statistics (e.g., neutrinos)
    rho_fd = rho_massless_FD(T, 0.0, 1)  # g=1 for a single neutrino species
    
    # Check that BE density is greater than FD for same parameters
    assert rho_be > rho_fd
    
    # Check temperature scaling (ρ ∝ T^4)
    T_2 = 2.0  # MeV
    rho_be_2 = rho_massless_BE(T_2, 0.0, 2)
    assert jnp.isclose(rho_be_2 / rho_be, 16.0, rtol=1e-5)  # T^4 scaling


def test_massless_pressure():
    """Test pressure calculations for massless particles."""
    T = 1.0  # MeV
    
    # For massless particles, p = ρ/3, so ρ+p = 4ρ/3
    rho_be = rho_massless_BE(T, 0.0, 2)
    rho_plus_p_be = rho_plus_p_massless_BE(T, 0.0, 2)
    
    # Check that ρ+p = 4ρ/3
    assert jnp.isclose(rho_plus_p_be, 4*rho_be/3, rtol=1e-5)
    
    # Same for Fermi-Dirac statistics
    rho_fd = rho_massless_FD(T, 0.0, 1)
    rho_plus_p_fd = rho_plus_p_massless_FD(T, 0.0, 1)
    assert jnp.isclose(rho_plus_p_fd, 4*rho_fd/3, rtol=1e-5)


def test_electromagnetic_energy_density():
    """Test energy density calculations for the electromagnetic plasma."""
    T = 1.0  # MeV
    
    # Standard electromagnetic energy density
    rho_em = rho_EM_std(T)
    
    # Should include photons and e+/e- pairs
    rho_photons = rho_massless_BE(T, 0.0, 2)
    assert rho_em > rho_photons  # EM density includes more than just photons
    
    # Test with and without QED corrections
    rho_em_no_qed = rho_EM_std(T, LO=False, NLO=False)
    rho_em_with_qed = rho_EM_std(T, LO=True, NLO=True)
    
    # QED corrections should be small but non-zero
    assert rho_em_with_qed != rho_em_no_qed
    assert jnp.isclose(rho_em_with_qed, rho_em_no_qed, rtol=0.1)  # Within ~10%


def test_electromagnetic_derivatives():
    """Test derivatives of energy density with respect to temperature."""
    T = 1.0  # MeV
    
    # Analytical derivative
    drho_dT = drho_EM_dT_g_std(T)
    
    # Numerical derivative for comparison
    def rho_func(T):
        return rho_EM_std(T)
    
    drho_dT_numerical = grad(rho_func)(T)
    
    # Check that analytical and numerical derivatives are close
    assert jnp.isclose(drho_dT, drho_dT_numerical, rtol=1e-2)


def test_neutrino_energy_density():
    """Test energy density calculations for neutrinos."""
    T = 1.0  # MeV
    
    # Energy density for electron neutrinos
    rho_nu = rho_nue_std(T)
    
    # Should be positive
    assert rho_nu > 0
    
    # Compare with massless Fermi-Dirac result
    rho_nu_massless = rho_massless_FD(T, 0.0, 1)
    assert jnp.isclose(rho_nu, rho_nu_massless, rtol=1e-2)
    
    # Test derivative
    drho_dT = drho_nue_dT_nue_std(T)
    
    # Derivative should be positive (energy density increases with temperature)
    assert drho_dT > 0