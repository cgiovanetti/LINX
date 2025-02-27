import pytest
import jax.numpy as jnp
import numpy as np

from linx.background import BackgroundModel
import linx.const as const


def test_background_initialization():
    """Test initialization of BackgroundModel with different parameters."""
    # Default initialization
    bg_default = BackgroundModel()
    
    # With custom parameters
    bg_custom = BackgroundModel(
        decoupled=True, 
        use_FD=False, 
        collision_me=False, 
        LO=False, 
        NLO=False
    )
    
    # Check that the objects were created
    assert isinstance(bg_default, BackgroundModel)
    assert isinstance(bg_custom, BackgroundModel)
    
    # Check that parameters were set correctly
    assert bg_custom.decoupled == True
    assert bg_custom.use_FD == False
    assert bg_custom.collision_me == False
    assert bg_custom.LO == False
    assert bg_custom.NLO == False


def test_background_evolution_zero_delta_neff():
    """Test background evolution with Delta N_eff = 0."""
    bg = BackgroundModel()
    
    # Run with standard parameters and Delta N_eff = 0
    delta_neff = 0.0
    result = bg(delta_neff)
    
    # Unpack results
    t_vec, a_vec, rho_g_vec, rho_nu_vec, rho_extra_vec, P_extra_vec, Neff_vec = result
    
    # Check that vectors have the expected shape
    assert t_vec.shape == a_vec.shape
    assert t_vec.shape == rho_g_vec.shape
    assert t_vec.shape == rho_nu_vec.shape
    assert t_vec.shape == rho_extra_vec.shape
    assert t_vec.shape == P_extra_vec.shape
    assert t_vec.shape == Neff_vec.shape
    
    # For Delta N_eff = 0, extra energy density should be zero
    assert jnp.allclose(rho_extra_vec, 0.0)
    assert jnp.allclose(P_extra_vec, 0.0)
    
    # N_eff should be close to 3 (standard model value)
    assert jnp.isclose(Neff_vec[-1], 3.0, rtol=0.05)
    
    # Scale factor should be strictly increasing
    assert jnp.all(jnp.diff(a_vec) > 0)
    
    # Energy densities should be strictly decreasing
    assert jnp.all(jnp.diff(rho_g_vec) < 0)
    assert jnp.all(jnp.diff(rho_nu_vec) < 0)


def test_background_evolution_positive_delta_neff():
    """Test background evolution with positive Delta N_eff."""
    bg = BackgroundModel()
    
    # Run with Delta N_eff = 0.5
    delta_neff = 0.5
    result = bg(delta_neff)
    
    # Unpack results
    t_vec, a_vec, rho_g_vec, rho_nu_vec, rho_extra_vec, P_extra_vec, Neff_vec = result
    
    # Extra energy density should be positive
    assert jnp.all(rho_extra_vec > 0)
    assert jnp.all(P_extra_vec > 0)
    
    # N_eff should be close to 3 + Delta N_eff
    assert jnp.isclose(Neff_vec[-1], 3.0 + delta_neff, rtol=0.05)


def test_background_evolution_with_decoupled():
    """Test background evolution with neutrinos decoupled."""
    # Create model with decoupled neutrinos
    bg_decoupled = BackgroundModel(decoupled=True)
    
    # Run with standard parameters
    delta_neff = 0.0
    result_decoupled = bg_decoupled(delta_neff)
    
    # Unpack results
    t_vec, a_vec, rho_g_vec, rho_nu_vec, rho_extra_vec, P_extra_vec, Neff_vec = result_decoupled
    
    # When neutrinos are decoupled, their temperature should scale as a^-1
    # This means their energy density should scale as a^-4
    T_nu_ratio = (rho_nu_vec[1:] / rho_nu_vec[:-1])**(1/4)
    a_ratio = a_vec[:-1] / a_vec[1:]
    
    # These ratios should be very close
    assert jnp.allclose(T_nu_ratio, a_ratio, rtol=1e-2)


def test_background_final_temperature():
    """Test that the final temperature matches the expected value."""
    bg = BackgroundModel()
    
    # Run with standard parameters
    delta_neff = 0.0
    result = bg(delta_neff)
    
    # Unpack results
    t_vec, a_vec, rho_g_vec, rho_nu_vec, rho_extra_vec, P_extra_vec, Neff_vec = result
    
    # Check that integration stops at the specified final temperature
    final_T_ratio = const.T0CMB / const.T_end
    final_a_ratio = a_vec[-1] / a_vec[0]
    
    # The ratio of final/initial scale factor should match the inverse of the temperature ratio
    assert jnp.isclose(final_a_ratio, final_T_ratio, rtol=1e-2)


def test_energy_conservation():
    """Test that total energy is conserved in comoving coordinates."""
    bg = BackgroundModel()
    
    # Run with standard parameters
    delta_neff = 0.0
    result = bg(delta_neff)
    
    # Unpack results
    t_vec, a_vec, rho_g_vec, rho_nu_vec, rho_extra_vec, P_extra_vec, Neff_vec = result
    
    # Calculate total energy density
    rho_tot = rho_g_vec + 3 * rho_nu_vec + rho_extra_vec
    
    # In comoving coordinates, rho * a^4 should be approximately constant
    # when dominated by radiation
    rho_comoving = rho_tot * a_vec**4
    
    # Check that it's roughly constant (it's not exactly constant due to 
    # energy transfer between neutrinos and the electromagnetic plasma)
    variation = jnp.std(rho_comoving) / jnp.mean(rho_comoving)
    assert variation < 0.1  # Less than 10% variation