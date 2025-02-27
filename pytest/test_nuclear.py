import pytest
import jax.numpy as jnp
import numpy as np

from linx.nuclear import NuclearRates
from linx.const import me, mn, mp, md, mt, mHe3, ma, mLi7, mBe7


def test_nuclear_rates_initialization():
    """Test initialization of NuclearRates with different parameters."""
    # Default initialization
    nr_default = NuclearRates()
    
    # With custom parameters
    nr_custom = NuclearRates(max_i_species=8, interp_type='log')
    
    # Check that the objects were created
    assert isinstance(nr_default, NuclearRates)
    assert isinstance(nr_custom, NuclearRates)
    
    # Check that parameters were set correctly
    assert nr_custom.max_i_species == 8
    assert nr_custom.interp_type == 'log'


def test_nuclear_rates_reaction_lists():
    """Test that the reaction lists are properly populated."""
    nr = NuclearRates()
    
    # Check that reactions are loaded
    assert len(nr.reactions) > 0
    assert len(nr.reactions_names) > 0
    
    # Check that all reactions have input and output states
    for reaction_name in nr.reactions_names:
        assert reaction_name in nr.in_states
        assert reaction_name in nr.out_states
        assert len(nr.in_states[reaction_name]) > 0
        assert len(nr.out_states[reaction_name]) > 0


def test_nuclear_rates_symmetry_factors():
    """Test that symmetry factors are set appropriately."""
    nr = NuclearRates()
    
    # Check that all reactions have symmetry factors
    for reaction_name in nr.reactions_names:
        assert reaction_name in nr.frwrd_symmetry_fac
        assert reaction_name in nr.bkwrd_symmetry_fac
        
        # Symmetry factors should be positive
        assert nr.frwrd_symmetry_fac[reaction_name] > 0
        assert nr.bkwrd_symmetry_fac[reaction_name] > 0


def test_nuclear_rates_rate_parameters():
    """Test that rate parameters are properly set."""
    nr = NuclearRates()
    
    # Check that all reactions have rate parameters
    for reaction_name in nr.reactions_names:
        assert reaction_name in nr.frwrd_rate_param
        assert reaction_name in nr.bkwrd_rate_param
        
        # Check that rate functions accept temperature input
        T = jnp.array([0.1])  # Temperature in MeV
        assert nr.frwrd_rate_param[reaction_name](T) is not None
        assert nr.bkwrd_rate_param[reaction_name](T) is not None


def test_nuclear_rates_by_particle():
    """Test that reactions are properly indexed by particle."""
    nr = NuclearRates()
    
    # Check that all particles have associated reactions
    for i in range(nr.max_i_species):
        assert i in nr.frwrd_reaction_by_particle
        assert i in nr.bkwrd_reaction_by_particle


def test_specific_reactions():
    """Test specific important reactions are included."""
    nr = NuclearRates()
    
    # Important reactions to check for
    important_reactions = [
        'npdg',    # n + p -> d + gamma
        'dpHe3g',  # d + p -> He3 + gamma
        'tpag',    # t + p -> a + gamma
        'He3ntp',  # He3 + n -> t + p
        'ddtp',    # d + d -> t + p
        'ddHe3n'   # d + d -> He3 + n
    ]
    
    # Check that these reactions are included
    for reaction in important_reactions:
        assert reaction in nr.reactions_names


def test_temperature_dependence():
    """Test that rates have expected temperature dependence."""
    nr = NuclearRates()
    
    # Test at different temperatures
    temps = jnp.array([0.001, 0.01, 0.1])
    
    # Check that important rates increase with temperature
    for reaction_name in ['npdg', 'dpHe3g', 'ddtp']:
        rates = jnp.array([nr.frwrd_rate_param[reaction_name](T) for T in temps])
        assert jnp.all(jnp.diff(rates) > 0)


def test_reverse_rates():
    """Test that reverse rates are correctly calculated from forward rates."""
    nr = NuclearRates()
    T = jnp.array([0.1])  # 0.1 MeV
    
    # For reactions without gammas, check detailed balance at high T
    for reaction_name in nr.reactions_names:
        # Skip reactions with gammas
        if 'g' in reaction_name:
            continue
            
        # Get rates
        frwrd_rate = nr.frwrd_rate_param[reaction_name](T)
        bkwrd_rate = nr.bkwrd_rate_param[reaction_name](T)
        
        # Both should be positive
        assert frwrd_rate > 0
        assert bkwrd_rate > 0
        
        # At high enough T, rates should be comparable
        if T > 0.05:
            # This is a rough check - real detailed balance would need Q-values
            assert 0.01 < frwrd_rate / bkwrd_rate < 100