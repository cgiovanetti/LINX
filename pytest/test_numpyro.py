import sys
import os
import pytest
# Add absolute path to the scripts directory
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))

from run_numpyro import run


@pytest.mark.slow
def test_run_numpyro():
    try:
        run(bbn_only=True, n_steps_svi=5, n_warmup_mcmc=5, n_samples_mcmc=5, n_chains=1)
    except Exception as e:
        pytest.fail(f"run_numpyro.run() raised an exception: {e}")