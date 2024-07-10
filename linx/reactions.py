import os

import numpy as np

import jax
import jax.numpy as jnp
import equinox as eqx

file_dir = os.path.dirname(__file__)

class Reaction(eqx.Module): 
    """
    Nuclear reaction. 

    Attributes
    ----------
    name : str
        Name of the reaction. 
    in_states : tuple
        Particles on the LHS of the reaction. Convention is 0:n, 1:p, 
        2:d, 3:t, 4:He3, 5:a, 6:Li7, 7:Be7, 8: He6, 9: Li6, 10: Li8, 11:B8. 
    out_states : tuple
        Particles on the RHS of the reaction. 
    frwrd_symmetry_fac : float
        Symmetry factor associated with forward direction. 
    bkwrd_symmetry_fac : float
        Symmetry factor associated with backward direction. 
    alpha : float
        Coefficient for relation between forward and backward reaction. 
        Dimensionful. 
    beta : float
        Coefficient, same as above, dimensionless. 
    gamma : float
        Coefficient, same as above, dimensionless. 
    T9_vec : list
        (T/10^9 K), abscissa for reaction rate parameter with spline 
        fit. 
    mu_median_vec : list
        Median reaction rate parameter with spline fit. 
    expsigma_vec : list
        Exponential uncertainty of reaction rate parameter with spline fit. 
    interp_type : str
        Interpolation for spline fit. Either 'linear' or 'log'. 
    frwrd_rate_param_func : callable
        Forward rate parameter function, if no spline fit. 

    Notes
    -----
    The rate functions are either <sigma v> or <sigma v^2> divided by 
    (1 amu)^(N_in-1)) for each reaction, units (cm^3/s/g or cm^6/s/g^2). 

    """

    name : str
    in_states : tuple 
    out_states : tuple 
    frwrd_symmetry_fac : float
    bkwrd_symmetry_fac : float
    alpha : float
    beta : float
    gamma : float
    T9_vec : list 
    mu_median_vec : list
    expsigma_vec : list
    interp_type : str 
    frwrd_rate_param_func : callable 

    def __init__(
        self, name, in_states, out_states, alpha, beta, gamma, 
        spline_data=None, frwrd_rate_param_func=None, interp_type=None
    ): 
        """

        Parameters
        ----------
        name : str
            Name of the reaction. 
        in_states : tuple
            Particles on the LHS of the reaction. Convention is 0:n, 1:p, 
            2:d, 3:t, 4:He3, 5:a, 6:Li7, 7:Be7, 8: He6, 9: Li6, 10: Li8, 11:B8. 
        out_states : tuple
            Particles on the RHS of the reaction. 
        alpha : float
            Coefficient for relation between forward and backward reaction. 
            Dimensionless. 
        beta : float
            Coefficient, same as above, dimensionless. 
        gamma : float
            Coefficient, same as above, dimensionless. 
        spline_data : string, optional
            If provided, reads data from 'data/nuclear_rates/'+spline_data
            in the code. Otherwise, frwrd_rate_param_func must be specified. 
        frwrd_rate_param_func : callable, optional
            If provided, is a function that returns the forward rate parameter.
            Takes two arguments, `T` for EM temperature in K and `p` for 
            rescaling of the rate. 

        Notes
        -----
        For `spline_data`, data file should contain three columns: first column 
        `T9` gives the EM temperature in units of 1e9 K, second column `mu` 
        gives the mean rate, and third column `expsigma`, with `log(expsigma)` 
        giving the uncertainty in log of the rate. 

        In other words, we take log(<sigma v>) = log(mu) + p*log(expsigma), 
        where p follows a Gaussian distribution, or equivalently <sigma v> = 
        mu * exp(p*log(expsigma)). 

        The rates in `spline_data` or `frwrd_rate_param_func` are <sigma v> or 
        <sigma v^2> divided by (1 amu)^(N_in-1)) for each reaction, units 
        (cm^3/s/g or cm^6/s/g^2). 
        """
        self.name = name
        self.in_states = in_states
        self.out_states = out_states

        multiplicity_in = jnp.array(
            [self.in_states.count(i) for i in set(self.in_states)]
        )
        self.frwrd_symmetry_fac = jnp.prod(1. / multiplicity_in)

        multiplicity_out = jnp.array(
            [self.out_states.count(i) for i in set(self.out_states)]
        )
        self.bkwrd_symmetry_fac = jnp.prod(1. / multiplicity_out)

        self.alpha = alpha 
        self.beta = beta 
        self.gamma = gamma 

        self.interp_type = interp_type 

        self.T9_vec = None
        self.mu_median_vec = None
        self.expsigma_vec = None
        self.frwrd_rate_param_func = None 

        if spline_data: 
            self.T9_vec, self.mu_median_vec, self.expsigma_vec = np.loadtxt(
                file_dir+'/data/nuclear_rates/'+spline_data,
                unpack=True 
            )
            try:
                gpus = jax.devices('gpu')
                self.T9_vec = jax.device_put(self.T9_vec, device=gpus[0])
                self.mu_median_vec = jax.device_put(
                    self.mu_media_vec, device=gpus[0]
                )
                self.expsigma_vec = jax.device_put(
                    self.expsigma_vec, device=gpus[0]
                )
            except: 
                pass

        elif frwrd_rate_param_func is not None: 

            self.frwrd_rate_param_func = frwrd_rate_param_func 

        else: 

            return TypeError('Must include spline data points or analytic \
                             function for rates.')

    @eqx.filter_jit
    def frwrd_rate_param(self, T, p):
        """
        Forward rate parameter. 

        Parameters
        ----------
        T : float
            Temperature in K. 
        p : float
            Rescaling parameter for expsigma. 
        interp_type : str, optional
            Interpolation type for spline data. Either 'linear' or 'log'. 

        Returns
        -------
        float

        Notes
        -----
        We take log(<sigma v>) = log(mu) + p*log(expsigma), 
        where p follows a Gaussian distribution, or equivalently <sigma v> = 
        mu * exp(p*log(expsigma)). 

        The rate here is either <sigma v> or <sigma v^2> divided by 
        (1 amu)^(N_in-1)) for each reaction, units (cm^3/s/g or cm^6/s/g^2). 
        """

        T9 = T*1e-9

        if self.T9_vec is not None: 

            rate_vec = self.mu_median_vec * jnp.exp(
                 p * jnp.log(self.expsigma_vec)
            )

            if self.interp_type == 'linear': 

                return jnp.interp(
                    T9, self.T9_vec, rate_vec, left=0., right=0.
                )
                
            elif self.interp_type == 'log': 
                
                return jnp.exp(jnp.interp(
                    jnp.log(T9), jnp.log(self.T9_vec), jnp.log(rate_vec), 
                    left=0., right=0.
                ))

        else: 

            return self.frwrd_rate_param_func(T, p)

    @eqx.filter_jit
    def bkwrd_rate_param(self, T, p): 
        """
        Backward rate parameter. 

        Parameters
        ----------
        T : float
            Temperature in K. 
        p : float
            Rescaling parameter for expsigma. 
        interp_type : str, optional
            Interpolation type for spline data. Either 'linear' or 'log'. 

        Returns
        -------
        float

        Notes
        -----
        We take log(<sigma v>) = log(mu) + p*log(expsigma), 
        where p follows a Gaussian distribution, or equivalently <sigma v> = 
        mu * exp(p*log(expsigma)). 

        The rate here is either <sigma v> or <sigma v^2> divided by 
        (1 amu)^(N_in-1)) for each reaction, units (cm^3/s/g or cm^6/s/g^2). 
        """
        T9 = T*1e-9

        return self.alpha*T9**self.beta*jnp.exp(self.gamma/T9) * (
            self.frwrd_rate_param(T, p)
        )

