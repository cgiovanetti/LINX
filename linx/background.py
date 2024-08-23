import jax
import jax.numpy as jnp
from jax import vmap

import equinox as eqx

from diffrax import diffeqsolve, ODETerm, Tsit5, PIDController, SaveAt, DiscreteTerminatingEvent 

import linx.thermo as thermo
import linx.const as const 

rho_massless_BE_v = vmap(
    thermo.rho_massless_BE, in_axes=(0, None, None)
)
rho_massless_FD_v = vmap(
    thermo.rho_massless_FD, in_axes=(0, None, None)
)

class BackgroundModel(eqx.Module): 
    """Background model.

    Attributes
    ----------
    decoupled : bool, optional
        Whether neutrinos are always decoupled. Default is ``False``.
    use_FD : bool, optional
        Whether to use Fermi-Dirac statistics for neutrinos, or a Maxwell-Boltzmann distribution. Default is ``True``.
    collision_me : bool, optional
        Finite electron mass correction in energy transfer collision terms. Default is ``True``.
    LO : bool, optional
        Whether to use leading order QED correction. Default is ``True``.
    NLO : bool, optional
        Whether to use next-to-leading order QED correction. Default is True.
    """

    decoupled : bool
    use_FD : bool
    collision_me : bool
    LO : bool
    NLO : bool

    def __init__(self, decoupled=False, use_FD=True, collision_me=True, LO=True, NLO = True): 

        self.decoupled = decoupled
        self.use_FD = use_FD
        self.collision_me = collision_me 
        self.LO = LO
        self.NLO = NLO

    @eqx.filter_jit
    def __call__(
        self, Delt_Neff_init, T_start=const.T_start, 
        T_end=const.T_end, rtol=1e-8, atol=1e-10, 
        solver=Tsit5(), max_steps=512
    ): 
        """ Calculate thermodynamics given a target delta Neff.

        Parameters
        ----------
        Delt_Neff_init : float
            Target delta Neff.  Can be positive or negative.  
        T_EM_init : float
            Initial EM (and neutrino) temperature. Default is const.T_start. 
        T_EM_end : float 
            Final EM temperature to terminate integration at. Default is
            const.T_end. 
        rtol : float, optional
            Relative tolerance of the abundance solver. Default is 1e-8.  
        atol : float, optional
            Absolute tolerance of the abundance solver. Default is 1e-10. 
        max_steps : int, optional
            Maximum number of steps taken by the solver. Default is 4096. 
            Increasing this slows down the code, while decreasing this could 
            mean that the solver cannot complete the solution. 
        solver : Diffrax ODE solver 
            The Diffrax ODE solver to use. A stiff solver is recommended. 
            Default is the 3rd order Kvaerno solver. 

        Returns
        -------
        t_vec : array_like
            Times at which thermodynamics are saved.
        a_vec : array_like
            Scale factor at each point in time.
        rho_g_vec : array_like
            Energy density of photons at each point in time.
        rho_nu_vec : array_like
            Energy density of one species of neutrinos at each point in time.
        rho_extra_vec : array_like
            Energy density of extra species at each point in time.
        """
        print('`\\         /´  ||||        ||||  |||||     ||||  ||||   ||||')
        print(' /\\_______/\\   ||||        ||||  |||||||   ||||   |||| ||||') 
        print(' ) __` ´__ (   ||||        ||||  |||| |||| ||||    |||||||')
        print('/  `-|_|-´  \\  ||||        ||||  ||||  |||| |||    ||||||| ')
        print('/   (_x_)   \\  ||||||||||  ||||  ||||   |||||||   |||| ||||')
        print('  )  `-´  (    ||||||||||  ||||  ||||    ||||||  ||||   ||||')
        print(' ')

        print('Compiling thermodynamics model...')

        lna_init = 0. 
        T_EM_init = T_start
        T_nu_init = T_EM_init 

        rho_extra_init = (7/8) * (4/11)**(4/3) * (
            thermo.rho_massless_BE(T_EM_init, 0., 2)
        ) * Delt_Neff_init

        Y0 = (lna_init, T_EM_init, T_nu_init)
        
        def T_EM_check(state, **kwargs): 

            return state.y[1] < T_end
            
        sol = diffeqsolve(
            ODETerm(self.dY), solver, args=(lna_init, rho_extra_init),
            t0=0., t1=jnp.inf, dt0=None, y0=Y0, 
            saveat=SaveAt(steps=True), discrete_terminating_event = DiscreteTerminatingEvent(T_EM_check),
            stepsize_controller = PIDController(
                rtol=rtol, atol=atol
            ), 
            max_steps=max_steps
        )

        a_vec = jnp.exp(sol.ys[0])
        rho_g_vec = rho_massless_BE_v(sol.ys[1], 0., 2) 
        rho_nu_vec = rho_massless_FD_v(sol.ys[2], 0., 2)
        
        # These vectors always have max_steps entries so that jit and grad 
        # work but the solver stops before hitting max_steps. Find the last 
        # legitimate step made by the solver, when T_g drops below T_end.

        last_step_ind = jnp.max(
            jnp.argwhere(
                sol.ys[1] < T_end,
                size=512
            )[:,0]
        )


        # Set every step after this in all vectors to be identical, 
        # so that there is no effect on interpolation. 
        t_vec = jnp.where(sol.ts == jnp.inf, sol.ts[last_step_ind], sol.ts)
        a_vec = jnp.where(a_vec == jnp.inf, a_vec[last_step_ind], a_vec)
        rho_g_vec = jnp.where(
            rho_g_vec == jnp.inf, rho_g_vec[last_step_ind], rho_g_vec
        )
        rho_nu_vec = jnp.where(
            rho_nu_vec == jnp.inf, rho_nu_vec[last_step_ind], rho_nu_vec
        )

        # Rescale a so that the present day CMB temperature is correct.
        final_a = const.T0CMB / (
            sol.ys[1][last_step_ind]
        )
        
        a_vec  *= final_a / a_vec[-1]

        # Trivial relation between rho_extra and a. 
        rho_extra_vec = rho_extra_init * (a_vec[0]**4 / a_vec**4)

        P_extra_vec = rho_extra_vec / 3

        T_g_vec  = thermo.T_g(rho_g_vec)
        rho_tot_vec = (
            thermo.rho_EM_std_v(T_g_vec) + 3 * rho_nu_vec + rho_extra_vec 
        ) 

        Neff_vec = thermo.N_eff(rho_tot_vec, rho_g_vec)

        return (
            t_vec, a_vec, rho_g_vec, rho_nu_vec, 
            rho_extra_vec, P_extra_vec, Neff_vec
        )
    
    @eqx.filter_jit
    def dY(self, t, Y, args): 

        lna, T_g, T_nu = Y
        lna_init, rho_extra_init = args

        rho_EM = thermo.rho_EM_std(T_g, LO=self.LO, NLO=self.NLO)
        rho_plus_p_EM = thermo.rho_plus_p_EM_std(T_g, LO=self.LO, NLO=self.NLO)
        drho_EM_dT_g = thermo.drho_EM_dT_g_std(T_g, LO=self.LO, NLO=self.NLO)

        rho_nu = 3*thermo.rho_nue_std(T_nu)
        rho_plus_p_nu = (4/3) * rho_nu
        drho_nu_dT_nu = 3*thermo.drho_nue_dT_nue_std(T_nu)

        rho_extra = rho_extra_init * jnp.exp(lna_init)**4 / jnp.exp(lna)**4 

        H = thermo.Hubble(rho_EM + rho_nu + rho_extra)

        C_rho_nue, C_rho_numu, _, _ = thermo.collision_terms_std(
            T_g, T_nu, T_nu, decoupled=self.decoupled, use_FD=self.use_FD, collision_me=self.collision_me
        )

        drho_EM_dt = -3 * H * rho_plus_p_EM - C_rho_nue - 2*C_rho_numu
        drho_nu_dt  = -3 * H * rho_plus_p_nu + C_rho_nue + 2*C_rho_numu

        dT_g_dt = drho_EM_dt / drho_EM_dT_g
        dT_nu_dt = drho_nu_dt / drho_nu_dT_nu

        return H, dT_g_dt, dT_nu_dt 
