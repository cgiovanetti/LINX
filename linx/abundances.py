
import sys
sys.path.append('..')

import jax.numpy as jnp
import equinox as eqx
from diffrax import diffeqsolve, ODETerm, Tsit5, Kvaerno3, PIDController, SaveAt

import linx.nuclear as nucl
import linx.const as const 
from linx.const import ma, me, mn, mp
import linx.weak_rates as wr
import linx.thermo as thermo
from linx.thermo import rho_EM_std_v, p_EM_std_v, nB
from linx.special_funcs import zeta_3 

class AbundanceModel(eqx.Module): 
    """
    Abundance model and BBN abundance prediction. 

    Attributes
    ----------
    nuclear_net : NuclearRates
        Nuclear network to be used for BBN prediction. 
    weak_rates : WeakRates
        Weak rates for n <-> p. 
    species_dict : dict
        Dictionary of species considered in LINX. 
    species_Z : list
        Number of protons in each species. 
    species_N : list
        Number of neutrons in each species. 
    species_A : list
        Atomic mass number of each species. 
    species_excess_mass : list
        Excess mass (mass - A*amu) of each species. 
    species_spin : list
        Spin of each species. 
    species_binding_energy : list
        Binding energy of each species. 
    species_mass : list
        Mass of each species. 
    """
    nuclear_net : nucl.NuclearRates  
    weak_rates : wr.WeakRates 
    species_dict : dict
    species_Z : list
    species_N : list
    species_A : list 
    species_excess_mass : dict
    species_spin : list
    species_binding_energy : list
    species_mass : list

    def __init__(self, nuclear_net, weak_rates=wr.WeakRates()):

        self.nuclear_net = nuclear_net  
        self.weak_rates = weak_rates

        self.species_dict = {
            0:'n', 1:'p', 2:'d', 3:'t', 4:'He3', 5:'a', 6:'Li7', 7:'Be7', 
            8: 'Li6', 9: 'He6', 10: 'Li8', 11:'B8'
        }

        self.species_Z = jnp.array([0, 1, 1, 1, 2, 2, 3, 4, 3, 2, 3, 5])

        self.species_N = jnp.array([1, 0, 1, 2, 1, 2, 4, 3, 3, 4, 5, 3])

        self.species_A = self.species_Z + self.species_N 

        # in MeV
        self.species_excess_mass = jnp.array([
            8071.3171, 7288.9706, 13135.722, 14949.81, 
            14931.218, 2424.9156, 14907.105, 15769.,
            14086.8789, 17592.10, 20945.80, 22921.6
        ]) * 1e-3

        self.species_spin = jnp.array([
            1./2., 1./2., 1., 1./2., 1./2., 0.,
            3./2., 3./2., 1., 0., 2., 2.
        ])

        # in MeV
        self.species_binding_energy = (
            self.species_N * self.species_excess_mass[0]
            + self.species_Z * self.species_excess_mass[1]
            - self.species_excess_mass 
        )

        # in MeV
        self.species_mass = (
            self.species_A * ma + self.species_excess_mass - self.species_Z * me
        )

    @eqx.filter_jit
    def __call__(
        self, rho_g_vec, rho_nu_vec, rho_NP_vec, P_NP_vec, 
        a_vec=None, t_vec=None, 
        eta_fac=jnp.asarray(1.), tau_n_fac = jnp.asarray(1.), 
        nuclear_rates_q=None, 
        Y_i=None, T_start=None, T_end=None, sampling_nTOp=150, 
        rtol=1e-6, atol=1e-9, solver=Kvaerno3(),
        max_steps=4096,
        save_history=False
    ):
        """
        Calculate BBN abundance. 

        Parameters
        ----------
        rho_g_vec : array
            Energy density of photons in MeV^4. 
        rho_nu_vec : array
            Energy density of a single neutrino species in MeV^4 
            (all neutrinos assumed to have the same temperature). 
        rho_NP_vec : array
            Energy density of all new physics particles in MeV^4. 
        P_NP_vec : array
            Pressure of all new physics particles in MeV^4. 
        a_vec : array, optional
            Scale factor. If `None`, will be computed in function. 
        t_vec : array, optional
            Time in seconds. If `None`, will be computed in function. 
        eta_fac : float, optional
            Rescaling factor for baryon-to-photon ratio, 1 for fiducial value 
            in const.eta0 (or const.Omegabh2). 
        tau_n_fac : float, optional
            Rescaling factor for neutron decay lifetime, 1 for fiducial value 
            in const.eta0 (or const.Omegabh2). 
        nuclear_rates_q : array, optional
            p ~ N(0,1) specifies the nuclear rate in its log-normal 
            distribution. If not specified, will be taken to be p = 0. 
        Y_i : tuple of float, optional
            Initial abundances n_i/n_b for species. Length must be equal to 
            self.nuclear_net.max_i_species. Must specify T_start and T_end if 
            not `None`. 
        T_start : float
            Temperature in MeV to start integration. Must specify Y_i and T_end if
            not `None`, otherwise const.T_start used. 
        T_end : float
            Temperature in MeV to end integration. 
        sampling_nTOp : int
            Number of points to subdivide (T_end, T_start) for n<->p rate 
            interpolation table. 
        rtol : float, optional
            Relative tolerance of the abundance solver. Default is 1e-4. 
        atol : float, optional
            Absolute tolerance of the abundance solver. Default is 1e-9. 
        max_steps : int, optional
            Maximum number of steps taken by the solver. Default is 4096. 
            Increasing this slows down the code, while decreasing this could 
            mean that the solver cannot complete the solution. 
        solver : Diffrax ODE solver 
            The Diffrax ODE solver to use. A stiff solver is recommended. 
            Default is the 3rd order Kvaerno solver. 
        save_history : bool
            If `True`, full solution is returned with temperature and time 
            abscissa.

        Returns
        -------
        tuple of array or array
            If `save_history`, a tuple containing an array of EM temperatures, 
            an array of times, and a Diffrax `Solution` instance, which can be 
            called as a function of time. Otherwise, returns yields of all 
            species considered in `self.nuclear_net`.  

        """

        print('Compiling abundance model...')

        if Y_i is not None: 
            if T_start is None: 
                raise TypeError('Specifying Y_i requires specifying a T_start')
        if T_start is not None: 
            if Y_i is None: 
                raise TypeError('Specifying T_start requires specifying Y_i')

        if nuclear_rates_q is None: 

            nuclear_rates_q = jnp.array(
                [0. for _ in self.nuclear_net.reactions]
            )

        if t_vec is None: 
            t_vec = self.get_t(rho_g_vec, rho_nu_vec, rho_NP_vec, P_NP_vec)

        if a_vec is None: 
            a_vec = self.get_a(rho_g_vec, rho_nu_vec, rho_NP_vec, P_NP_vec)

        if T_start is None: 

            T_start  = const.T_start

        if T_end is None: 

            T_end  = const.T_end

        # These are in MeV
        T_g_vec  = thermo.T_g(rho_g_vec)
        T_nu_vec = thermo.T_nu(rho_nu_vec) 

        a_start  = jnp.exp(
            jnp.interp(
                jnp.log(T_start), 
                jnp.flip(jnp.log(T_g_vec)), 
                jnp.flip(jnp.log(a_vec)), 
                left=jnp.log(a_vec[-1]), right=jnp.log(a_vec[0])
            )
        )

        t_start = jnp.exp(
            jnp.interp(
                jnp.log(T_start), 
                jnp.flip(jnp.log(T_g_vec)), 
                jnp.flip(jnp.log(t_vec)),
                left=jnp.log(t_vec[-1]), right=jnp.log(t_vec[0])
            )
        )

        t_end = jnp.exp(
            jnp.interp(
                jnp.log(T_end), 
                jnp.flip(jnp.log(T_g_vec)), 
                jnp.flip(jnp.log(t_vec)),
                left=jnp.log(t_vec[-1]), right=jnp.log(t_vec[0])
            )
        )

        ##################################
        # Weak Rates                     #
        ##################################

        T_interval_nTOp, nTOp_frwrd, nTOp_bkwrd = self.weak_rates(
            jnp.array([T_g_vec, T_nu_vec]), 
            T_start=T_start, T_end=T_end, sampling_nTOp=sampling_nTOp
        )

        ##################################
        # Initialization of Abundances   #
        ##################################
        
        
        if Y_i is None: 
            # If not provided, initialized to const.T_start. 

            # Neutron and proton yields, based on rates. 
            Yn_i = nTOp_bkwrd[0] / (nTOp_frwrd[0] + nTOp_bkwrd[0])
            Yp_i = 1. - Yn_i 

            # Other elements start at statistical equilibrium. 
            n_CMB_start = thermo.n_massless_BE(T_start, 0., 2.)
            eta_T_start = nB(a_start, eta_fac=eta_fac) / n_CMB_start

            Y_YNSE = self.YNSE(Yn_i, Yp_i, const.T_start, eta_T_start) 

            Y_others_i = Y_YNSE[2:self.nuclear_net.max_i_species]

            Y_i = (Yn_i, Yp_i) + tuple(Y_others_i)

        if save_history: 
            saveat = SaveAt(dense=True) 
        else: 
            # Default SaveAt
            saveat = SaveAt(t1=True)

        sol = diffeqsolve(
            ODETerm(self.Y_prime), solver, 
            t0=t_start, t1=t_end, dt0=None, y0=Y_i, 
            args = (
                a_vec, t_vec, T_g_vec, T_interval_nTOp, nTOp_frwrd, 
                nTOp_bkwrd, eta_fac, tau_n_fac, nuclear_rates_q
            ), saveat=saveat, stepsize_controller = PIDController(
                rtol=rtol, atol=atol,
            ), 
            max_steps=max_steps
        )

        if save_history: 
            return sol 
        else: 
            Y_f = jnp.array(sol.ys).flatten()
            return Y_f

    @eqx.filter_jit
    def get_t(self, rho_g_vec, rho_nu_vec, rho_NP_vec, P_NP_vec):
        """ 
        Time elapsed. 

        Parameters
        ----------
        rho_g_vec : array
            Energy density of photons. 
        rho_nu_vec : array
            Energy density of one species of neutrinos (assumed identical for 
            all species). 
        rho_NP_vec : array
            Energy density of all new physics fluids. 
        P_NP_vec : array
            Pressure of all new physics fluids. 

        Returns
        -------
        array
            Array of times in seconds corresponding to physical parameters 
            above. Initial time taken to be 1 / (2H). 
        
        """

        T_g_vec  = thermo.T_g(rho_g_vec)

        rho_tot_vec = (
            rho_EM_std_v(T_g_vec) + 3 * rho_nu_vec + rho_NP_vec 
        )

        P_tot_vec = p_EM_std_v(T_g_vec) + 3 * (rho_nu_vec/3) + P_NP_vec

        def P_tot(rho_tot): 

            return jnp.exp(
                jnp.interp(
                    jnp.log(rho_tot), 
                    jnp.flip(jnp.log(rho_tot_vec)), 
                    jnp.flip(jnp.log(P_tot_vec))
                )
            )

        def dt_prime(rho_tot, t, args): 

            return 1. / (
                -3. * thermo.Hubble(rho_tot) *(rho_tot + P_tot(rho_tot))
            )

        rho_tot_init = rho_tot_vec[0]
        rho_tot_fin  = rho_tot_vec[-1]

        sol_t = diffeqsolve(
            ODETerm(dt_prime), Tsit5(), 
            t0=rho_tot_init, t1=rho_tot_fin, 
            y0=1. / (2 * thermo.Hubble(rho_tot_init)), 
            dt0=None, max_steps=4096,
            saveat=SaveAt(ts=rho_tot_vec), 
            stepsize_controller=PIDController(rtol=1e-8, atol=1e-10)
        )

        return sol_t.ys

    @eqx.filter_jit
    def get_a(self, rho_g_vec, rho_nu_vec, rho_NP_vec, P_NP_vec): 
        """ 
        Scale factor. 

        Parameters
        ----------
        rho_g_vec : array
            Energy density of photons. 
        rho_nu_vec : array
            Energy density of one species of neutrinos (assumed identical for 
            all species). 
        rho_NP_vec : array
            Energy density of all new physics fluids. 
        P_NP_vec : array
            Pressure of all new physics fluids. 

        Returns
        -------
        array
            Array of scale factors corresponding to physical parameters above. 
            
        Notes
        -----
        The final entry a[-1] is given by TCMB0 / T_gamma[-1], where TCMB0 is 
        the CMB temperature measured today, and T_gamma[-1] is the temperature 
        of photons in the last entry of rho_g_vec. In other words, we assume no 
        subsequent entropy dump in the electromagnetic sector.
        
        """

        T_g_vec  = thermo.T_g(rho_g_vec)

        rho_tot_vec = (
            rho_EM_std_v(T_g_vec) + 3 * rho_nu_vec + rho_NP_vec 
        )  

        P_tot_vec = p_EM_std_v(T_g_vec) + 3 * (rho_nu_vec/3) + P_NP_vec

        def P_tot(rho_tot): 

            return jnp.exp(
                jnp.interp(
                    jnp.log(rho_tot), jnp.flip(jnp.log(rho_tot_vec)), 
                    jnp.flip(jnp.log(P_tot_vec))
                )
            )   

        def dlna_prime(rho_tot, t, args): 

            return 1. / (-3. * (rho_tot + P_tot(rho_tot)))
        
        rho_tot_init = rho_tot_vec[0]
        rho_tot_fin  = rho_tot_vec[-1]

        # a_0 = 1 arbitrarily, will rescale later. 
        sol_lna = diffeqsolve(
            ODETerm(dlna_prime), Tsit5(), 
            t0=rho_tot_init, t1=rho_tot_fin, 
            y0=0., dt0=None, max_steps=4096,
            saveat=SaveAt(ts=rho_tot_vec),
            stepsize_controller=PIDController(rtol=1e-8, atol=1e-10)
        )

        a_fin = const.T0CMB / T_g_vec[-1] 

        a_vec = jnp.exp(sol_lna.ys)
        # Rescale so that the last a is given by T_g_vec[-1] / TCMB today. 
        a_vec = a_vec / a_vec[-1] * a_fin

        return a_vec


    @eqx.filter_jit
    def Y_prime(self, t, Y, args):

        a_vec_in    = args[0]
        t_vec_in    = args[1]
        T_g_vec_in  = args[2]
        T_interval_in = args[3]
        nTOp_frwrd_vec_in = args[4]
        nTOp_bkwrd_vec_in = args[5]
        eta_fac = args[6]
        tau_n_fac = args[7] 
        nuclear_rates_q = args[8]
        
        a_in  = a_vec_in[0]
        a_fin = a_vec_in[-1]

        a = jnp.interp(
            t, t_vec_in, a_vec_in,
            left=a_in,right=a_fin
        )
        
        # Baryon density rescaled by eta_fac. 
        n0B = eta_fac*const.n0CMB*const.eta0
        rho0BmaOvermB = ma * n0B
        # number density times amu. 
        rhoBBN = rho0BmaOvermB * const.MeV4_to_gcmm3/a**3 

        T_t = jnp.interp(
            t, t_vec_in, T_g_vec_in, left=T_g_vec_in[0],right=T_g_vec_in[-1]
        )

        dY = self.nuclear_net(
            Y, T_t, rhoBBN, T_interval_in, nTOp_frwrd_vec_in,
            nTOp_bkwrd_vec_in, tau_n_fac=tau_n_fac, 
            nuclear_rates_q=nuclear_rates_q
        )

        return dY

    def YNSE(self, Yn, Yp, T, eta): 
        """
        Nuclear statistical equilibrium yields for all species. 

        Parameters
        ----------
        Yn : float
            The yield n_n / n_b of free neutrons. 
        Yp : float
            The yield n_p / n_b of free protons. 
        T : float
            The temperature of the baryons in MeV.
        eta : float
            The baryon-to-photon ratio. 

        Returns
        -------
        array
            Yields for all species considered in LINX (13 of them). 
        """

        A32Overmn = (
            self.species_mass / (
                mn**(self.species_A - self.species_Z) 
                * mp**self.species_Z
            )
        )**(3/2)

        return (
            (2 * self.species_spin + 1) * zeta_3**(self.species_A-1) 
            * jnp.pi**((1-self.species_A)/2) * 2**((3*self.species_A-5)/2) 
            * A32Overmn * T**(3/2*(self.species_A-1)) 
            * eta**(self.species_A-1) * Yp**self.species_Z 
            * Yn**(self.species_A-self.species_Z) 
            * jnp.exp(self.species_binding_energy / T)
        )

