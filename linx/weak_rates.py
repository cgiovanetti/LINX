import os

import numpy as np

import jax
import jax.numpy as jnp
# As of jax v0.4.24, jax.numpy.trapz has been deprecated, and replaced
# by scipy.integrate.trapezoid.
try: 
    import jax.numpy.trapz as trapz 
except ImportError: 
    from jax.scipy.integrate import trapezoid as trapz

import equinox as eqx

import linx.const as const 
from linx.special_funcs import gamma 
from jax.scipy.special import spence
from jax.scipy.special import expit

file_dir = os.path.dirname(__file__)

# Particle masses
from linx.const import me, mn, mp
Q = mn - mp # Mass difference between neutrons and protons

class WeakRates(eqx.Module): 
    """
    Class for weak rates calculation. 

    Attributes
    ----------
    RC_corr : bool
        Radiative corrections to weak rates. 
    thermal_corr : bool
        Thermal and bremsstrahlung corrections to weak rates 
        (pre-tabulated, assuming standard BBN). 
    FM_corr : bool
        Finite mass corrections to weak rates. 
    weak_mag_corr : bool
        Weak magnetism corrections to weak rates. 
    T_nTOp_thermal_interval : array
        EM temperature abscissa for tabulated n->p thermal corrections. 
    T_pTOn_thermal_interval : array
        EM temperature abscissa for tabulated p->n thermal corrections.
    L_nTOpCCRTh_res : array
        Tabulated dimensionless thermal corrections to n->p rate. 
        Divide by (tau_n * lambda_0) for the actual correction. 
    L_pTOnCCRTh_res : array
        Tabulated dimensionless thermal corrections to p->n rate. 
        Divide by (tau_n * lambda_0) for the actual correction. 
    """

    RC_corr : bool
    thermal_corr : bool
    FM_corr : bool
    weak_mag_corr : bool

    T_nTOp_thermal_interval : list
    T_pTOn_thermal_interval : list
    L_nTOpCCRTh_res : list
    L_pTOnCCRTh_res : list 

    lambda_0 : float

    def __init__(self, 
        RC_corr=True, FM_corr=True, weak_mag_corr=True, 
        thermal_corr=True
    ): 
        """
        Parameters
        ----------
        RC_corr : bool
            Include radiative corrections. 
        FM_corr : bool
            Include finite mass corrections. 
        weak_mag_corr : bool
            Include weak magnetism corrections. 
        thermal_corr : bool
            Include thermal and bremsstrahlung corrections. 
        """
        
        self.RC_corr = RC_corr
        self.FM_corr = FM_corr 
        self.weak_mag_corr = weak_mag_corr
        self.thermal_corr = thermal_corr
        
        if self.thermal_corr: 

            self.T_nTOp_thermal_interval, self.L_nTOpCCRTh_res = np.loadtxt(
                file_dir+"/data/weak_thermal_corrections/"
                +"nTOp_thermal_corrections_SBBN.txt", 
                unpack = True
            )

            try:
                gpus = jax.devices('gpu')
                self.T_nTOp_thermal_interval = jax.device_put(
                    self.T_nTOp_thermal_interval, device=gpus[0] 
                )
                self.L_nTOpCCRTh_res = jax.device_put(
                    self.L_nTOpCCRTh_res, device=gpus[0]
                )
            except: 
                pass

            self.T_pTOn_thermal_interval, self.L_pTOnCCRTh_res = np.loadtxt(
                file_dir+"/data/weak_thermal_corrections/"
                +"pTOn_thermal_corrections_SBBN.txt", 
                unpack = True
            )

            try:
                gpus = jax.devices('gpu')
                self.T_pTOn_thermal_interval = jax.device_put(
                    self.T_pTOn_thermal_interval, device=gpus[0] 
                )
                self.L_pTOnCCRTh_res = jax.device_put(
                    self.L_pTOnCCRTh_res, device=gpus[0]
                )
            except: 
                pass

        else: 

            self.T_nTOp_thermal_interval = [] 
            self.T_pTOn_thermal_interval = [] 
            self.L_nTOpCCRTh_res = [] 
            self.L_pTOnCCRTh_res = [] 

        self.lambda_0 = 0. 

        # Slight shift in limits to avoid unimportant singularities.
        en_vals = jnp.linspace(1.+.1e-6, Q/me-1e-6, 1000)
        dlambda_den_vals = self.dlambda_den_RC(en_vals)
        self.lambda_0 += trapz(dlambda_den_vals, en_vals)

        if self.FM_corr:

            pe_vals = jnp.linspace(
                0.+1e-4, jnp.sqrt((Q/me)**2 - 1.)-1e-5, 1000
            )
            y_vals = self.dlambda_dp_FM(pe_vals)
            self.lambda_0 += trapz(y_vals, pe_vals)

    @eqx.filter_jit
    def __call__(
        self, T_vec_ref, T_start, T_end, sampling_nTOp
    ): 
        """
        Evaluate n <-> p rates over range of EM temperatures. 

        Parameters
        ----------
        T_vec_ref : tuple of ndarray
            Reference (T_gamma, T_nu) to evaluate weak rates. In MeV. 
        T_start : float
            Highest photon temperature to evaluate rates at. In MeV.   
        T_end : float
            Lowest photon temperature to evaluate rates at. In MeV. 
        sampling_nTOp : int
            Number of points between T_start and T_end to evaluate at. 

        Returns
        -------
        tuple
            (T_EM abscissa, n->p rates, p->n rates). Note that rates are 
            dimensionless, normalized to the neutron decay width. T_EM abscissa
            limits are T_start, T_end, binned according to sampling_nTOp. 
        
        
        """
    
        T_interval = jnp.logspace(
            jnp.log10(T_start), jnp.log10(T_end), sampling_nTOp
        )

        nTOp_rates = self.nTOp_rates(T_interval, T_vec_ref)

        return (T_interval, ) + nTOp_rates

    @eqx.filter_vmap(in_axes=(None, 0, None))
    def nTOp_rates(self, Tg, T_vec_ref): 
        """
        Dimensionless n <-> p rates, normalized to neutron decay width.

        Parameters
        ----------
        Tg : float
            EM temperature in MeV at which to evaluate rate. 
        T_vec_ref : tuple
            (EM temperature array, neutrino temperature array) in MeV, used 
            for computing the weak rates. 

        Returns
        -------
        tuple
            (n -> p rate, p -> n rate). Note that rates are 
            dimensionless, normalized to the neutron decay width. 
        """

        Tg_vec_ref, Tnu_vec_ref = T_vec_ref
        Tnu_of_Tg_ref = Tnu_vec_ref / Tg_vec_ref
        
        x = me / Tg
        xnu = me / (
            Tg * jnp.interp(
                Tg, jnp.flip(Tg_vec_ref), jnp.flip(Tnu_of_Tg_ref), left=Tnu_of_Tg_ref[-1], right=Tnu_of_Tg_ref[0]
            )
        )

        pemin = 0.00001 
        
        pemax = jnp.maximum(7., 30./x) 
        p_vals = jnp.linspace(pemin+1e-4, pemax-1e-5, 1000) 

        nTOp_rate = 0. 
        pTOn_rate = 0. 

        y_CCR_vals = jnp.array([
            self.dGamma_nTOp_dp(p_vals, x, xnu), 
            self.dGamma_pTOn_dp(p_vals, x, xnu)
        ])
        CCR_rates = trapz(y_CCR_vals, p_vals)
        nTOp_rate += CCR_rates[0] / self.lambda_0 
        pTOn_rate += CCR_rates[1] / self.lambda_0 

        if self.FM_corr:
    
            y_FMCCR_vals = jnp.array([ 
                self.ddelt_Gamma_nTOp_FM_dp(p_vals, x, xnu),
                self.ddelt_Gamma_pTOn_FM_dp(p_vals, x, xnu)
            ])
            FMCCR_rates = trapz(y_FMCCR_vals, p_vals)
            nTOp_rate += FMCCR_rates[0] / self.lambda_0 
            pTOn_rate += FMCCR_rates[1] / self.lambda_0 

        if self.thermal_corr: 

            thermal_rates = jnp.array([
                jnp.interp(
                    Tg, 
                    const.kB * self.T_nTOp_thermal_interval, 
                    self.L_nTOpCCRTh_res,
                    left=self.L_nTOpCCRTh_res[0],
                    right=self.L_nTOpCCRTh_res[-1]
                ),
                jnp.interp(
                    Tg, 
                    const.kB * self.T_pTOn_thermal_interval, 
                    self.L_pTOnCCRTh_res,
                    left=self.L_pTOnCCRTh_res[0], 
                    right=self.L_pTOnCCRTh_res[-1]
                )
            ])
            nTOp_rate += thermal_rates[0] / self.lambda_0 
            pTOn_rate += thermal_rates[1] / self.lambda_0 

        return (nTOp_rate, pTOn_rate)

    
    def Sirlin_G(self, kmax, en):
        """
        Sirlin's universal function. 

        Parameters
        ----------
        kmax : float
            Dimensionless maximum energy of photon considered in 
            radiative correction, normalized to electron mass. 
        en : float
            Dimensionless electron energy, normalized to electron mass)

        Returns
        -------
        float

        Notes
        -----
        See Pitrou+ 1801.08023 Eq. (B32). 

        """

        b = jnp.sqrt(en**2 - 1.) / en

        Rd_b = jnp.where(b==0, 1, jnp.arctanh(b) / b)

        return (
            3.*jnp.log(mp/me) - 3./4. + 4.*(Rd_b - 1.) * (
                kmax/(3.*en) - 3./2. + jnp.log(2.*kmax)
            ) 
            + Rd_b * (2.*(1. + b**2) + kmax**2/(6.*en**2) - 4.*b*Rd_b) 
            + (4./b)*(-spence(1. - 2.*b/(1. + b)))
        )

    def R_RC(self, kmax, en):
        """
        Resummed radiative correction term at T = 0. 

        Parameters
        ----------
        kmax : float
            Dimensionless maximum energy of photon considered in 
            radiative correction, normalized to electron mass. 
        en : float
            Dimensionless electron energy, normalized to electron mass)

        Returns
        -------
        float

        Notes
        -----
        See Pitrou+ 1801.08023 Eq. (103), full definition in Eq. (B35).
        Note that the constant 1/134 below is approximately the fine-structure
        constant at the proton mass scale.

        """

        # Constants defined in Pitrou+ 1801.08023 Eq. (B30)

        C = 0.891
        A_g = -0.34
        mA = 1.2e3 # in MeV 

        # Constants defined in Pitrou+ 1801.08023 Eq. (B36)
        L = 1.02094
        S = 1.02248 
        delta_factor = -0.00043 # alpha_FS/(2*pi) * delta
        NLL = -1e-4
            
        return (
            (
                1. + const.aFS / (2.*jnp.pi) * (
                    self.Sirlin_G(kmax, en) - 3.*jnp.log(mp / (2*Q))
                )
            ) * (L + (const.aFS/jnp.pi)*C + delta_factor) * (
                S + 1./(134.*2.*jnp.pi)*(jnp.log(mp/mA) + A_g) + NLL
            )
        )

    def Fermi(self, b):
        """
        Fermi function in the relativistic limit. 

        Parameters
        ----------
        b : float
            Speed of the electron. 
        
        Notes
        -----
        See Pitrou+ 1801.08023 Eq. (100). Equivalent to Sommerfeld 
        enhancement factor when v << 1. 
        """

        Gamma = jnp.sqrt(1. - const.aFS**2.) - 1.
        gamma1 = 1. + Gamma
        gamma2 = 3. + 2.*Gamma
        lambda_Compton = 1. / (me / (const.hbar * const.c))

        return (
            (1. + Gamma/2.) * 4. 
            * (2. * const.radproton * b / lambda_Compton)**(2.*Gamma)
            / gamma(gamma2)**2
            * jnp.exp(jnp.pi * const.aFS / b) / (1. - b**2)**Gamma 
            * jnp.abs(gamma(gamma1 + const.aFS/b*1j))**2
        )
        
    def bFermi(self, b):
        """
        Fermi function in the relativistic limit, times speed of electron. 

        Parameters
        ----------
        b : float
            Speed of the electron. 

        Notes
        -----
        See Pitrou+ 1801.08023 Eq. (100). Equivalent to Sommerfeld 
        enhancement factor when v << 1. 

        """

        return b*self.Fermi(b)
    
    @eqx.filter_vmap(in_axes=(None, 0))
    def dlambda_den_RC(self, en):
        """
        Derivative of lambda with respect to energy of electron, 
        including radiative corrections at T = 0. 

        Parameters
        ----------
        en : float
            Energy of the electron produced in neutron decay, normalized to 
            the electron mass. 

        Returns
        -------
        float

        Notes
        -----
        See Pitrou+ 1801.08023 Eq. (106), but note the change in variables. 

        """

        q = Q/me
        b = jnp.sqrt(en**2 - 1.)/en

        if self.RC_corr: 
            R_term = (
                self.Fermi(b)
                * self.R_RC(q-en, en)
            )
        else: 
            R_term = 1. 

        return (
            en**2 * (q-en)**2 * b * R_term
        )
    
    def chi_n_decay(self, pe):
        """
        Finite nucleon mass correction term for neutron decay. 

        Parameters
        ----------
        pe : float
            Dimensionless momentum of the electron, normalized to electron 
            mass. 

        Returns
        -------
        float

        Notes
        -----
        See Pitrou+ 1801.08023 Eq. (B26). 

        """

        en = jnp.sqrt(pe**2 + 1.)

        # Equivalent to 2*fWM in Pitrou+ 1801.08023 Eq (B9). 
        # See also Ivanov+ 1212.0332
        if self.weak_mag_corr:
            delta_kappa = 3.7058
        else: 
            delta_kappa = 0. 

        M = mn/me
        q = Q/me
        
        # Pitrou+ 1801.08023 Eq. (B24a)--(B24c)
        f1 = (
            ((1. + const.gA)**2. + 2.*delta_kappa*const.gA)
            / (1. + 3.*const.gA**2)
        )
        f2 = (
            ((1. - const.gA)**2. - 2.*delta_kappa*const.gA)
            / (1. + 3.*const.gA**2)
        )
        f3 = (const.gA**2 - 1.) / (1. + 3.*const.gA**2)

        return (
            f1*(q - en)**2*(pe**2/(M*en)) 
            + f2/M*(q - en)**3 
            - (f1 + f2 + f3) / M * (
                2*(q - en)**3 + (q - en)*pe**2
            ) 
            + f3 * (q - en)**2 * (pe**2) / (M*en)
        )
    
    @eqx.filter_vmap(in_axes=(None, 0))
    def dlambda_dp_FM(self, pe):
        """
        Derivative of the correction to lambda with respect to momentum, 
        due to finite mass effects. 

        Parameters
        ----------
        pe : float
            Dimensionless momentum of the electron, normalized to the electron  
            mass.

        Returns
        -------
        float

        Notes
        -----
        See Pitrou+ 1801.08023 Eq. (118). If radiative correction is included, 
        add a factor of R. Derivative with respect to momentum (and not energy) 
        to keep consistency with PRyMordial. 

        """

        en = jnp.sqrt(pe**2 + 1.)
        b = pe/en

        if self.RC_corr:

            R_rad = self.R_RC(Q/me - en, en) 

        else: 

            R_rad = 1. 

        return (
            pe**2 * self.chi_n_decay(pe) 
            * R_rad * self.Fermi(b)
        )

    def chi_Born(self, en, x, x_nu, sgnq):
        """
        Integrand in momentum integral for Born weak rate. 

        Parameters
        ----------
        en : float
            Dimensionless energy of the electron, normalized to electron mass.
        x : float
            Dimensionless inverse EM temperature, normalized to electron mass. 
        x_nu : float 
            Dimensionless inverse nu temperature, normalized to electron mass. 
        sqnq : int
            Should have value +1 or -1, to switch between chi_+ and chi_-. 

        Notes
        -----
        See Pitrou+ 1801.08023 Eq. (79). 

        """

        # xi_nu = 0. # nu chemical potential set to zero for now. 

        # sgnq = +1 corresponds to chi_plus. 
        e_nu = en - sgnq*(Q/me) 
        g_nu = expit(-x_nu*e_nu)
        g_e  = expit(-x*(-en))

        return e_nu**2 * g_nu * g_e 
    
    def Fermi_sgn(self, sgnq, sgnE, b):
        """
        Fermi function, with a check for proton and electron final state. 

        Parameters
        ----------
        sgnq : int
            +1 or -1 to choose between F_+ and F_-. 
        sgnE : int
            +1 or -1 to choose between positive or negative arguments of F. 
        b : float
            The speed of the electron. 

        Returns
        -------
        float

        Notes
        -----
        See Pitrou+ 1801.08023 Eq. (102). 
        """
        
        result = jnp.where(sgnq*sgnE > 0, self.Fermi(b), 1.)
        return result
    
    ###############################
    # n<->p Rates                 #
    ###############################
       
    def dGamma_dp(self, p, x, x_nu, sgnq):
        """
        Integrand over momentum for n <-> p rate.  including radiative 
        corrections.

        Parameters
        ----------
        p : float
            Dimensionless momentum of the electron, normalized to the 
            electron mass. 
        x : float
            Dimensionless EM inverse temperature, normalized to the electron 
            mass. 
        x_nu : float
            Dimensionless neutrino inverse temperature, normalized to the
            electron mass. 
        sgnq : int
            +1 or -1, to select between n -> p or p -> n. 

        Returns
        -------
        float

        Notes
        -----
        See Pitrou+ 1801.08023 Eq. (101) and (104). Radiative corrections 
        included as desired. Without radiative corrections, this gives the 
        Born approximation result. 
        """

        en = jnp.sqrt(p**2 + 1.)

        if self.RC_corr: 
            RC_term_plus = (
                self.R_RC(jnp.abs(sgnq*Q/me - en), en)
                * self.Fermi_sgn(sgnq, 1, p/en)
            )
            RC_term_minus = (
                self.R_RC(jnp.abs(sgnq*Q/me + en), en)
                * self.Fermi_sgn(sgnq, -1, p/en)
            )
        else: 
            RC_term_plus = 1. 
            RC_term_minus = 1. 

        return p**2 * (
            (
                self.chi_Born(en, x, x_nu, sgnq)
                * RC_term_plus
            ) + (
                self.chi_Born(-en, x, x_nu, sgnq)
                * RC_term_minus
            )
        )

    @eqx.filter_vmap(in_axes=(None, 0, None, None))
    def dGamma_nTOp_dp(self, p, x, xnu):
        """
        Integrand over momentum for n -> p rate including radiative 
        corrections.

        Parameters
        ----------
        p : float
            Dimensionless momentum of the electron, normalized to the 
            electron mass. 
        x : float
            Dimensionless EM inverse temperature, normalized to the electron 
            mass. 
        x_nu : float
            Dimensionless neutrino inverse temperature, normalized to the
            electron mass. 

        Returns
        -------
        float

        Notes
        -----
        See Pitrou+ 1801.08023 Eq. (101). 
        """

        return self.dGamma_dp(p, x, xnu, 1)
    
    @eqx.filter_vmap(in_axes=(None, 0, None, None))
    def dGamma_pTOn_dp(self, p, x, xnu):
        """
        Integrand over momentum for p -> n rate including radiative 
        corrections.

        Parameters
        ----------
        p : float
            Dimensionless momentum of the electron, normalized to the 
            electron mass. 
        x : float
            Dimensionless EM inverse temperature, normalized to the electron 
            mass. 
        x_nu : float
            Dimensionless neutrino inverse temperature, normalized to the
            electron mass. 
        sgnq : int
            +1 or -1, to select between n -> p or p -> n. 

        Returns
        -------
        float

        Notes
        -----
        See Pitrou+ 1801.08023 Eq. (104). 
        """

        return self.dGamma_dp(p, x, xnu, -1)
    
    
    ###########################
    # Finite Mass Corrections #
    ###########################
       
       
    def chi_FM(self, en, x, x_nu, sgnq):
        """
        Integrand over momentum for finite mass correction to n <-> p rate.

        Parameters
        ----------
        en : float
            Dimensionless energy of the electron, normalized to the electron 
            mass. 
        x : float
            Dimensionless EM inverse temperature, normalized to the electron 
            mass. 
        x_nu : float
            Dimensionless neutrino inverse temperature, normalized to the 
            electron mass. 
        sgnq : int
            +1 or -1 corresponding to chi_+ or chi_-. 

        Notes
        -----
        See Pitrou+ 1801.08023 Eq. (B23). However, instead of m_N, this appears
        to choose m_p for + and m_n for -. This is consistent with PRIMAT, and 
        probably makes sense, since we should apply the finite mass correction 
        to the outgoing state. 

        Note also a typo on the first term Eq. (B23) (on dimensional grounds), 
        corrected in PRIMAT. It should be g_nu^(2,0) as well. 

        """
        pe = jnp.sqrt(en**2 - 1)

        Mp = mp/me
        Mn = mn/me
        
        # Toggles between proton and neutron mass, normalized to electron mass. 
        M_sgnq = (mp + mn - sgnq*Q)/(2. * me)

        # Equivalent to 2*fWM in Pitrou+ 1801.08023 Eq (B9). 
        # See also Ivanov+ 1212.0332
        if self.weak_mag_corr:
            delta_kappa = 3.7058
        else: 
            delta_kappa = 0. 

        f_1 = (
            (1. + sgnq*const.gA)**2. + 2.*delta_kappa*sgnq*const.gA
        ) / (1. + 3.*const.gA**2)
        f_2 = (
            (1. - sgnq*const.gA)**2. - 2.*delta_kappa*sgnq*const.gA
        )/(1. + 3.*const.gA**2)
        f_3 = (const.gA**2 - 1.)/(1. + 3.*const.gA**2)

        # Electron distribution function term. 
        g_e = expit(-x*(-en))

        # Dimensionless energy of the neutrino. 
        en_nu = en - sgnq*Q / me

        expit_neg = expit(-en_nu*x_nu)
        expit_pos = expit(en_nu*x_nu)

        res_e2p1 = (
            2 * en_nu * expit_neg**2 
            + (en_nu * expit_neg) * (2 - en_nu*x_nu) * expit_pos
        )
        
        res_e3p1 = (
            3 * en_nu**2 * expit_neg**2 
            + (en_nu**2 * expit_neg) * (3 - en_nu*x_nu) * expit_pos
        )
        
        res_e4p1 = (
            4 * en_nu**3 * expit_neg**2 
            + (en_nu**3 * expit_neg) * (4 - en_nu*x_nu) * expit_pos
        )
        
        res_e2p0 = en_nu**2 * expit_neg
        
        res_e3p0 = en_nu**3 * expit_neg
        
        res_e2p2 = (
            (en_nu*x_nu * (en_nu*x_nu - 4) + 2) * expit_neg * expit_pos**2 
            + (4 - en_nu*x_nu*(en_nu*x_nu + 4))*expit_neg**2 * expit_pos 
            + 2 * expit_neg**3
        )
        
        res_e3p2 = en_nu * (
            (en_nu*x_nu * (en_nu*x_nu - 6) + 6) * expit_neg * expit_pos**2 
            + (12 - en_nu*x_nu*(en_nu*x_nu + 6)) * expit_neg**2 * expit_pos 
            + 6 * expit_neg**3
        )
        
        res_e4p2 = en_nu**2 * (
            (en_nu*x_nu * (en_nu*x_nu - 8) + 12) * expit_neg * expit_pos**2 
            + (24 - en_nu*x_nu*(en_nu*x_nu + 8)) * expit_neg**2 * expit_pos 
            + 12 * expit_neg**3
        )

        result =  g_e * (
            f_1 * res_e2p0 * (pe**2 / (M_sgnq * en))
            + f_2 * res_e3p0 * (-(1. / M_sgnq))
            + (f_1+f_2+f_3) / (2. * x * M_sgnq) * (
                res_e4p2 + res_e2p2 * pe**2
            )
            + (f_1+f_2+f_3) / (2. * M_sgnq) * (
                res_e4p1 + res_e2p1 * pe**2
            )
            - (f_1+f_2) / (x * M_sgnq) * (
                res_e3p1 + res_e2p1 * pe**2 / (-en)
            )
            - f_3 * 3. / (x * M_sgnq) * res_e2p0 
            + f_3 / (3 * M_sgnq) * res_e3p1 * pe**2 / en
            + f_3 * 2. / (2. * x * 3. * M_sgnq) * res_e3p2 * pe**2 / en
            - (f_1 + f_2 + f_3) * 3. / (2. * x) * (
                (1. - (Mn/Mp)**sgnq) * res_e2p1
            )
        )
        return result

    def ddelt_Gamma_FM_dp(self, p, x, znu, sgnq):
        """
        Integrand over momentum for finite mass corrections to the n <-> p
        rate. 

        Parameters
        ----------
        p : float
            Dimensionless momentum of the electron, normalized to the 
            electron mass. 
        x : float
            Dimensionless EM inverse temperature, normalized to the electron 
            mass. 
        x_nu : float
            Dimensionless neutrino inverse temperature, normalized to the
            electron mass. 
        sgnq : int
            +1 or -1, to select between n -> p or p -> n. 

        Returns
        -------
        float

        Notes
        -----
        See Pitrou+ 1801.08023 Eq. (115). Radiative corrections included as 
        desired.
        """

        en = jnp.sqrt(p**2 + 1.)
        b = p / en

        if self.RC_corr: 
            RC_term_plus = self.R_RC(
                jnp.abs(sgnq*Q/me - en), en
            ) * self.Fermi_sgn(sgnq, 1, b) 
            RC_term_minus = self.R_RC(
                jnp.abs(sgnq*Q/me + en), en
            ) * self.Fermi_sgn(sgnq, -1, b)
        
        else: 
            RC_term_plus = 1. 
            RC_term_minus = 1. 

        result =  p**2 * (
            (
                self.chi_FM(en, x, znu, sgnq) 
                * RC_term_plus 
            ) + (
                self.chi_FM(-en, x, znu, sgnq) 
                * RC_term_minus
            )
        )
        return result

    @eqx.filter_vmap(in_axes=(None, 0, None, None))
    def ddelt_Gamma_nTOp_FM_dp(self, p, x, xnu):
        """
        Integrand over momentum for finite mass corrections to the n -> p
        rate. 

        Parameters
        ----------
        p : float
            Dimensionless momentum of the electron, normalized to the 
            electron mass. 
        x : float
            Dimensionless EM inverse temperature, normalized to the electron 
            mass. 
        x_nu : float
            Dimensionless neutrino inverse temperature, normalized to the
            electron mass. 

        Returns
        -------
        float

        Notes
        -----
        See Pitrou+ 1801.08023 Eq. (115). 
        """

        return self.ddelt_Gamma_FM_dp(p, x, xnu, 1)

    @eqx.filter_vmap(in_axes=(None, 0, None, None))
    def ddelt_Gamma_pTOn_FM_dp(self, p, x, xnu):
        """
        Integrand over momentum for finite mass corrections to the p -> n 
        rate. 

        Parameters
        ----------
        p : float
            Dimensionless momentum of the electron, normalized to the 
            electron mass. 
        x : float
            Dimensionless EM inverse temperature, normalized to the electron 
            mass. 
        x_nu : float
            Dimensionless neutrino inverse temperature, normalized to the
            electron mass. 

        Returns
        -------
        float

        Notes
        -----
        See Pitrou+ 1801.08023 Eq. (115). 
        """

        return self.ddelt_Gamma_FM_dp(p, x, xnu, -1)