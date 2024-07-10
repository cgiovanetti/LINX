import jax.numpy as jnp
import equinox as eqx

import linx.const as const 
from linx.reactions import Reaction

class NuclearRates(eqx.Module): 
    """
    Nuclear net and rates. 

    Attributes
    ----------
    max_i_species : int
        Number of nuclear species to track. Convention is 0:n, 1:p, 2:d, 3:t,
        4:He3, 5:a, 6:Li7, 7:Be7, 8: Li6, 9: He6, 10: Li8, 11: B8. 
        max_i_species = 8 would go up to Be7, for example. 
    interp_type : str
        Interpolation method for nuclear rates with spline data. Either 'linear'
        or 'log'. 
    reactions : list of Reaction
        Reactions to consider in the network. 
    in_states : dict of tuple
        Nuclear species on the LHS of each reaction. 
    out_states : dict of tuple
        Nuclear species on the RHS of each reaction. 
    frwrd_symmetry_fac : dict of float
        Symmetry factor for forward direction of reaction. 
    bkwrd_symmetry_fac : dict of float
        Symmetry factor for backward direction of reaction. 
    frwrd_rate : dict of callable
        Dictionary of forward rate parameter (Gamma, <sigma v> or 
        <sigma v^2> divided by (1 amu)^(N_in-1)) for each reaction, 
        units (s^-1/g, cm^3/s/g or cm^6/s/g^2)
    bkwrd_rate : dict of callable
        Dictionary of backward rate parameter for each reaction. 
    """

    max_i_species : int
    interp_type : str
    reactions : list
    reactions_names: list
    in_states : dict
    out_states : dict
    frwrd_symmetry_fac : dict
    bkwrd_symmetry_fac : dict 
    frwrd_rate_param : dict
    bkwrd_rate_param : dict
    frwrd_reaction_by_particle : dict
    bkwrd_reaction_by_particle : dict

    def __init__(
        self, reactions=None, nuclear_net=None, interp_type='linear',
        max_i_species=None
    ):
        """
        Populates the class. 

        Parameters
        ----------
        reactions : list of Reaction
            Reactions to be considered in the nuclear net. 
        nuclear_net : str
            Used for fixed networks, choices are: 
            'np_only': Only n <-> p rates will be used. 
            'key_PRIMAT_2018': PRIMAT 2018 small network rates, also used 
            by PRyMordial. 
            'key_YOF': YOF small network rates, also used by 
            PRyMordial.
            'key_PRIMAT_2023': PRIMAT 2022 rates.
            'key_PArthENoPE': PArthENoPE rates. 
            'full_PRIMAT_2022': PRIMAT 2022 full network rates (unverified). 
        interp_type : str
            Interpolation type for nuclear rates with spline data. Either 
            'linear' or 'log'. 
        max_i_species : int
            Number of nuclear species to consider. 
        """
 
        self.interp_type = interp_type
        self.max_i_species = max_i_species

        if reactions is not None: 

            self.reactions = reactions

        elif nuclear_net == 'np_only': 
            # No nuclear reactions. n<->p rates are always included. 

            self.reactions = []
            self.max_i_species = 2

        else: 

            self.reactions = self.populate(nuclear_net)

            if nuclear_net[:3] == 'key': 
 
                self.max_i_species = 8 

            elif nuclear_net[:5] == 'small': 

                self.max_i_species = 9

            else: 

                self.max_i_species = 12
            
        self.in_states = {} 
        self.out_states = {} 
        self.frwrd_symmetry_fac = {} 
        self.bkwrd_symmetry_fac = {} 
        self.frwrd_rate_param = {} 
        self.bkwrd_rate_param = {}
        self.reactions_names = [] 

        self.frwrd_reaction_by_particle = {
            i:[] for i in range(self.max_i_species)
        }
        self.bkwrd_reaction_by_particle = {
            i:[] for i in range(self.max_i_species)
        }

        for rxn in self.reactions: 

            self.in_states[rxn.name]  = rxn.in_states
            self.out_states[rxn.name] = rxn.out_states 
            self.frwrd_symmetry_fac[rxn.name] = rxn.frwrd_symmetry_fac 
            self.bkwrd_symmetry_fac[rxn.name] = rxn.bkwrd_symmetry_fac 
            self.frwrd_rate_param[rxn.name] = rxn.frwrd_rate_param
            self.bkwrd_rate_param[rxn.name] = rxn.bkwrd_rate_param

            for i in range(self.max_i_species): 
                if i in self.in_states[rxn.name]:
                    self.frwrd_reaction_by_particle[i].append(rxn.name)
                if i in self.out_states[rxn.name]: 
                    self.bkwrd_reaction_by_particle[i].append(rxn.name)
            
            self.reactions_names.append(rxn.name)
            

    @eqx.filter_jit
    def __call__(
        self, Y, T_t, rhoBBN, T_interval, 
        nTOp_frwrd_vec, nTOp_bkwrd_vec, tau_n_fac=1., nuclear_rates_q=None
    ): 
        """
        Returns the rate of change of the abundances.  

        Parameters
        ----------
        Y : Array
            Current abundance of species (n_species/n_b).
        T_t : float
            Current temperature in MeV. 
        rhoBBN : float
            Number density of baryons x 1amu in g/cm^3. 
        T_interval : array of float
            Interval over which n<->p rates were calculated in MeV. 
        nTOp_frwrd_vec : array of float
            n -> p dimensionless rates corresponding to T_interval, normalized 
            to the neutron decay width, i.e. Yn * nTOp_frwrd / tau_n + ... 
            = -dYn/dt. 
        nTOp_bkwrd_vec : array of float
            p -> n dimensionless rates corresponding to T_interval. 
        tau_n_fac : float
            Rescaling parameter for neutron decay lifetime. 
        nuclear_rates_q : array
            Rescaling parameter of expsigma in nuclear rate. If None, 
            no rescaling is assumed.

        Returns
        -------
        Array
            dY/dt in s^-1. Same dimensions as Y. 
        """

        if nuclear_rates_q is None: 

            nuclear_rates_q = jnp.array([0. for _ in self.reactions])

        dYdt_vec = jnp.zeros(len(Y))
        
        _nTOp_frwrd = jnp.interp(
            T_t, jnp.flip(T_interval), jnp.flip(nTOp_frwrd_vec),
            left=nTOp_frwrd_vec[-1],right=nTOp_frwrd_vec[0]
        ) / (const.tau_n * tau_n_fac)

        _nTOp_bkwrd = jnp.interp(
            T_t, jnp.flip(T_interval), jnp.flip(nTOp_bkwrd_vec),
            left=nTOp_bkwrd_vec[-1],right=nTOp_bkwrd_vec[0]
        ) / (const.tau_n * tau_n_fac)

        # These functions take temperature in K. 
        frwrd_rate_params = {
            rxn.name:self.frwrd_rate_param[rxn.name](
                T_t / const.kB, nuclear_rates_q[i]
            ) for i,rxn in enumerate(self.reactions)
        } 
        bkwrd_rate_params = {
            rxn.name:self.bkwrd_rate_param[rxn.name](
                T_t / const.kB, nuclear_rates_q[i]
            ) for i,rxn in enumerate(self.reactions)
        }

        dYdt_vec = dYdt_vec.at[0].set(-_nTOp_frwrd*Y[0] + _nTOp_bkwrd*Y[1])
        dYdt_vec = dYdt_vec.at[1].set( _nTOp_frwrd*Y[0] - _nTOp_bkwrd*Y[1])

        for rxn in self.reactions: 
            dYdt_vec += self.get_dYdt_rxn(
                rxn.name, Y, rhoBBN, 
                frwrd_rate_params[rxn.name], bkwrd_rate_params[rxn.name]
            )

        return tuple(dYdt_vec)
    
    def get_dYdt_rxn(
        self, rxn, Y, rhoBBN, frwrd_rate_param, bkwrd_rate_param
    ):
        """
        Returns the rate of change of abundances due to a particular reaction.

        Parameters
        ----------
        rxn : str
            Name of the reaction. 
        Y : Array
            Current abundance of species (n_species/n_b).
        rhoBBN : float
            Number density of baryons x 1amu in g/cm^3. 
        frwrd_rate_param : callable
            Function returning the forward rate (either <sigma v> or 
            <sigma v^2>). Takes two arguments, `T` for EM temperature *in K* and `p` for rescaling of the rate. 
        bkwrd_rate_param : callable
            Similar to `frwrd_rate_param`, but for the backward rate. 

        Returns
        -------
        Array
            dY/dt in s^-1. Same dimensions as Y. 
        """
        
        dYdt_vec = jnp.zeros(len(Y))
        Y_prod_frwrd = 1. 
        Y_prod_bkwrd = 1. 
        for i in self.in_states[rxn]: 
            Y_prod_frwrd *= Y[i]
        for j in self.out_states[rxn]: 
            Y_prod_bkwrd *= Y[j] 

        for i in self.in_states[rxn]: 
            dYdt_vec = dYdt_vec.at[i].add(
                - self.frwrd_symmetry_fac[rxn]
                * rhoBBN**(len(self.in_states[rxn])-1)
                * Y_prod_frwrd * frwrd_rate_param 
            )
            dYdt_vec = dYdt_vec.at[i].add(
                + self.bkwrd_symmetry_fac[rxn] 
                * rhoBBN**(len(self.out_states[rxn])-1)
                * Y_prod_bkwrd * bkwrd_rate_param
            )

        for j in self.out_states[rxn]: 
            dYdt_vec = dYdt_vec.at[j].add(
                + self.frwrd_symmetry_fac[rxn]
                * rhoBBN**(len(self.in_states[rxn])-1)
                * Y_prod_frwrd * frwrd_rate_param 
            )
            dYdt_vec = dYdt_vec.at[j].add(
                - self.bkwrd_symmetry_fac[rxn] 
                * rhoBBN**(len(self.out_states[rxn])-1)
                * Y_prod_bkwrd * bkwrd_rate_param
            )

        return dYdt_vec
    
    def populate(self, network): 
        """
        Populate the nuclear rates. 

        Parameters
        ----------
        network : str
            Nuclear network of interest. Choices are: 
            'key_PRIMAT_2018': PRIMAT 2018 key network rates, also used 
            by PRyMordial. 
            'key_YOF': YOF key network rates, also used by 
            PRyMordial.
            'key_PRIMAT_2023': PRIMAT 2023 key network rates.
            'key_PArthENoPE': PArthENoPE key network rates. 
            'full_PRIMAT_2023': PRIMAT 2023 full network rates.

        Returns
        -------
        list of Reaction
        """

        key_rxns_dict = {} 

        if network != 'key_PArthENoPE': 

            if (
                network == 'key_PRIMAT_2018' 
                or network == 'key_YOF' 
                or network == 'key_PRIMAT_2023'
            ): 
                key_network_str = network 
            elif network == 'full_PRIMAT_2018': 
                key_network_str = 'key_PRIMAT_2018'
            elif (
                network == 'small_PRIMAT_2023'
                or network == 'full_PRIMAT_2023'
            ): 
                key_network_str = 'key_PRIMAT_2023'

            key_rxns_dict['npdg'] = Reaction(
                'npdg', (0, 1), (2, ), 4.7161402e9, 1.5, -25.81502, 
                spline_data=key_network_str+'/npdg.txt', 
                interp_type=self.interp_type
            )
            key_rxns_dict['dpHe3g'] = Reaction(
                'dpHe3g', (1, 2), (4, ), 1.6335102e10, 1.5, -63.749132, 
                spline_data=key_network_str+'/dpHe3g.txt',
                interp_type=self.interp_type
            )
            key_rxns_dict['ddHe3n'] = Reaction(
                'ddHe3n', (2, 2), (0, 4), 1.7318296, 0., -37.934112, 
                spline_data=key_network_str+'/ddHe3n.txt', 
                interp_type=self.interp_type
            )
            key_rxns_dict['ddtp'] = Reaction(
                'ddtp', (2, 2), (1, 3), 1.7349209, 0., -46.797116, 
                spline_data=key_network_str+'/ddtp.txt', 
                interp_type=self.interp_type
            )
            key_rxns_dict['tpag'] = Reaction(
                'tpag', (1, 3), (5, ), 2.610575e10, 1.5, -229.93039, 
                spline_data=key_network_str+'/tpag.txt', 
                interp_type=self.interp_type
            )
            key_rxns_dict['tdan'] = Reaction(
                'tdan', (2, 3), (0, 5), 5.5354059, 0., -204.11537, 
                spline_data=key_network_str+'/tdan.txt', 
                interp_type=self.interp_type
            )
            key_rxns_dict['taLi7g'] = Reaction(
                'taLi7g', (3, 5), (6, ), 1.1132988e10, 1.5, -28.635551, 
                spline_data=key_network_str+'/taLi7g.txt', 
                interp_type=self.interp_type
            )
            key_rxns_dict['He3ntp'] = Reaction(
                'He3ntp', (0, 4), (1, 3), 1.001785, 0., -8.8630042, 
                spline_data=key_network_str+'/He3ntp.txt', 
                interp_type=self.interp_type
            )
            key_rxns_dict['He3dap'] = Reaction(
                'He3dap', (2, 4), (1, 5), 5.5452865, 0., -212.97837, 
                spline_data=key_network_str+'/He3dap.txt', 
                interp_type=self.interp_type
            )
            key_rxns_dict['He3aBe7g'] = Reaction(
                'He3aBe7g', (4, 5), (7, ), 1.1128943e10, 1.5, -18.417922, 
                spline_data=key_network_str+'/He3aBe7g.txt', 
                interp_type=self.interp_type
            )
            key_rxns_dict['Be7nLi7p'] = Reaction(
                'Be7nLi7p', (0, 7), (1, 6), 1.0021491, 0., -19.080633, 
                spline_data=key_network_str+'/Be7nLi7p.txt', 
                interp_type=self.interp_type
            )
            key_rxns_dict['Li7paa'] = Reaction(
                'Li7paa', (1, 6), (5, 5), 4.6898011, 0., -201.29484, 
                spline_data=key_network_str+'/Li7paa.txt', 
                interp_type=self.interp_type
            )
            

        else: 

            from linx.data.nuclear_rates import key_PArthENoPE as PArth

            key_rxns_dict['npdg'] = Reaction(
                'npdg', (0, 1), (2, ), 4.7161402e9, 1.5, -25.81502,
                frwrd_rate_param_func = PArth.npdg_frwrd_rate
            )
            key_rxns_dict['dpHe3g'] = Reaction(
                'dpHe3g', (1, 2), (4, ), 1.6335102e10, 1.5, -63.749132,  
                frwrd_rate_param_func = PArth.dpHe3g_frwrd_rate
            )
            key_rxns_dict['ddHe3n'] = Reaction(
                'ddHe3n', (2, 2), (0, 4), 1.7318296, 0., -37.934112, 
                frwrd_rate_param_func = PArth.ddHe3n_frwrd_rate
            )
            key_rxns_dict['ddtp'] = Reaction(
                'ddtp', (2, 2), (1, 3), 1.7349209, 0., -46.797116, 
                frwrd_rate_param_func = PArth.ddtp_frwrd_rate
            )
            key_rxns_dict['tpag'] = Reaction(
                'tpag', (1, 3), (5, ), 2.610575e10, 1.5, -229.93039,  
                frwrd_rate_param_func = PArth.tpag_frwrd_rate
            )
            key_rxns_dict['tdan'] = Reaction(
                'tdan', (2, 3), (0, 5), 5.5354059, 0., -204.11537, 
                frwrd_rate_param_func = PArth.tdan_frwrd_rate
            )
            key_rxns_dict['taLi7g'] = Reaction(
                'taLi7g', (3, 5), (6, ), 1.1132988e10, 1.5, -28.635551, 
                frwrd_rate_param_func = PArth.taLi7g_frwrd_rate
            )
            key_rxns_dict['He3ntp'] = Reaction(
                'He3ntp', (0, 4), (1, 3), 1.001785, 0., -8.8630042, 
                frwrd_rate_param_func = PArth.He3ntp_frwrd_rate
            )
            key_rxns_dict['He3dap'] = Reaction(
                'He3dap', (2, 4), (1, 5), 5.5452865, 0., -212.97837, 
                frwrd_rate_param_func = PArth.He3dap_frwrd_rate
            )
            key_rxns_dict['He3aBe7g'] = Reaction(
                'He3aBe7g', (4, 5), (7, ), 1.1128943e10, 1.5, -18.417922, 
                frwrd_rate_param_func = PArth.He3aBe7g_frwrd_rate
            )
            key_rxns_dict['Be7nLi7p'] = Reaction(
                'Be7nLi7p', (0, 7), (1, 6), 1.0021491, 0., -19.080633, 
                frwrd_rate_param_func = PArth.Be7nLi7p_frwrd_rate
            )
            key_rxns_dict['Li7paa'] = Reaction(
                'Li7paa', (1, 6), (5, 5), 4.6898011, 0., -201.29484, 
                frwrd_rate_param_func = PArth.Li7paa_frwrd_rate
            )


        if network[:4] == 'full': 

            other_rxns_dict = {}

            dir_name = 'other_PRIMAT_2023/'

            other_rxns_dict['Li7paag'] = Reaction(
                'Li7paag', (1, 6), (5, 5), 4.6898, 0.,-201.295, 
                spline_data=dir_name+'Li7paag.txt',
                interp_type=self.interp_type
            )
            other_rxns_dict['Be7naa'] = Reaction(
                'Be7naa', (0, 7), (5, 5), 4.6982, 0.,-220.3871, 
                spline_data=dir_name+'Be7naa.txt',
                interp_type=self.interp_type
            )
            other_rxns_dict['Be7daap'] = Reaction(
                'Be7daap', (2, 7), (1, 5, 5), 9.9579e-10, -1.5, -194.5722, 
                spline_data=dir_name+'Be7daap.txt',
                interp_type=self.interp_type
            ) 
            other_rxns_dict['daLi6g'] = Reaction(
                'daLi6g', (2, 5), (8, ), 1.53053e10, 1.5, -17.1023,
                spline_data=dir_name+'daLi6g.txt',
                interp_type=self.interp_type
            ) 
            other_rxns_dict['Li6pBe7g'] = Reaction(
                'Li6pBe7g', (1, 8), (7, ), 1.18778e10, 1.5, -65.0648, 
                spline_data=dir_name+'Li6pBe7g.txt',
                interp_type=self.interp_type
            ) 
            other_rxns_dict['Li6pHe3a'] = Reaction(
                'Li6pHe3a', (1, 8), (4, 5), 1.06729, 0., -46.6469, 
                spline_data=dir_name+'Li6pHe3a.txt',
                interp_type=self.interp_type
            ) 
            other_rxns_dict['B8naap'] = Reaction(
                'B8naap', (0, 11), (1, 5, 5), 3.6007e-10, -1.5, -218.7915, 
                spline_data=dir_name+'B8naap.txt',
                interp_type=self.interp_type
            )
            other_rxns_dict['Li6He3aap'] = Reaction(
                'Li6He3aap', (4, 8), (1, 5, 5), 7.2413e-10, -1.5, -195.8748, 
                spline_data=dir_name+'Li6He3aap.txt',
                interp_type=self.interp_type
            )
            other_rxns_dict['Li6taan'] = Reaction(
                'Li6taan', (3, 8), (0, 5, 5), 7.2333e-10, -1.5, -187.0131, 
                spline_data=dir_name+'Li6taan.txt',
                interp_type=self.interp_type
            )
            other_rxns_dict['Li6tLi8p'] = Reaction(
                'Li6tLi8p', (3, 8), (1, 10), 2.0167, 0., -9.306, 
                spline_data=dir_name+'Li6tLi8p.txt',
                interp_type=self.interp_type
            )
            other_rxns_dict['Li7He3Li6a'] = Reaction(
                'Li7He3Li6a', (4, 6), (5, 8), 2.1972, 0., -154.6607, 
                spline_data=dir_name+'Li7He3Li6a.txt',
                interp_type=self.interp_type
            )
            other_rxns_dict['Li8He3Li7a'] = Reaction(
                'Li8He3Li7a', (4, 10), (5, 6), 1.9994, 0., -215.2055, 
                spline_data=dir_name+'Li8He3Li7a.txt',
                interp_type=self.interp_type
            )
            other_rxns_dict['Be7tLi6a'] = Reaction(
                'Be7tLi6a', (3, 7), (5, 8), 2.1977, 0., -164.8783, 
                spline_data=dir_name+'Be7tLi6a.txt',
                interp_type=self.interp_type
            )
            other_rxns_dict['B8tBe7a'] = Reaction(
                'B8tBe7a', (3, 11), (5, 7), 1.9999, 0., -228.3344, 
                spline_data=dir_name+'B8tBe7a.txt',
                interp_type=self.interp_type
            )
            other_rxns_dict['B8nLi6He3'] = Reaction(
                'B8nLi6He3', (0, 11), (4, 8), 0.49669, 0., -22.9167, 
                spline_data=dir_name+'B8nLi6He3.txt',
                interp_type=self.interp_type
            )
            other_rxns_dict['B8nBe7d'] = Reaction(
                'B8nBe7d', (0, 11), (2, 7), 0.36119, 0., -24.2194, 
                spline_data=dir_name+'B8nBe7d.txt',
                interp_type=self.interp_type
            )
            other_rxns_dict['Li6tLi7d'] = Reaction(
                'Li6tLi7d', (3, 8), (2, 6), 0.72734, 0., -11.5332, 
                spline_data=dir_name+'Li6tLi7d.txt',
                interp_type=self.interp_type
            )
            other_rxns_dict['Li6He3Be7d'] = Reaction(
                'Li6He3Be7d', (4, 8), (2, 7), 0.72719, 0., -1.3157, 
                spline_data=dir_name+'Li6He3Be7d.txt',
                interp_type=self.interp_type
            )
            other_rxns_dict['Li7He3aad'] = Reaction(
                'Li7He3aad', (4, 6), (2, 5, 5), 2.8700e-10, -1.5, -137.5575, 
                spline_data=dir_name+'Li7He3aad.txt',
                interp_type=self.interp_type
            )
            other_rxns_dict['Li8He3aat'] = Reaction(
                'Li8He3aat', (4, 10), (3, 5, 5), 3.5907e-10, -1.5, -186.5821, 
                spline_data=dir_name+'Li8He3aat.txt',
                interp_type=self.interp_type
            )
            other_rxns_dict['Be7taad'] = Reaction(
                'Be7taad', (3, 7), (2, 5, 5), 2.8706e-10, -1.5, -147.7751, 
                spline_data=dir_name+'Be7taad.txt',
                interp_type=self.interp_type
            )
            other_rxns_dict['Be7tLi7He3'] = Reaction(
                'Be7tLi7He3', (3, 7), (4, 6), 1.0002, 0., -10.2176, 
                spline_data=dir_name+'Be7tLi7He3.txt',
                interp_type=self.interp_type
            )
            other_rxns_dict['B8dBe7He3'] = Reaction(
                'B8dBe7He3', (2, 11), (4, 7), 1.2514, 0, -62.1535, 
                spline_data=dir_name+'B8dBe7He3.txt',
                interp_type=self.interp_type
            )
            other_rxns_dict['B8taaHe3'] = Reaction(
                'B8taaHe3', (3, 11), (4, 5, 5), 3.5922e-10, -1.5, -209.9285, 
                spline_data=dir_name+'B8taaHe3.txt',
                interp_type=self.interp_type
            )
            other_rxns_dict['Be7He3ppaa'] = Reaction(
                'Be7He3ppaa', (4, 7), (1, 1, 5, 5), 1.2201e-19, -3., -130.8113, 
                spline_data=dir_name+'Be7He3ppaa.txt',
                interp_type=self.interp_type
            )
            other_rxns_dict['ddag'] = Reaction(
                'ddag', (2, 2), (5, ), 4.5310e10, 1.5, -276.7271, 
                spline_data=dir_name+'ddag.txt',
                interp_type=self.interp_type
            )
            other_rxns_dict['He3He3app'] = Reaction(
                'He3He3app', (4, 4), (1, 1, 5), 3.3915e-10, -1.5, -149.2290, 
                spline_data=dir_name+'He3He3app.txt',
                interp_type=self.interp_type
            )
            other_rxns_dict['Be7pB8g'] = Reaction(
                'Be7pB8g', (1, 7), (11, ), 1.3063e10, 1.5, -1.5825, 
                spline_data=dir_name+'Be7pB8g.txt',
                interp_type=self.interp_type
            ) 
            other_rxns_dict['Li7daan'] = Reaction(
                'Li7daan', (2, 6), (0, 5, 5), 9.9435e-10, -1.5, -175.4916, 
                spline_data=dir_name+'Li7daan.txt',
                interp_type=self.interp_type
            )
            other_rxns_dict['dntg'] =   Reaction(
                'dntg', (0, 2), (3, ), 1.6364262e10, 1.5, -72.612132, 
                spline_data=dir_name+'dntg.txt',
                interp_type=self.interp_type
            )
            other_rxns_dict['ttann'] =   Reaction(
                'ttann', (3, 3), (0, 0, 5), 3.3826187e-10, -1.5, -131.50322, 
                spline_data=dir_name+'ttann.txt',
                interp_type=self.interp_type
            )
            other_rxns_dict['He3nag'] = Reaction(
                'He3nag', (0, 4), (5, ), 2.6152351e10, 1.5, -238.79338, 
                spline_data=dir_name+'He3nag.txt',
                interp_type=self.interp_type    
            )
            other_rxns_dict['He3tad'] = Reaction(
                'He3tad', (3, 4), (2, 5), 1.5981381, 0., -166.18124, 
                spline_data=dir_name+'He3tad.txt',
                interp_type=self.interp_type
            )
            other_rxns_dict['He3tanp'] = Reaction(
                'He3tanp', (3, 4), (0, 1, 5), 3.3886566e-10, -1.5, -140.36623, 
                spline_data=dir_name+'He3tanp.txt',
                interp_type=self.interp_type
            )
            other_rxns_dict['Li7taann'] = Reaction(
                'Li7taann', (3, 6), (0, 0, 5, 5), 
                1.2153497e-19, -3., -102.86767, 
                spline_data=dir_name+'Li7taann.txt',
                interp_type=self.interp_type
            )                                               
            other_rxns_dict['Li7He3aanp'] = Reaction(
                'Li7He3aanp', (4, 6), (0, 1, 5, 5), 6.0875952e-20, -3., -111.73068, 
                spline_data=dir_name+'Li7He3aanp.txt',
                interp_type=self.interp_type
            )
            other_rxns_dict['Li8dLi7t'] = Reaction(
                'Li8dLi7t', (2, 10), (3, 6), 1.2509926, 0., -49.02453, 
                spline_data=dir_name+'Li8dLi7t.txt',
                interp_type=self.interp_type
            )
            other_rxns_dict['Be7taanp'] = Reaction(
                'Be7taanp', (3, 7), (0, 1, 5, 5), 
                6.0898077e-20, -3., -121.9483, 
                spline_data=dir_name+'Be7taanp.txt',
                interp_type=self.interp_type
            )
            other_rxns_dict['Li6nta'] = Reaction(
                'Li6nta', (0, 8), (3, 5), 1.0691921, 0., -55.509875, 
                spline_data=dir_name+'Li6nta.txt',
                interp_type=self.interp_type
            )
            other_rxns_dict['He3tLi6g'] = Reaction(
                'He3tLi6g', (3, 4), (8, ), 2.4459918e10, 1.5, -183.2835, 
                spline_data=dir_name+'He3tLi6g.txt',
                interp_type=self.interp_type
            )
            other_rxns_dict['anpLi6g'] = Reaction(
                'anpLi6g', (0, 1, 5), (8, ), 7.2181753e19, 3., -42.917276, 
                spline_data=dir_name+'anpLi6g.txt',
                interp_type=self.interp_type
            )
            other_rxns_dict['Li6nLi7g'] = Reaction(
                'Li6nLi7g', (0, 8), (6, ), 1.1903305e10, 1.5, -84.145424, 
                spline_data=dir_name+'Li6nLi7g.txt',
                interp_type=self.interp_type
            )
            other_rxns_dict['Li6dLi7p'] = Reaction(
                'Li6dLi7p', (2, 8), (1, 6), 2.5239503, 0., -58.330405, 
                spline_data=dir_name+'Li6dLi7p.txt',
                interp_type=self.interp_type
            )
            other_rxns_dict['Li6dBe7n'] = Reaction(
                'Li6dBe7n', (2, 8), (0, 7), 2.5185377, 0., -39.249773, 
                spline_data=dir_name+'Li6dBe7n.txt',
                interp_type=self.interp_type
            )
            other_rxns_dict['Li7nLi8g'] = Reaction(
                'Li7nLi8g', (0, 6), (10, ), 1.3081022e10, 1.5, -23.587602, 
                spline_data=dir_name+'Li7nLi8g.txt',
                interp_type=self.interp_type
            )
            other_rxns_dict['Li7dLi8p'] = Reaction(
                'Li7dLi8p', (2, 6), (1, 10), 2.7736709, 0., 2.2274166, 
                spline_data=dir_name+'Li7dLi8p.txt',
                interp_type=self.interp_type
            )
            other_rxns_dict['Li8paan'] = Reaction(
                'Li8paan', (1, 10), (0, 5, 5), 3.5851946e-10, -1.5, -177.70722, 
                spline_data=dir_name+'Li8paan.txt',
                interp_type=self.interp_type
            )
            other_rxns_dict['annHe6g'] = Reaction(
                'annHe6g', (0, 0, 5), (9, ), 1.0837999e20, 3., -11.319626, 
                spline_data=dir_name+'annHe6g.txt',
                interp_type=self.interp_type
            )
            other_rxns_dict['ppndp'] = Reaction(
                'ppndp', (1, 1, 0), (1, 2), 2.3580703e9, 1.5, -25.815019, 
                spline_data=dir_name+'ppndp.txt',
                interp_type=self.interp_type
            )

        if network[:5] == 'small': 

            other_rxns_dict = {}

            dir_name = 'other_PRIMAT_2023/' 

            other_rxns_dict['Li7paag'] = Reaction(
                'Li7paag', (1, 6), (5, 5), 4.6898, 0.,-201.295, 
                spline_data=dir_name+'Li7paag.txt',
                interp_type=self.interp_type
            )
            other_rxns_dict['Be7naa'] = Reaction(
                'Be7naa', (0, 7), (5, 5), 4.6982, 0.,-220.3871, 
                spline_data=dir_name+'Be7naa.txt',
                interp_type=self.interp_type
            )
            other_rxns_dict['Be7daap'] = Reaction(
                'Be7daap', (2, 7), (1, 5, 5), 9.9579e-10, -1.5, -194.5722, 
                spline_data=dir_name+'Be7daap.txt',
                interp_type=self.interp_type
            ) 
            other_rxns_dict['daLi6g'] = Reaction(
                'daLi6g', (2, 5), (8, ), 1.53053e10, 1.5, -17.1023,
                spline_data=dir_name+'daLi6g.txt',
                interp_type=self.interp_type
            ) 
            other_rxns_dict['Li6pBe7g'] = Reaction(
                'Li6pBe7g', (1, 8), (7, ), 1.18778e10, 1.5, -65.0648, 
                spline_data=dir_name+'Li6pBe7g.txt',
                interp_type=self.interp_type
            ) 
            other_rxns_dict['Li6pHe3a'] = Reaction(
                'Li6pHe3a', (1, 8), (4, 5), 1.06729, 0., -46.6469, 
                spline_data=dir_name+'Li6pHe3a.txt',
                interp_type=self.interp_type
            ) 
            other_rxns_dict['Li6He3aap'] = Reaction(
                'Li6He3aap', (4, 8), (1, 5, 5), 7.2413e-10, -1.5, -195.8748, 
                spline_data=dir_name+'Li6He3aap.txt',
                interp_type=self.interp_type
            )
            other_rxns_dict['Li6taan'] = Reaction(
                'Li6taan', (3, 8), (0, 5, 5), 7.2333e-10, -1.5, -187.0131, 
                spline_data=dir_name+'Li6taan.txt',
                interp_type=self.interp_type
            )
            other_rxns_dict['Li7He3Li6a'] = Reaction(
                'Li7He3Li6a', (4, 6), (5, 8), 2.1972, 0., -154.6607, 
                spline_data=dir_name+'Li7He3Li6a.txt',
                interp_type=self.interp_type
            )
            other_rxns_dict['Be7tLi6a'] = Reaction(
                'Be7tLi6a', (3, 7), (5, 8), 2.1977, 0., -164.8783, 
                spline_data=dir_name+'Be7tLi6a.txt',
                interp_type=self.interp_type
            )
            other_rxns_dict['Li6tLi7d'] = Reaction(
                'Li6tLi7d', (3, 8), (2, 6), 0.72734, 0., -11.5332, 
                spline_data=dir_name+'Li6tLi7d.txt',
                interp_type=self.interp_type
            )
            other_rxns_dict['Li6He3Be7d'] = Reaction(
                'Li6He3Be7d', (4, 8), (2, 7), 0.72719, 0., -1.3157, 
                spline_data=dir_name+'Li6He3Be7d.txt',
                interp_type=self.interp_type
            )
            other_rxns_dict['Li7He3aad'] = Reaction(
                'Li7He3aad', (4, 6), (2, 5, 5), 2.8700e-10, -1.5, -137.5575, 
                spline_data=dir_name+'Li7He3aad.txt',
                interp_type=self.interp_type
            )
            other_rxns_dict['Be7taad'] = Reaction(
                'Be7taad', (3, 7), (2, 5, 5), 2.8706e-10, -1.5, -147.7751, 
                spline_data=dir_name+'Be7taad.txt',
                interp_type=self.interp_type
            )
            other_rxns_dict['Be7tLi7He3'] = Reaction(
                'Be7tLi7He3', (3, 7), (4, 6), 1.0002, 0., -10.2176, 
                spline_data=dir_name+'Be7tLi7He3.txt',
                interp_type=self.interp_type
            )
            other_rxns_dict['Be7He3ppaa'] = Reaction(
                'Be7He3ppaa', (4, 7), (1, 1, 5, 5), 1.2201e-19, -3., -130.8113, 
                spline_data=dir_name+'Be7He3ppaa.txt',
                interp_type=self.interp_type
            )
            other_rxns_dict['ddag'] = Reaction(
                'ddag', (2, 2), (5, ), 4.5310e10, 1.5, -276.7271, 
                spline_data=dir_name+'ddag.txt',
                interp_type=self.interp_type
            )
            other_rxns_dict['He3He3app'] = Reaction(
                'He3He3app', (4, 4), (1, 1, 5), 3.3915e-10, -1.5, -149.2290, 
                spline_data=dir_name+'He3He3app.txt',
                interp_type=self.interp_type
            )
            other_rxns_dict['Li7daan'] = Reaction(
                'Li7daan', (2, 6), (0, 5, 5), 9.9435e-10, -1.5, -175.4916, 
                spline_data=dir_name+'Li7daan.txt',
                interp_type=self.interp_type
            )
            other_rxns_dict['dntg'] =   Reaction(
                'dntg', (0, 2), (3, ), 1.6364262e10, 1.5, -72.612132, 
                spline_data=dir_name+'dntg.txt',
                interp_type=self.interp_type
            )
            other_rxns_dict['ttann'] =   Reaction(
                'ttann', (3, 3), (0, 0, 5), 3.3826187e-10, -1.5, -131.50322, 
                spline_data=dir_name+'ttann.txt',
                interp_type=self.interp_type
            )
            other_rxns_dict['He3nag'] = Reaction(
                'He3nag', (0, 4), (5, ), 2.6152351e10, 1.5, -238.79338, 
                spline_data=dir_name+'He3nag.txt',
                interp_type=self.interp_type    
            )
            other_rxns_dict['He3tad'] = Reaction(
                'He3tad', (3, 4), (2, 5), 1.5981381, 0., -166.18124, 
                spline_data=dir_name+'He3tad.txt',
                interp_type=self.interp_type
            )
            other_rxns_dict['He3tanp'] = Reaction(
                'He3tanp', (3, 4), (0, 1, 5), 3.3886566e-10, -1.5, -140.36623, 
                spline_data=dir_name+'He3tanp.txt',
                interp_type=self.interp_type
            )
            other_rxns_dict['Li7taann'] = Reaction(
                'Li7taann', (3, 6), (0, 0, 5, 5), 
                1.2153497e-19, -3., -102.86767, 
                spline_data=dir_name+'Li7taann.txt',
                interp_type=self.interp_type
            )                                               
            other_rxns_dict['Li7He3aanp'] = Reaction(
                'Li7He3aanp', (4, 6), (0, 1, 5, 5), 6.0875952e-20, -3., -111.73068, 
                spline_data=dir_name+'Li7He3aanp.txt',
                interp_type=self.interp_type
            )
            other_rxns_dict['Be7taanp'] = Reaction(
                'Be7taanp', (3, 7), (0, 1, 5, 5), 
                6.0898077e-20, -3., -121.9483, 
                spline_data=dir_name+'Be7taanp.txt',
                interp_type=self.interp_type
            )
            other_rxns_dict['Li6nta'] = Reaction(
                'Li6nta', (0, 8), (3, 5), 1.0691921, 0., -55.509875, 
                spline_data=dir_name+'Li6nta.txt',
                interp_type=self.interp_type
            )
            other_rxns_dict['He3tLi6g'] = Reaction(
                'He3tLi6g', (3, 4), (8, ), 2.4459918e10, 1.5, -183.2835, 
                spline_data=dir_name+'He3tLi6g.txt',
                interp_type=self.interp_type
            )
            other_rxns_dict['anpLi6g'] = Reaction(
                'anpLi6g', (0, 1, 5), (8, ), 7.2181753e19, 3., -42.917276, 
                spline_data=dir_name+'anpLi6g.txt',
                interp_type=self.interp_type
            )
            other_rxns_dict['Li6nLi7g'] = Reaction(
                'Li6nLi7g', (0, 8), (6, ), 1.1903305e10, 1.5, -84.145424, 
                spline_data=dir_name+'Li6nLi7g.txt',
                interp_type=self.interp_type
            )
            other_rxns_dict['Li6dLi7p'] = Reaction(
                'Li6dLi7p', (2, 8), (1, 6), 2.5239503, 0., -58.330405, 
                spline_data=dir_name+'Li6dLi7p.txt',
                interp_type=self.interp_type
            )
            other_rxns_dict['Li6dBe7n'] = Reaction(
                'Li6dBe7n', (2, 8), (0, 7), 2.5185377, 0., -39.249773, 
                spline_data=dir_name+'Li6dBe7n.txt',
                interp_type=self.interp_type
            )
            other_rxns_dict['ppndp'] = Reaction(
                'ppndp', (1, 1, 0), (1, 2), 2.3580703e9, 1.5, -25.815019, 
                spline_data=dir_name+'ppndp.txt',
                interp_type=self.interp_type
            )

        if network[:3] == 'key': 

            return list(key_rxns_dict.values())
        
        else: 

            return (
                list(key_rxns_dict.values()) + list(other_rxns_dict.values())
            )
        


