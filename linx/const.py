import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True) # need this to enable float64

from linx.special_funcs import zeta_3 

############################
#    Conversion Factors    #
############################

hbar = 6.582119 * 1e-16 * 1e-6 # s MeV
kB   = 1. / (1.160451812 * 1e4 * 1e6)  # MeV / K
c    = 2.99792458e10 # cm / s

MeV4_to_gcmm3 = 232011.501875
km_to_Mpc = 1e3 * 1e2 / (3.0856777807e18 * 1e6)

##########################
#    Masses              #
##########################

# All masses in MeV, according to PDG 2020. 

# Electron mass
me = 0.51099895 

# Neutron mass
mn = 939.56542052

# Proton mass
mp = 938.27208816

# W mass
mW = 80.379*1e3

# Z mass
mZ = 91.1876*1e3

# Atomic unit
ma = 931.494061

# Hydrogen atom mass
mH = 1.00782503223 * ma 

# He-4 atom mass 
mHe4 = 4.0026032541 * ma

# The baryon mass is a weighted average of hydrogen and helium, 
# and therefore depends on the relative abundance. But we need this to convert
# between Omegabh2 and the baryon-to-photon ratio! At the current levels of 
# of precision we can just take the rough predicted Y_p. 

# Approximate He4 abundance by mass
# Y_p = 4 * nHe4 / n_b
Y_p_0 = 24.7/100.

# Baryon mass 
# 1 - Y_p = nH / n_b
mbaryon = (1. - Y_p_0)*mH + Y_p_0*mHe4/4.

# Neutron decay lifetime and uncertainty, in seconds.
tau_n = 879.4
sigma_tau_n = 0.6

# Fine-structure constant.
aFS = 1./137.035999084 # fine structure constant 

# G_Fermi, experimentally measured 
GF = 1.1663787e-5*1.e-6 # MeV-2 

# sin^2(theta_Weinberg), tree-level. 
sW2 = 0.5*(1.-jnp.sqrt(1.-2.*jnp.sqrt(2.)*jnp.pi*aFS/(GF*mZ**2)))

# Electron and muon coupling to Z 
geL, geR, gmuL, gmuR = 1./2.+sW2, sW2, -1./2.+sW2, sW2

# G_Newton in MeV^-2
GN  = 6.70883e-39*1e-6
# Planck mass in MeV 
Mpl = 1./jnp.sqrt(GN)


# Constants inherent to n <--> p weak rates

gA = 1.2756 # Axial current constant of structure of nucleons PDG2020
radproton = 0.841*1.e-15 * 1e2 # proton radius in cm arXiv:1212.0332

#################################
#    Cosmological Parameters    #
#################################

# CMB temperature today in MeV. 
T0CMB = 2.7255 * kB

# Photon number density today in MeV^3. 
n0CMB = 2. * zeta_3 / (jnp.pi**2) * T0CMB**3 

# 100 km/sec/Mpc in MeV 
HubbleOverh = 100 * km_to_Mpc * hbar 

# Critical density today / h in MeV^4
rhocOverh2 = 3. / (8. * jnp.pi * GN) * HubbleOverh**2

# baryon abundance measured from Planck 2018 TTTEEE+lowE+lensing+BAO
Omegabh2 = 0.02242
sigma_Omegabh2 = 0.00014
Omegabh2_to_eta0 = rhocOverh2 / n0CMB / mbaryon 
eta0 = Omegabh2_to_eta0 * Omegabh2
sigma_eta0 = Omegabh2_to_eta0 * sigma_Omegabh2


#######################################
#    Important Temperature Regimes    #
#######################################

T_start = 1e11 * kB # MeV  (8.6 MeV)
# T_start to T_Middle: neutron freezeout, only n, p important to track. 
T_middle = 1e10 * kB # MeV (0.86 MeV)
# T_Middle to T_switch: small network always sufficient. 
T_switch = 1.25e9 * kB # MeV
# T_switch to T_end: turn on full network if desired. 
T_end = 6.e7 * kB # MeV