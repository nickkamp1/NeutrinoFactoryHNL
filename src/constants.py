# constants

import numpy as np


N_muon_decays = 1e22 # typical # of muon decays in 10 years at a neutrino factory
E_muon = 25. # muon energy in GeV


det_min_baseline = 0 # average detector baseline in meters
det_max_baseline = 755 # average detector baseline in meters
det_length = 100.0 # detector length in meters
det_radius = 5.0 # detector radius in meters

G_F = 1.1663787e-5 # Fermi coupling constant in GeV^-2
m_mu = 0.1056583745 # muon mass in GeV
m_e = 0.0005109989461 # electron mass in GeV
sin2_theta_W = 0.23126 # weak mixing angle
alpha = 1/137.035999084 # fine-structure constant
speed_of_light = 3e8 # speed of light in m/s
hbar_in_GeV_s = 6.582119569e-25 # hbar in GeV*s


# units
picobarn_to_cm2 = 1e-36 # cm^2
picobarn_to_m2 = 1e-40 # m^2

# Cherenkov light
n_air = 1.0003 # index of refraction
theta_C = np.arccos(1./n_air) # cherenkov angle
Ch_dN_dx = 2*np.pi/137 * np.sin(theta_C)**2 * (1/300 - 1/1000) * 1e9 # in m^-1

# balloon
L_det = 33e3 # m
R_det = 2 # m

# earth
m_nucleon = 0.938 # GeV (proton mass)
N_avogadro = 6.022e23 # mol^-1
rho_earth = 2.65 # g/cm^3
A_earth = 1 # g/mol for average nucleon
n_earth = N_avogadro * rho_earth / A_earth # nucleons/cm^3
n_earth_m3 = n_earth * 1e6 # nucleons/m^3