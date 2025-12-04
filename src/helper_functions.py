import numpy as np
from scipy.integrate import quad
from src.constants import *


def muon_differential_decay_width(E_N, m_N, Umu2, Ue2):
    """
    Calculate the differential decay width of a muon with HNL in final state.

    Parameters:
    E_N : float
        Energy of the Heavy Neutral Lepton (HNL) in GeV.
    m_N : float
        Mass of the Heavy Neutral Lepton (HNL) in GeV.
    Umu2 : float
        Muon mixing parameter squared.
    Ue2 : float
        Electron mixing parameter squared.

    Returns:
    float
        Differential decay width in GeV.
    """

    kinematic_flag = np.logical_or(E_N < m_N, E_N > (m_mu**2 + m_N**2) / (2 * m_mu))

    # muon mixing term
    prefactor = (G_F**2 * Umu2) / (12 * np.pi**3)
    term1 = 3*E_N * (m_mu**2 + m_N**2)
    term2 = 4 * m_mu * E_N**2
    term3 = 2 * m_mu * m_N**2
    p_N = np.sqrt(E_N**2 - m_N**2)
    mu_term = prefactor*(term1-term2-term3)*p_N

    # electron mixing term
    prefactor = (G_F**2 * Ue2 * E_N) / (2 * np.pi**3)
    term1 = m_mu**2
    term2 = m_N**2
    term3 = 2*m_mu*E_N
    p_N = np.sqrt(E_N**2 - m_N**2)
    e_term = prefactor*(term1+term2-term3)*p_N
    dGamma = np.where(kinematic_flag, 0, mu_term + e_term)
    return dGamma

def muon_total_decay_width(m_N, Umu2, Ue2):
    """
    Calculate the total decay width of a muon with HNL in final state.

    Parameters:
    m_N : float
        Mass of the Heavy Neutral Lepton (HNL) in GeV.
    Umu2 : float
        Muon mixing parameter squared.
    Ue2 : float
        Electron mixing parameter squared.

    Returns:
    float
        Total decay width in GeV.
    """

    lower_bound = m_N
    upper_bound = (m_mu**2 + m_N**2)/(2*m_mu)
    result, error = quad(muon_differential_decay_width, lower_bound, upper_bound, args=(m_N, Umu2, Ue2))
    return result

Gamma_muon = muon_total_decay_width(0,1,0)


def HNL_decay_width(m_N, U2, d=0):
    """Calculate the decay width of a Heavy Neutral Lepton (HNL).

    Parameters:
    m_N : float
        Mass of the HNL in GeV.
    U2 : float
        Mixing parameter squared.
    d : float
        Transition magnetic moment in GeV^-1.

    Returns:
    float
        Decay width in GeV.
    """
    G_F = 1.1663787e-5 # Fermi coupling constant in GeV^-2
    mixing_gamma = (G_F**2 * m_N**5 * U2) / (192 * np.pi**3)
    magnetic_gamma = (d**2 * m_N**3) / (4 * np.pi)
    return mixing_gamma + magnetic_gamma


def HNL_decay_width_approx(m_N, U2):
    Gamma_mu = muon_total_decay_width(0,1,0)
    return Gamma_mu*(m_N/m_mu)**5 * U2

def HNL_decay_length(m_N, U2, E_N , d=0):
    """Calculate the decay length of a Heavy Neutral Lepton (HNL).

    Parameters:
    m_N : float
        Mass of the HNL in GeV.
    U2 : float
        Mixing parameter squared.
    E_N : float
        Energy of the HNL in GeV.
    d : float
        Transition magnetic moment in GeV^-1.

    Returns:
    float
        Decay length in meters.
    """
    decay_width = HNL_decay_width(m_N, U2, d)
    gamma = E_N / m_N

    return (hbar_in_GeV_s * c * gamma) / decay_width

def HNL_ee_decay_width(m_N, Umu2, Ue2, d=0):
    """Calculate the decay width of HNL to e+ e- nu.

    Parameters:
    m_N : float
        Mass of the HNL in GeV.
    Umu2 : float
        Muon mixing parameter squared.
    Ue2 : float
        Electron mixing parameter squared.
    d : float
        Transition magnetic moment in GeV^-1.

    Returns:
    float
        Decay width in GeV.
    """
    G_F = 1.1663787e-5 # Fermi coupling constant in GeV^-2
    prefactor = (G_F**2 * m_N**5) / (768 * np.pi**3)
    term_mu = Umu2 * (1 - 4 * sin2_theta_W + 8 * sin2_theta_W**2)
    term_e = Ue2 * (1 + 4 * sin2_theta_W + 8 * sin2_theta_W**2)
    mixing_gamma = prefactor * (term_mu + term_e)

    r = m_e/m_N
    L = np.zeros_like(r)
    if 2*r > 1:
        L = 0
    elif r < 1e-1:
        L = 2*np.log(1/r) - 3.5
    else:
        L = (2 - (r**6)/8) * np.cosh(r) - (24 - 10*r**2 + r**4) / 8 * np.sqrt(1 - 4*r**2)
    magnetic_gamma = alpha*d**2 * m_N**3 / (12 * np.pi) * L
    return mixing_gamma + magnetic_gamma

def expected_HNL_events(m_N, Umu2, Ue2, d = 0, det_eff = 1):
    """Calculate the expected number of HNL decay events in the detector.

    Parameters:
    m_N : float
        Mass of the HNL in GeV.
    Ue2 : float
        Electron mixing parameter squared.
    Umu2 : float
        Muon mixing parameter squared.

    Returns:
    float
        Expected number of events.
    """

    prefactor = N_muon_decays * det_eff / Gamma_muon

    dw = HNL_decay_width(m_N, Umu2+Ue2, d=d)
    ee_dw = HNL_ee_decay_width(m_N, Umu2, Ue2, d=d)

    def integrand(E_N):
        dGamma_mu = muon_differential_decay_width(E_N, m_N, Umu2, Ue2)
        E_N_lab = E_muon/m_mu * E_N
        beta_gamma = np.sqrt(E_N_lab**2 - m_N**2) / m_N
        ee_decay_length = hbar_in_GeV_s * c * beta_gamma / ee_dw
        decay_factor = det_length / ee_decay_length
        return dGamma_mu * decay_factor


    lower_bound = m_N
    upper_bound = (m_mu**2 + m_N**2)/(2*m_mu)
    result, error = quad(integrand, lower_bound, upper_bound)

    return prefactor*result

def get_frac_events_within_transverse_displacement(E_muon,m_N,det_radius,N_rand=1e5,mixing="mu"):
    E_N_range = np.linspace(m_N, (m_mu**2 + m_N**2)/(2*m_mu), 1000)
    dG_dE = muon_differential_decay_width(E_N_range,m_N,1,0) if mixing=="mu" else muon_differential_decay_width(E_N_range,m_N,0,1)
    dG_dE = np.where(dG_dE < 0, 0, dG_dE) # set negative values to zero
    dG_dE = np.where(np.isnan(dG_dE), 0, dG_dE) # set NaN values to zero
    CosTheta_rest = np.random.uniform(-1, 1, int(N_rand))
    Phi_rest = np.random.uniform(0, 2*np.pi, int(N_rand))
    E_N_sample_rest = np.random.choice(E_N_range, size=int(N_rand), p=dG_dE/np.sum(dG_dE))
    det_baseline = np.random.uniform(det_min_baseline,det_max_baseline, int(N_rand))

    # now boost the 4-momenta
    gamma = E_muon / m_mu
    beta = np.sqrt(1 - 1/gamma**2)

    # compute the 4-momentum in the rest frame
    p_N_rest = np.sqrt(E_N_sample_rest**2 - m_N**2)
    px_rest = p_N_rest * np.sin(np.arccos(CosTheta_rest)) * np.cos(Phi_rest)
    py_rest = p_N_rest * np.sin(np.arccos(CosTheta_rest)) * np.sin(Phi_rest)
    pz_rest = p_N_rest * CosTheta_rest
    E_N_rest = E_N_sample_rest

    # boost to lab frame
    E_N_lab = gamma * E_N_rest + beta*gamma * pz_rest
    p_N_lab = np.sqrt(E_N_lab**2 - m_N**2)
    pz_lab = beta * gamma * E_N_rest + (1 + ((gamma*beta)**2)/(1+gamma))*pz_rest
    CosTheta_lab = pz_lab / p_N_lab
    SinTheta_lab = np.sqrt(1 - CosTheta_lab**2)
    transverse_displacement = det_baseline * SinTheta_lab

    return sum(transverse_displacement < det_radius) / N_rand