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

    return (hbar_in_GeV_s * speed_of_light * gamma) / decay_width

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
    a = 1 - r**2
    L = (2 - (r**6)/8) * np.cosh(r) - (24 - 10*r**2 + r**4) / 8 * np.sqrt(1 - 4*r**2)
    L = np.where(r > 0.1, a**3 * np.sqrt(1 - 4*r**2)*(1./2. + r**2), L)
    L = np.where(r < 1e-1, 2*np.log(1/r) - 3, L)
    L = np.where(2*r > 1, 0, L)
    magnetic_gamma = alpha*d**2 * m_N**3 / (12 * np.pi) * L
    return mixing_gamma + magnetic_gamma

# cross section for mu nucleon to HNL nucleon
def sigma(E_mu, m_N, U2):
    s = 2 * m_nucleon * E_mu  # COM energy squared, nucleon at rest
    sqrt_s = np.sqrt(s)

    # Threshold behavior with smoothing
    # Use a form that transitions more smoothly near kinematic limits
    s_threshold = (m_N + m_mu)**2

    # Basic threshold factor
    if np.isscalar(m_N):
        if s <= s_threshold or m_N >= sqrt_s - m_mu:
            return 0.0
    else:
        return 0.67e-42 * E_mu * U2 * (1 - m_N**2/s)**2  # in m^2, with threshold behavior


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
        ee_decay_length = hbar_in_GeV_s * speed_of_light * beta_gamma / ee_dw
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


# ═══════════════════════════════════════════════════════════════════
# HNL decay branching ratios and multi-channel decay kinematics
# ═══════════════════════════════════════════════════════════════════

# Particle masses used for decay kinematics [GeV]
M_PI_CHARGED = 0.13957   # charged pion
M_PI_NEUTRAL = 0.13498   # neutral pion
M_ELECTRON = m_e
M_MUON = m_mu

# Decay channel labels
# CC channels (via U_mu): the HNL vertex produces a muon + virtual W
CHANNEL_MU_E_NU = "N->mu_e_nu"         # N -> mu- + e+ + nu_e (CC, 3-body leptonic)
CHANNEL_MU_PI = "N->mu_pi"             # N -> mu- + pi+ (CC, 2-body)
# CC channels (via U_e): the HNL vertex produces an electron + virtual W
CHANNEL_E_MU_NU = "N->e_mu_nu"         # N -> e- + mu+ + nu_mu (CC via Ue, 3-body)
CHANNEL_E_PI = "N->e_pi"               # N -> e- + pi+ (CC via Ue, 2-body)
# NC channels (via any U_alpha): the HNL vertex produces nu + virtual Z
CHANNEL_NU_EE = "N->nu_e_e"            # N -> nu + e+ + e- (NC, 3-body)
CHANNEL_NU_PI0 = "N->nu_pi0"           # N -> nu + pi0 (NC, 2-body)


def _kallen_lambda(a, b, c):
    """Kallen triangle function: lambda(a,b,c) = a^2 + b^2 + c^2 - 2ab - 2ac - 2bc."""
    return a**2 + b**2 + c**2 - 2*a*b - 2*a*c - 2*b*c


def _Gamma_N_nu_ll(m_N, m_l, U2):
    """
    Partial width for N -> nu_alpha + l^- + l^+ (NC-mediated, same-flavor pair).

    Uses the full expression from Bondarenko et al. (2018), Eq. (B.1)-(B.3),
    simplified for pure muon mixing with NC contribution.

    For N -> nu + e+ + e- (mediated by Z):
        Gamma ~ (G_F^2 m_N^5 U2) / (192 pi^3) * f(m_l/m_N)
    where f accounts for charged lepton mass phase space.
    """
    x_l = m_l / m_N
    if 2 * x_l >= 1:
        return 0.0

    # NC couplings for charged leptons
    gL = -0.5 + sin2_theta_W
    gR = sin2_theta_W

    prefactor = G_F**2 * m_N**5 * U2 / (192 * np.pi**3)

    # Phase-space integral approximation (leading order in m_l/m_N)
    # Full integral is complex; use approximate form valid for m_l << m_N
    if x_l < 0.01:
        # Massless limit
        f = (gL * gR + gR**2) + (gL**2 + gL * gR) + (gL + gR)/4
        # Simplified: for nu_mu + e+ e- via Z, the leading term
        f = (1 - 4 * sin2_theta_W + 8 * sin2_theta_W**2) / 4
    else:
        # Include phase space suppression for massive leptons
        f = (1 - 4 * sin2_theta_W + 8 * sin2_theta_W**2) / 4
        # Leading phase space correction
        f *= (1 - 4 * x_l**2)**0.5 * (1 - 14 * x_l**2 - 2 * x_l**4 - 12 * x_l**6)
        f += 12 * x_l**4 * (x_l**4 - 1) * np.log(
            (1 - 3 * x_l**2 - (1 - x_l**2) * np.sqrt(1 - 4 * x_l**2))
            / (x_l**2 * (1 + np.sqrt(1 - 4 * x_l**2))) + 1e-30
        )

    return max(0.0, prefactor * f)


def _Gamma_N_l_pi(m_N, m_l, U_l2):
    """
    Partial width for N -> l + pi (CC-mediated, 2-body).

    Gamma = (G_F^2 * f_pi^2 * |V_ud|^2 * U_l^2 * m_N^3) / (16 pi)
            * [(1 - x_l^2)^2 - x_pi^2 * (1 + x_l^2)]
            * sqrt(kallen(1, x_l^2, x_pi^2))

    where x_l = m_l/m_N, x_pi = m_pi/m_N.
    """
    x_l = m_l / m_N
    x_pi = M_PI_CHARGED / m_N

    if x_l + x_pi >= 1:
        return 0.0

    f_pi = 0.1302   # pion decay constant [GeV]
    V_ud2 = 0.9743**2  # |V_ud|^2

    kallen = _kallen_lambda(1, x_l**2, x_pi**2)
    if kallen <= 0:
        return 0.0

    prefactor = G_F**2 * f_pi**2 * V_ud2 * U_l2 * m_N**3 / (16 * np.pi)
    bracket = (1 - x_l**2)**2 - x_pi**2 * (1 + x_l**2)

    return max(0.0, prefactor * bracket * np.sqrt(kallen))


def _Gamma_N_nu_pi0(m_N, U2):
    """
    Partial width for N -> nu + pi0 (NC-mediated, 2-body).

    Gamma = (G_F^2 * f_pi^2 * U^2 * m_N^3) / (64 pi) * (1 - x_pi0^2)^2
    """
    x_pi0 = M_PI_NEUTRAL / m_N
    if x_pi0 >= 1:
        return 0.0

    f_pi = 0.1302   # pion decay constant [GeV]

    prefactor = G_F**2 * f_pi**2 * U2 * m_N**3 / (64 * np.pi)
    return prefactor * (1 - x_pi0**2)**2


def _Gamma_N_3nu(m_N, U2):
    """
    Partial width for N -> 3 neutrinos.

    Gamma = G_F^2 * m_N^5 * U^2 / (768 pi^3)
    """
    return G_F**2 * m_N**5 * U2 / (768 * np.pi**3)


def _Gamma_N_CC_leptonic(m_N, m_l_vertex, m_l_W, U_l2):
    """
    Partial width for CC 3-body leptonic decay:
        N -> l_vertex^- + l_W^+ + nu

    The HNL couples to l_vertex via U_l, producing l_vertex + W*.
    The virtual W decays to l_W + nu.
    Charge is conserved: N(0) -> l^-(−1) + l'^+(+1) + nu(0).

    Example: N -> mu^- + e^+ + nu_e  (U_mu vertex, W* -> e+ nu_e)

    Gamma = G_F^2 * m_N^5 * U_l^2 / (192 pi^3) * I(x_vertex, x_W)

    where x = m_l / m_N, and I is the phase space integral.
    For x_W << 1 (e.g. e from W*), I reduces to f(x_vertex).

    Parameters
    ----------
    m_N : float
        HNL mass [GeV]
    m_l_vertex : float
        Mass of the charged lepton at the HNL vertex [GeV]
    m_l_W : float
        Mass of the charged lepton from the W* decay [GeV]
    U_l2 : float
        Mixing parameter squared for the vertex lepton flavor

    Returns
    -------
    float : partial width [GeV]
    """
    if m_l_vertex + m_l_W >= m_N:
        return 0.0

    x_v = m_l_vertex / m_N
    x_w = m_l_W / m_N

    prefactor = G_F**2 * m_N**5 * U_l2 / (192 * np.pi**3)

    # Phase space function including both lepton masses
    # For the general case, we use the Dalitz integral.
    # When x_w is negligible, this reduces to the standard formula:
    #   f(x) = 1 - 8x^2 + 8x^6 - x^8 - 12x^4 ln(x^2)
    # For comparable masses, we use a numerical approximation.
    if x_w < 0.01:
        # Light W-daughter (e.g. electron): use standard formula with vertex lepton mass
        x = x_v
        f = 1 - 8*x**2 + 8*x**6 - x**8 - 12*x**4 * np.log(x**2 + 1e-30)
    elif x_v < 0.01:
        # Light vertex lepton (e.g. electron at vertex): use formula with W-daughter mass
        x = x_w
        f = 1 - 8*x**2 + 8*x**6 - x**8 - 12*x**4 * np.log(x**2 + 1e-30)
    else:
        # Both massive: use approximate 2-body-like suppression
        # This is a rough approximation; for precision, integrate the Dalitz plot
        x_eff = np.sqrt(x_v**2 + x_w**2)
        if x_eff >= 1:
            return 0.0
        f = 1 - 8*x_eff**2 + 8*x_eff**6 - x_eff**8 - 12*x_eff**4 * np.log(x_eff**2 + 1e-30)
        # Additional phase space suppression
        f *= np.sqrt(max(0, _kallen_lambda(1, x_v**2, x_w**2)))

    return max(0.0, prefactor * f)


def HNL_branching_ratios(m_N, Umu2=1.0, Ue2=0.0):
    """
    Compute HNL decay branching ratios as a function of mass.

    Assumes muon-mixing dominance (Umu2 >> Ue2) by default.
    For a Dirac HNL. (Majorana would double the CC widths.)

    Channels (for U_mu mixing):
      CC: N -> mu^- e^+ nu_e        (via U_mu, W* -> e+ nu_e)
      CC: N -> mu^- pi^+            (via U_mu, W* -> pi+)
      NC: N -> nu_mu e^+ e^-        (via U_mu, Z* -> e+e-)
      NC: N -> nu_mu pi^0           (via U_mu, Z* -> pi0)

    Additional channels if U_e != 0:
      CC: N -> e^- mu^+ nu_mu       (via U_e, W* -> mu+ nu_mu)
      CC: N -> e^- pi^+             (via U_e, W* -> pi+)

    Parameters
    ----------
    m_N : float
        HNL mass [GeV]
    Umu2 : float
        Muon mixing parameter squared (default 1 for ratios)
    Ue2 : float
        Electron mixing parameter squared (default 0)

    Returns
    -------
    dict : channel_name -> branching_ratio (sum = 1)
    """
    U2 = Umu2 + Ue2
    widths = {}

    # --- CC channels from U_mu mixing ---
    # N -> mu^- + e^+ + nu_e  (vertex: U_mu, W* -> e+ nu_e)
    widths[CHANNEL_MU_E_NU] = _Gamma_N_CC_leptonic(m_N, M_MUON, M_ELECTRON, Umu2)

    # N -> mu^- + pi^+  (vertex: U_mu, W* -> pi+)
    widths[CHANNEL_MU_PI] = _Gamma_N_l_pi(m_N, M_MUON, Umu2)

    # --- CC channels from U_e mixing ---
    # N -> e^- + mu^+ + nu_mu  (vertex: U_e, W* -> mu+ nu_mu)
    widths[CHANNEL_E_MU_NU] = _Gamma_N_CC_leptonic(m_N, M_ELECTRON, M_MUON, Ue2)

    # N -> e^- + pi^+  (vertex: U_e, W* -> pi+)
    widths[CHANNEL_E_PI] = _Gamma_N_l_pi(m_N, M_ELECTRON, Ue2)

    # --- NC channels (proportional to total U^2) ---
    # N -> nu + e^+ + e^-  (Z* -> e+e-)
    widths[CHANNEL_NU_EE] = _Gamma_N_nu_ll(m_N, M_ELECTRON, U2)

    # N -> nu + pi^0  (Z* -> pi0)
    widths[CHANNEL_NU_PI0] = _Gamma_N_nu_pi0(m_N, U2)

    total = sum(widths.values())
    if total <= 0:
        return {ch: 0.0 for ch in widths}

    return {ch: w / total for ch, w in widths.items()}


def _sample_2body_decay(m_parent, m1, m2, E_parent, p_parent_hat):
    """
    Sample 2-body decay kinematics: parent -> daughter1 + daughter2.

    Generates isotropic decay in rest frame, then boosts to lab frame.

    Parameters
    ----------
    m_parent : float
        Parent mass [GeV]
    m1, m2 : float
        Daughter masses [GeV]
    E_parent : float
        Parent energy in lab frame [GeV]
    p_parent_hat : array, shape (3,)
        Parent direction in lab frame (unit vector)

    Returns
    -------
    (E1, p1_hat), (E2, p2_hat) : tuples
        Energy and direction unit vector for each daughter in lab frame
    """
    if m1 + m2 >= m_parent:
        return None

    # Rest frame momentum magnitude
    p_star = np.sqrt(_kallen_lambda(m_parent**2, m1**2, m2**2)) / (2 * m_parent)

    # Isotropic direction in rest frame
    cos_theta = np.random.uniform(-1, 1)
    sin_theta = np.sqrt(1 - cos_theta**2)
    phi = np.random.uniform(0, 2 * np.pi)
    p_hat_rest = np.array([sin_theta * np.cos(phi),
                           sin_theta * np.sin(phi),
                           cos_theta])

    # Rest frame 4-momenta (daughter 1 along p_hat_rest, daughter 2 opposite)
    E1_rest = np.sqrt(p_star**2 + m1**2)
    E2_rest = np.sqrt(p_star**2 + m2**2)
    p1_rest = p_star * p_hat_rest
    p2_rest = -p_star * p_hat_rest

    # Boost to lab frame
    gamma = E_parent / m_parent
    beta = np.sqrt(1 - 1 / gamma**2) if gamma > 1 else 0.0
    boost_dir = np.asarray(p_parent_hat, dtype=float)

    def _boost(E_rest, p_rest):
        p_parallel = np.dot(p_rest, boost_dir)
        p_perp = p_rest - p_parallel * boost_dir
        E_lab = gamma * (E_rest + beta * p_parallel)
        p_par_lab = gamma * (p_parallel + beta * E_rest)
        p_lab = p_par_lab * boost_dir + p_perp
        p_mag = np.linalg.norm(p_lab)
        if p_mag > 0:
            p_hat_lab = p_lab / p_mag
        else:
            p_hat_lab = boost_dir.copy()
        return E_lab, p_hat_lab

    return _boost(E1_rest, p1_rest), _boost(E2_rest, p2_rest)


def _sample_3body_CC_leptonic(m_parent, m_vertex, m_W_lepton, E_parent, p_parent_hat):
    """
    Sample 3-body CC leptonic decay: N -> l_vertex^- + l_W^+ + nu.

    The HNL decays via CC: N -> l_vertex + W*, then W* -> l_W + nu.
    Both l_vertex and l_W are charged and returned.
    Charge conservation: N(0) -> l^-(−1) + l'^+(+1) + nu(0).

    Uses flat phase-space sampling with accept-reject on the Dalitz plot.
    Particle 1 = l_vertex (e.g. mu^-), Particle 2 = l_W (e.g. e^+),
    Particle 3 = nu (massless).

    Parameters
    ----------
    m_parent : float
        HNL mass [GeV]
    m_vertex : float
        Mass of the charged lepton at the HNL-W vertex [GeV]
    m_W_lepton : float
        Mass of the charged lepton from W* decay [GeV]
    E_parent : float
        HNL energy in lab frame [GeV]
    p_parent_hat : array, shape (3,)
        HNL direction in lab frame (unit vector)

    Returns
    -------
    list of (particle_label, E_lab, p_hat_lab) tuples for charged daughters
        particle_label is 'vertex_lepton' or 'W_lepton'
    """
    M = m_parent
    m1 = m_vertex    # vertex lepton (e.g. mu^-)
    m2 = m_W_lepton  # W* lepton (e.g. e^+)
    # m3 = 0 (neutrino)

    if m1 + m2 >= M:
        return []

    gamma = E_parent / M
    beta = np.sqrt(1 - 1 / gamma**2) if gamma > 1 else 0.0
    boost_dir = np.asarray(p_parent_hat, dtype=float)

    def _boost(E_rest, p_rest):
        p_parallel = np.dot(p_rest, boost_dir)
        p_perp = p_rest - p_parallel * boost_dir
        E_lab = gamma * (E_rest + beta * p_parallel)
        p_par_lab = gamma * (p_parallel + beta * E_rest)
        p_lab = p_par_lab * boost_dir + p_perp
        p_mag = np.linalg.norm(p_lab)
        if p_mag > 0:
            p_hat_lab = p_lab / p_mag
        else:
            p_hat_lab = boost_dir.copy()
        return E_lab, p_hat_lab

    # Dalitz plot sampling for 3-body decay M -> 1 + 2 + 3
    # Invariant mass variables: s12 = (p1+p2)^2, s23 = (p2+p3)^2
    # Ranges: s12 in [(m1+m2)^2, M^2], s23 in [m2^2, (M-m1)^2]
    s12_min = (m1 + m2)**2
    s12_max = M**2

    max_attempts = 200
    for _ in range(max_attempts):
        s12 = np.random.uniform(s12_min, s12_max)
        sqrt_s12 = np.sqrt(s12)

        # In the s12 rest frame:
        E2_star = (s12 - m1**2 + m2**2) / (2 * sqrt_s12)
        E3_star = (M**2 - s12) / (2 * sqrt_s12)

        if E2_star < m2 or E3_star < 0:
            continue

        p2_star = np.sqrt(max(0, E2_star**2 - m2**2))
        p3_star = E3_star  # massless neutrino

        s23_max = (E2_star + E3_star)**2 - (p2_star - p3_star)**2
        s23_min = (E2_star + E3_star)**2 - (p2_star + p3_star)**2
        s23_min = max(s23_min, m2**2)

        if s23_min >= s23_max:
            continue

        s23 = np.random.uniform(s23_min, s23_max)

        # This point is inside the Dalitz boundary by construction
        break
    else:
        # Fallback: give equal energy split, both along boost direction
        E_each = M / 3
        results = []
        for ml, label in [(m1, 'vertex_lepton'), (m2, 'W_lepton')]:
            p_mag = np.sqrt(max(0, E_each**2 - ml**2))
            cos_th = np.random.uniform(-1, 1)
            sin_th = np.sqrt(1 - cos_th**2)
            phi = np.random.uniform(0, 2 * np.pi)
            p_rest = p_mag * np.array([sin_th*np.cos(phi), sin_th*np.sin(phi), cos_th])
            E_lab, p_hat_lab = _boost(E_each, p_rest)
            results.append((label, E_lab, p_hat_lab))
        return results

    # Compute rest-frame energies from Dalitz variables
    # E1 = (M^2 + m1^2 - s23) / (2M)
    # E2 = (M^2 + m2^2 - s13) / (2M), where s13 = M^2 + m1^2 + m2^2 - s12 - s23
    E1 = (M**2 + m1**2 - s23) / (2 * M)
    s13 = M**2 + m1**2 + m2**2 - s12 - s23
    E2 = (M**2 + m2**2 - s13) / (2 * M)
    E3 = M - E1 - E2

    p1_mag = np.sqrt(max(0, E1**2 - m1**2))
    p2_mag = np.sqrt(max(0, E2**2 - m2**2))

    # Generate particle 1 direction isotropically in rest frame
    cos_th1 = np.random.uniform(-1, 1)
    sin_th1 = np.sqrt(1 - cos_th1**2)
    phi1 = np.random.uniform(0, 2 * np.pi)
    p1_vec = p1_mag * np.array([sin_th1*np.cos(phi1), sin_th1*np.sin(phi1), cos_th1])

    # Particle 2 direction: constrained by momentum conservation with p3
    # p2 + p3 = -p1 (rest frame), so p2 lies on a cone around -p1
    # The opening angle is fixed by energy-momentum conservation
    # cos(theta_12) = (2*E1*E2 - s12 + m1^2 + m2^2) / (2*p1_mag*p2_mag)
    if p1_mag > 0 and p2_mag > 0:
        cos_12 = (2*E1*E2 - s12 + m1**2 + m2**2) / (2*p1_mag*p2_mag)
        cos_12 = np.clip(cos_12, -1, 1)
    else:
        cos_12 = 1.0

    # Rotate p2 to have angle cos_12 relative to p1, with random azimuth
    sin_12 = np.sqrt(1 - cos_12**2)
    phi_12 = np.random.uniform(0, 2 * np.pi)

    # Build orthonormal basis around p1 direction
    p1_hat_rest = p1_vec / p1_mag if p1_mag > 0 else np.array([0, 0, 1.0])
    if abs(p1_hat_rest[0]) < 0.9:
        v = np.array([1., 0., 0.])
    else:
        v = np.array([0., 1., 0.])
    e1 = v - np.dot(v, p1_hat_rest) * p1_hat_rest
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(p1_hat_rest, e1)

    p2_vec = p2_mag * (cos_12 * p1_hat_rest +
                       sin_12 * (np.cos(phi_12) * e1 + np.sin(phi_12) * e2))

    # Boost both to lab
    E1_lab, p1_hat_lab = _boost(E1, p1_vec)
    E2_lab, p2_hat_lab = _boost(E2, p2_vec)

    return [('vertex_lepton', E1_lab, p1_hat_lab),
            ('W_lepton', E2_lab, p2_hat_lab)]


def _sample_3body_NC_ll(m_parent, m_l, E_parent, p_parent_hat):
    """
    Sample NC 3-body decay: N -> nu + l^+ + l^-.

    The HNL decays via NC: N -> nu + Z*, then Z* -> l^+ l^-.
    Both the l^+ and l^- are charged and returned.

    Parameters
    ----------
    m_parent : float
        HNL mass [GeV]
    m_l : float
        Mass of the charged lepton [GeV]
    E_parent : float
        HNL energy in lab frame [GeV]
    p_parent_hat : array, shape (3,)
        HNL direction in lab frame (unit vector)

    Returns
    -------
    list of ('electron', E_lab, p_hat_lab) tuples for both e+ and e-
    """
    # This is structurally the same as the CC case with:
    # particle 1 = nu (massless), particle 2 = l^+, particle 3 = l^-
    # We want particles 2 and 3 (both charged).
    M = m_parent

    if 2 * m_l >= M:
        return []

    gamma = E_parent / M
    beta = np.sqrt(1 - 1 / gamma**2) if gamma > 1 else 0.0
    boost_dir = np.asarray(p_parent_hat, dtype=float)

    def _boost(E_rest, p_rest):
        p_parallel = np.dot(p_rest, boost_dir)
        p_perp = p_rest - p_parallel * boost_dir
        E_lab = gamma * (E_rest + beta * p_parallel)
        p_par_lab = gamma * (p_parallel + beta * E_rest)
        p_lab = p_par_lab * boost_dir + p_perp
        p_mag = np.linalg.norm(p_lab)
        if p_mag > 0:
            p_hat_lab = p_lab / p_mag
        else:
            p_hat_lab = boost_dir.copy()
        return E_lab, p_hat_lab

    # Sample the l+l- invariant mass (Z* virtuality)
    # s_ll ranges from (2*m_l)^2 to M^2
    s_ll_min = (2 * m_l)**2
    s_ll_max = M**2

    # Sample s_ll uniformly (flat phase space)
    s_ll = np.random.uniform(s_ll_min, s_ll_max)
    sqrt_s_ll = np.sqrt(s_ll)

    # In the N rest frame: 2-body decay N -> nu + (ll system)
    # The ll system has invariant mass sqrt(s_ll)
    p_ll_star = (M**2 - s_ll) / (2 * M)  # momentum of ll system (nu is massless)
    E_ll = (M**2 + s_ll) / (2 * M)
    E_nu = (M**2 - s_ll) / (2 * M)

    # Direction of ll system in N rest frame: isotropic
    cos_th = np.random.uniform(-1, 1)
    sin_th = np.sqrt(1 - cos_th**2)
    phi = np.random.uniform(0, 2 * np.pi)
    p_ll_hat = np.array([sin_th*np.cos(phi), sin_th*np.sin(phi), cos_th])
    p_ll_vec = p_ll_star * p_ll_hat

    # Now decay the ll system in its rest frame
    # l+ and l- are back-to-back with momentum p_l = sqrt(s_ll/4 - m_l^2)
    p_l = np.sqrt(max(0, s_ll / 4 - m_l**2))
    E_l_rest = sqrt_s_ll / 2

    cos_th_l = np.random.uniform(-1, 1)
    sin_th_l = np.sqrt(1 - cos_th_l**2)
    phi_l = np.random.uniform(0, 2 * np.pi)
    p_l_hat = np.array([sin_th_l*np.cos(phi_l), sin_th_l*np.sin(phi_l), cos_th_l])

    # l+ and l- in ll rest frame
    p_lp_rest = p_l * p_l_hat
    p_lm_rest = -p_l * p_l_hat

    # Boost from ll rest frame to N rest frame
    gamma_ll = E_ll / sqrt_s_ll
    beta_ll = p_ll_star / E_ll if E_ll > 0 else 0.0

    def _boost_ll(E_r, p_r):
        p_par = np.dot(p_r, p_ll_hat)
        p_perp_v = p_r - p_par * p_ll_hat
        E_new = gamma_ll * (E_r + beta_ll * p_par)
        p_par_new = gamma_ll * (p_par + beta_ll * E_r)
        return E_new, p_par_new * p_ll_hat + p_perp_v

    E_lp_Nrest, p_lp_Nrest = _boost_ll(E_l_rest, p_lp_rest)
    E_lm_Nrest, p_lm_Nrest = _boost_ll(E_l_rest, p_lm_rest)

    # Boost from N rest frame to lab
    E_lp_lab, p_lp_hat_lab = _boost(E_lp_Nrest, p_lp_Nrest)
    E_lm_lab, p_lm_hat_lab = _boost(E_lm_Nrest, p_lm_Nrest)

    return [('electron', E_lp_lab, p_lp_hat_lab),
            ('electron', E_lm_lab, p_lm_hat_lab)]


def sample_hnl_decay(m_N, E_N, p_N_hat, Umu2=1.0, Ue2=0.0):
    """
    Sample one HNL decay event, returning all charged daughters.

    Picks a decay channel weighted by branching ratio, generates daughter
    4-momenta in the rest frame, and boosts to the lab frame.

    Parameters
    ----------
    m_N : float
        HNL mass [GeV]
    E_N : float
        HNL energy in lab frame [GeV]
    p_N_hat : array, shape (3,)
        HNL direction in lab frame (unit vector)
    Umu2 : float
        Muon mixing parameter squared
    Ue2 : float
        Electron mixing parameter squared

    Returns
    -------
    daughters : list of (particle_type, energy, direction) tuples
        particle_type : str, one of 'muon', 'electron', 'pion_charged', 'pion_neutral'
        energy : float [GeV]
        direction : ndarray, shape (3,) — unit vector
    """
    br = HNL_branching_ratios(m_N, Umu2, Ue2)

    # Pick a decay channel
    channels = list(br.keys())
    probs = np.array([br[ch] for ch in channels])
    if probs.sum() <= 0:
        return []
    probs = probs / probs.sum()
    channel = np.random.choice(channels, p=probs)

    daughters = []

    if channel == CHANNEL_MU_E_NU:
        # N -> mu^- + e^+ + nu_e  (CC via U_mu)
        # Both mu^- and e^+ are charged daughters
        result = _sample_3body_CC_leptonic(m_N, M_MUON, M_ELECTRON, E_N, p_N_hat)
        for label, E, p_hat in result:
            if label == 'vertex_lepton':
                daughters.append(("muon", E, p_hat))
            else:  # W_lepton = e^+
                daughters.append(("electron", E, p_hat))

    elif channel == CHANNEL_E_MU_NU:
        # N -> e^- + mu^+ + nu_mu  (CC via U_e)
        # Both e^- and mu^+ are charged daughters
        result = _sample_3body_CC_leptonic(m_N, M_ELECTRON, M_MUON, E_N, p_N_hat)
        for label, E, p_hat in result:
            if label == 'vertex_lepton':
                daughters.append(("electron", E, p_hat))
            else:  # W_lepton = mu^+
                daughters.append(("muon", E, p_hat))

    elif channel == CHANNEL_MU_PI:
        # N -> mu^- + pi^+  (CC via U_mu, 2-body)
        result = _sample_2body_decay(m_N, M_MUON, M_PI_CHARGED, E_N, p_N_hat)
        if result is not None:
            (E_mu, mu_hat), (E_pi, pi_hat) = result
            daughters.append(("muon", E_mu, mu_hat))
            daughters.append(("pion_charged", E_pi, pi_hat))

    elif channel == CHANNEL_E_PI:
        # N -> e^- + pi^+  (CC via U_e, 2-body)
        result = _sample_2body_decay(m_N, M_ELECTRON, M_PI_CHARGED, E_N, p_N_hat)
        if result is not None:
            (E_e, e_hat), (E_pi, pi_hat) = result
            daughters.append(("electron", E_e, e_hat))
            daughters.append(("pion_charged", E_pi, pi_hat))

    elif channel == CHANNEL_NU_EE:
        # N -> nu + e^+ + e^-  (NC via Z*, 3-body)
        # Both e^+ and e^- are charged
        result = _sample_3body_NC_ll(m_N, M_ELECTRON, E_N, p_N_hat)
        for label, E, p_hat in result:
            daughters.append(("electron", E, p_hat))

    elif channel == CHANNEL_NU_PI0:
        # N -> nu + pi^0  (NC, 2-body, pi0 -> 2gamma -> EM showers)
        result = _sample_2body_decay(m_N, 0.0, M_PI_NEUTRAL, E_N, p_N_hat)
        if result is not None:
            (_, _), (E_pi0, pi0_hat) = result
            daughters.append(("pion_neutral", E_pi0, pi0_hat))

    return daughters