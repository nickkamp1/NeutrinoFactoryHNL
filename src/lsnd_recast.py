"""
Recast of the LSND HNL e+e- search (arXiv:2306.07315) for the dipole portal.

Following the same approximation that 2109.03831 used for T2K/PS191: keep the
HNL *production* mechanism the same as in the mixing analysis (pion DAR and
muon DAR, both proceeding via |U_alpha|^2), and replace only the in-detector
decay-to-e+e- factor with its dipole-portal analogue.

This is NOT a full recast: dipole-portal HNLs can also be produced directly
at LSND (e.g. through neutrino upscattering on the detector), and that
production is not included here. See arXiv:2412.15051 for the dipole
production machinery.

Reference numbers from 2306.07315:
  - N_pi * eps_det = N_mu * eps_det = 3e19  (effective DAR yield * acceptance)
  - L_baseline ~ 30 m,  L_det = 7.6 m fiducial
  - 90% CL upper limit:  N_ee^(mu) * eps^(mu) + N_ee^(pi) * eps^(pi) < 55

Equations 3.3, 3.4 (mixing case, before efficiency):
  mu-DAR :  N_ee^(mu) ~ A_mu * (|U|^2/1e-6)^2 * (mN/m_mu)^6 * (1 - mN/m_mu)^4
                       * (1 + 4 mN/m_mu + (mN/m_mu)^2)
            with A_mu = 1.5e5 (mu-mixing), 8.1e5 (e-mixing)

  pi-DAR :  N_ee^(pi) (mu-mix) ~ 8.7e4 * (|U|^2/1e-6)^2 * (mN/m_mu)^6
              * m_pi [(m_mu^2+mN^2) m_pi^2 - (m_mu^2+mN^2)^2 + 4 m_mu^2 mN^2]
              / [m_mu (m_pi^2 - m_mu^2)^2]
            pi-DAR (e-mix)  ~ 4.1e5 * (|U|^2/1e-6)^2 * (mN/m_mu)^7
              * m_N m_pi (m_pi^2 - mN^2) / (m_pi^2 - m_mu^2)^2
"""

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d

from src.constants import (
    m_e, m_mu, alpha, sin2_theta_W, G_F,
    hbar_in_GeV_s, speed_of_light,
)
from src.xs_and_decays import HNL_ee_decay_width, HNL_decay_width


M_PI = 0.13957039  # charged pion mass [GeV]
N_LIMIT_90CL = 55.0
N_DAR_EFF = 3e19   # N_pi * eps_det = N_mu * eps_det per 2306.07315
L_BASELINE = 30.0  # m, detector baseline downstream of beam stop
L_DET = 7.6        # m, fiducial length

LSND_efficiency_data = np.loadtxt("data/dipole_constraints/LSND_efficiency.txt", delimiter=",")
LSND_efficiency_interp = interp1d(LSND_efficiency_data[:,0], LSND_efficiency_data[:,1], bounds_error=False, fill_value=1.0)


# ------------------------------------------------------------------
# Mixing-scenario expected events (eqs 3.3 and 3.4 of 2306.07315)
# ------------------------------------------------------------------

def _kinematic_mu_DAR(m_N):
    """Common kinematic factor in eq 3.3."""
    if m_N >= m_mu:
        return 0.0
    x = m_N / m_mu
    return x**6 * (1 - x)**4 * (1 + 4*x + x**2)


def _kinematic_pi_DAR_mu_mix(m_N):
    """Kinematic factor for pi-DAR, mu-mixing case (eq 3.4 top)."""
    if m_N + m_mu >= M_PI:
        return 0.0
    x = m_N / m_mu
    s = m_mu**2 + m_N**2
    num = M_PI * (s * M_PI**2 - s**2 + 4 * m_mu**2 * m_N**2)
    den = m_mu * (M_PI**2 - m_mu**2)**2
    return x**6 * num / den


def _kinematic_pi_DAR_e_mix(m_N):
    """Kinematic factor for pi-DAR, e-mixing case (eq 3.4 bottom)."""
    if m_N + m_e >= M_PI:
        return 0.0
    x = m_N / m_mu
    num = m_N * M_PI * (M_PI**2 - m_N**2)
    den = (M_PI**2 - m_mu**2)**2
    return x**7 * num / den


def N_ee_mixing(m_N, U_e2=0.0, U_mu2=0.0, eff = True):
    """Total expected N_ee at LSND in the mixing scenario, eqs 3.3 + 3.4.

    Parameters
    ----------
    m_N : float
        HNL mass [GeV].
    U_e2, U_mu2 : float
        Mixing parameters squared.
    eff_mu, eff_pi : float
        Detection efficiencies from fig 1 of 2306.07315 evaluated at this m_N.
        Mixing-flavor-averaged (a refinement would split eps^(mu) by flavor).
    """
    u_mu = (U_mu2 / 1e-6)**2
    u_e  = (U_e2  / 1e-6)**2

    f_mu_DAR = _kinematic_mu_DAR(m_N)
    f_pi_mu  = _kinematic_pi_DAR_mu_mix(m_N)
    f_pi_e   = _kinematic_pi_DAR_e_mix(m_N)

    N_mu = (1.5e5 * u_mu + 8.1e5 * u_e) * f_mu_DAR
    N_pi = 8.7e4  * u_mu * f_pi_mu + 4.1e5 * u_e * f_pi_e

    if eff:
        eff_mu = LSND_efficiency_interp(m_N)
        eff_pi = LSND_efficiency_interp(m_N)
    else:
        eff_mu = 1.0
        eff_pi = 1.0

    return eff_mu * N_mu + eff_pi * N_pi


# ------------------------------------------------------------------
# HNL energy / boost from DAR sources
# ------------------------------------------------------------------

def _E_N_pi_DAR(m_N):
    """Monochromatic HNL energy from pi+ -> e+ N at rest."""
    return (M_PI**2 + m_N**2 - m_e**2) / (2 * M_PI)


def _E_N_mu_DAR_spectrum(m_N, n_grid=200):
    """
    Spectrum of HNL energies from mu+ -> e+ nu N at rest.

    Returns (E_grid, w_grid) such that sum(w_grid) == 1 and the distribution
    is dN/dE in the mu+ rest frame.  Uses the SM differential decay rate
    differential in E_N -- for ultra-relativistic limit it's the standard
    Michel spectrum reshaped for massive N.
    """
    E_min = m_N
    E_max = (m_mu**2 + m_N**2) / (2 * m_mu)
    if E_max <= E_min:
        return np.array([E_min]), np.array([1.0])

    E = np.linspace(E_min, E_max, n_grid)
    # dGamma/dE_N for mu -> e nu N via U^2 (massless e limit, m_e ~ 0).
    # We only need the *shape*, so prefactors don't matter.
    # Use the same form as src.xs_and_decays.muon_differential_decay_width
    # with one flavor channel; below we only need a normalized spectrum.
    p = np.sqrt(np.maximum(E**2 - m_N**2, 0.0))
    # mu-mixing piece (matches term in muon_differential_decay_width):
    shape = (3*E*(m_mu**2 + m_N**2) - 4*m_mu*E**2 - 2*m_mu*m_N**2) * p
    shape = np.where(shape > 0, shape, 0.0)
    w = shape / shape.sum()
    return E, w


def _beta_gamma(E_N, m_N):
    return np.sqrt(np.maximum(E_N**2 - m_N**2, 0.0)) / m_N


# ------------------------------------------------------------------
# In-detector decay probability x BR_ee, both scenarios
# ------------------------------------------------------------------

def _P_decay_ee(Gamma_ee, Gamma_tot, beta_gamma):
    """
    Probability that an HNL with given total/partial widths decays to e+e-
    inside the LSND fiducial volume.

    P = (Gamma_ee / Gamma_tot) * [exp(-L_b/d) - exp(-(L_b+L_det)/d)]
    where d = beta*gamma * c * hbar / Gamma_tot in metres.

    Vectorizes over beta_gamma.
    """
    bg = np.atleast_1d(beta_gamma)
    if Gamma_tot <= 0:
        return np.zeros_like(bg)
    d_decay = bg * hbar_in_GeV_s * speed_of_light / Gamma_tot
    # avoid /0
    d_decay = np.where(d_decay > 0, d_decay, np.inf)
    survive_to_det = np.exp(-L_BASELINE / d_decay)
    # use expm1 so that for d >> L_det we don't lose precision in (1 - exp(-x))
    decay_in_det   = -np.expm1(-L_DET / d_decay)
    return (Gamma_ee / Gamma_tot) * survive_to_det * decay_in_det


def _avg_P_decay_DAR(m_N, Gamma_ee, Gamma_tot, source="pi"):
    """Average P(decay->ee in det) over the DAR-source HNL kinematics."""
    if source == "pi":
        E_N = _E_N_pi_DAR(m_N)
        return float(_P_decay_ee(Gamma_ee, Gamma_tot, _beta_gamma(E_N, m_N))[0])
    elif source == "mu":
        E_grid, w = _E_N_mu_DAR_spectrum(m_N)
        bg = _beta_gamma(E_grid, m_N)
        return float(np.sum(w * _P_decay_ee(Gamma_ee, Gamma_tot, bg)))
    else:
        raise ValueError(source)


# ------------------------------------------------------------------
# Dipole-portal recast
# ------------------------------------------------------------------

def N_ee_dipole_recast(m_N, mu_tr, U_e2, U_mu2, eff = True):
    """
    Expected N_ee at LSND in the dipole-portal recast.

    We hold the production rate (via mixing) fixed and replace the in-detector
    decay probability with the dipole-portal value.

        N_ee^dipole = N_HNL_produced(U^2) * <P(decay->ee in det)>^dipole(mu_tr, m_N)

    where N_HNL_produced is obtained from the published mixing prediction
    divided by the mixing decay probability evaluated at the same U^2.

    Parameters
    ----------
    m_N : float
        HNL mass [GeV].
    mu_tr : float
        Transition magnetic moment [GeV^-1].
    U_e2, U_mu2 : float
        Mixing parameters squared used for production.
    eff_mu, eff_pi : float
        Detection efficiencies from fig 1 at this m_N.
    """
    U2 = U_e2 + U_mu2
    if U2 <= 0:
        return 0.0  # no production via mixing

    # ---- mixing scenario decay rates (used to back out N_produced) ----
    G_ee_mix  = HNL_ee_decay_width(m_N, U_mu2, U_e2, d=0.0)
    G_tot_mix = HNL_decay_width(m_N, U2, d=0.0)
    if G_ee_mix <= 0 or G_tot_mix <= 0:
        return 0.0

    P_mu_mix = _avg_P_decay_DAR(m_N, G_ee_mix, G_tot_mix, source="mu")
    P_pi_mix = _avg_P_decay_DAR(m_N, G_ee_mix, G_tot_mix, source="pi")
    if P_mu_mix <= 0 or P_pi_mix <= 0:
        return 0.0

    # ---- split eq 3.3/3.4 by source to get N_produced per source ----
    u_mu = (U_mu2 / 1e-6)**2
    u_e  = (U_e2  / 1e-6)**2
    N_ee_mu_mix = (1.5e5 * u_mu + 8.1e5 * u_e) * _kinematic_mu_DAR(m_N)
    N_ee_pi_mix = (8.7e4 * u_mu * _kinematic_pi_DAR_mu_mix(m_N)
                   + 4.1e5 * u_e * _kinematic_pi_DAR_e_mix(m_N))

    N_prod_mu = N_ee_mu_mix / P_mu_mix
    N_prod_pi = N_ee_pi_mix / P_pi_mix

    # ---- dipole-portal decay rates ----
    G_ee_dip  = HNL_ee_decay_width(m_N, 0.0, 0.0, d=mu_tr)
    G_tot_dip = HNL_decay_width(m_N, 0.0, d=mu_tr)
    if G_tot_dip <= 0:
        return 0.0

    P_mu_dip = _avg_P_decay_DAR(m_N, G_ee_dip, G_tot_dip, source="mu")
    P_pi_dip = _avg_P_decay_DAR(m_N, G_ee_dip, G_tot_dip, source="pi")

    if eff:
        eff_mu = LSND_efficiency_interp(m_N)
        eff_pi = LSND_efficiency_interp(m_N)
    else:
        eff_mu = 1.0
        eff_pi = 1.0

    return eff_mu * N_prod_mu * P_mu_dip + eff_pi * N_prod_pi * P_pi_dip


# ------------------------------------------------------------------
# Convenience: exclusion checks and contour generation
# ------------------------------------------------------------------

def is_excluded(m_N, mu_tr, U_e2, U_mu2, eff_mu=1.0, eff_pi=1.0,
                N_limit=N_LIMIT_90CL):
    """Return True if (m_N, mu_tr) gives more than N_limit predicted events."""
    return N_ee_dipole_recast(m_N, mu_tr, U_e2, U_mu2, eff_mu, eff_pi) > N_limit


def _U2_at_mixing_boundary(m_N, flavor="mu", eff_mu=1.0, eff_pi=1.0,
                           N_limit=N_LIMIT_90CL):
    """
    Solve N_ee_mixing(U^2) = N_limit for U_alpha^2 at fixed m_N.
    Returns the LSND mixing-bound value of U_alpha^2 at this mass.
    """
    if flavor == "mu":
        kwargs0 = dict(U_e2=0.0, U_mu2=1e-6, eff_mu=eff_mu, eff_pi=eff_pi)
        N_at_1em6 = N_ee_mixing(m_N, **kwargs0)
        kw_key = "U_mu2"
    elif flavor == "e":
        kwargs0 = dict(U_e2=1e-6, U_mu2=0.0, eff_mu=eff_mu, eff_pi=eff_pi)
        N_at_1em6 = N_ee_mixing(m_N, **kwargs0)
        kw_key = "U_e2"
    else:
        raise ValueError(flavor)
    if N_at_1em6 <= 0:
        return np.nan
    # eqs 3.3/3.4 scale as (U^2)^2, so:
    return 1e-6 * np.sqrt(N_limit / N_at_1em6)


def mu_tr_exclusion_curve(m_N_grid, U_e2_fn=None, U_mu2_fn=None,
                          eff_mu_fn=None, eff_pi_fn=None,
                          N_limit=N_LIMIT_90CL,
                          mu_grid=np.logspace(-9, -3, 200),
                          flavor="mu"):
    """
    For each m_N, find the smallest mu_tr that yields > N_limit events.

    Parameters
    ----------
    m_N_grid : array
        HNL masses to scan [GeV].
    U_e2_fn, U_mu2_fn : callable(m_N) -> float, optional
        Functions returning U^2 vs m_N to use for production.  If both None,
        defaults to the LSND mixing-bound value of U_{flavor}^2 at each mass
        (the most aggressive recast).
    eff_mu_fn, eff_pi_fn : callable(m_N) -> float, optional
        Efficiency curves from fig 1 of 2306.07315.  Default to 1.0.
    flavor : {"mu", "e"}
        Which mixing flavor to default to (only used if U_*2_fn are None).

    Returns
    -------
    mu_excl : array, same length as m_N_grid
        Lower bound on mu_tr (smallest mu that yields > N_limit).  nan
        where no point in mu_grid exceeds N_limit.
    """
    if eff_mu_fn is None: eff_mu_fn = lambda m: 1.0
    if eff_pi_fn is None: eff_pi_fn = lambda m: 1.0

    mu_excl = np.full_like(m_N_grid, np.nan, dtype=float)
    for i, m_N in enumerate(m_N_grid):
        em = eff_mu_fn(m_N); ep = eff_pi_fn(m_N)
        if U_e2_fn is None and U_mu2_fn is None:
            U_bound = _U2_at_mixing_boundary(m_N, flavor=flavor,
                                             eff_mu=em, eff_pi=ep,
                                             N_limit=N_limit)
            if not np.isfinite(U_bound):
                continue
            U_e2  = U_bound if flavor == "e"  else 0.0
            U_mu2 = U_bound if flavor == "mu" else 0.0
        else:
            U_e2  = U_e2_fn(m_N)  if U_e2_fn  is not None else 0.0
            U_mu2 = U_mu2_fn(m_N) if U_mu2_fn is not None else 0.0

        N_vals = np.array([N_ee_dipole_recast(m_N, mu, U_e2, U_mu2, em, ep)
                           for mu in mu_grid])
        crossing = np.where(N_vals > N_limit)[0]
        if len(crossing):
            mu_excl[i] = mu_grid[crossing[0]]
    return mu_excl


if __name__ == "__main__":
    # Quick sanity check: scan m_N and print N_ee at mu_tr = 1e-6
    print(f"{'m_N [MeV]':>10} {'U_mu^2 LSND':>14} {'N_dip(mu=1e-6)':>16}")
    for m_N_MeV in [5, 10, 20, 50, 80, 100, 130]:
        m_N = m_N_MeV * 1e-3
        Ubnd = _U2_at_mixing_boundary(m_N, flavor="mu")
        if not np.isfinite(Ubnd):
            print(f"{m_N_MeV:>10} {'--':>14} {'--':>16}")
            continue
        N = N_ee_dipole_recast(m_N, 1e-6, U_e2=0.0, U_mu2=Ubnd)
        print(f"{m_N_MeV:>10} {Ubnd:>14.3e} {N:>16.3e}")
