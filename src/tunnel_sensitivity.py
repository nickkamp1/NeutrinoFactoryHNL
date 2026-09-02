"""Background-free HNL sensitivity for a short VACUUM DECAY TUNNEL at the end of
the muon beamline.

Instead of imaging HNL decays with an atmospheric Cherenkov camera (see
hnl_sensitivity.py / balloon.py), we place a ~100 m evacuated pipe just past the
beam-dump surface exit and simply COUNT HNL decays inside it.  If the residual
gas pressure is low enough that the expected number of beam-neutrino
interactions in the tunnel is < 1, any HNL decay in the pipe is background-free.

Two observables per (m_N, U2), both pure counting -- no Cherenkov ray tracing:

  * inclusive HNL decay rate in the tunnel  ->  N_incl  (all channels)
  * dimuon HNL decay rate in the tunnel     ->  N_dimuon = N_incl * BR_mumu,
    with BR(N4 -> nu_mu mu- mu+) taken from SIREN exactly as the atmospheric
    analysis does (balloon_siren.SIRENDimuonGeometry.dimuon_branching_ratio), so
    the two scenarios quote the same branching ratio.

Production is IDENTICAL to the balloon HNL signal (muon SCATTERING in the dump,
mu N -> N X), reusing HNLFluxGeometry.  The only new geometry is the exact
ray/cylinder path of each MC-sampled HNL through the finite tunnel, which folds
the transverse acceptance (wide-angle HNLs leaving through the side wall) into a
single analytic per-event decay probability.

The residual-gas pressure requirement (required_tunnel_pressure) is HNL-parameter
independent -- a single number -- and sums BOTH beam-neutrino sources that make
the charm background: muon scattering and muon decay.

Units note: src.background.sigma_CC_nu is [cm^2 / nucleon]; column depth is
[nucleons / cm^2].  All tunnel path lengths are converted m -> cm accordingly.
"""
import numpy as np

from src.constants import N_muon_decays
from src.background import sigma_CC_nu
from src.muon_beam_dump_helpers import muon_energy_in_earth
from src.xs_and_decays import HNL_decay_length

# nominal beam-dump config (matches hnl_sensitivity.py: E_MU/DUMP_DEPTH/DUMP_ANGLE)
E_MU, DUMP_DEPTH, DUMP_ANGLE = 5000.0, 100.0, 1.53

# tunnel geometry: a cylinder on the beam axis, z measured from the surface exit
# (z=0), extending [TUNNEL_START, TUNNEL_START + TUNNEL_LENGTH] with radius
# TUNNEL_RADIUS.  Production points sit at z in [-L_target, 0] (upstream).
TUNNEL_START = -1400.0       # m from the surface exit to the tunnel mouth
TUNNEL_LENGTH = 100.0    # m
TUNNEL_RADIUS = 10.0      # m

# ideal-gas / composition constants for the pressure conversion
K_B = 1.380649e-23               # J / K
NUCLEONS_PER_MOLECULE = 29.0     # ~air (N2/O2); use 2 for H2, etc.


def default_geom():
    """Nominal pure-numpy HNLFluxGeometry (no SIREN), matching the paper config."""
    from src.balloon import HNLFluxGeometry
    return HNLFluxGeometry(E_mu=E_MU, dump_depth=DUMP_DEPTH, dump_angle=DUMP_ANGLE)


# --------------------------------------------------------------------------- #
# Shared geometry: ray x finite cylinder
# --------------------------------------------------------------------------- #
def tunnel_path_bounds(prod_pts, dirs, z0, z1, radius):
    """Path-length interval [t_in, t_out] (m, measured from the production point
    along each unit direction) over which the ray is INSIDE the tunnel cylinder:
    axial z in [z0, z1], transverse r < radius, cylinder axis = beam (z).

    Assumes production ON-AXIS (x = y = 0) and upstream (z_prod <= z0), exactly as
    produced by HNLFluxGeometry.sample_production_points_weighted.  Because the
    ray starts on axis, r(t) = t * sin(theta) grows monotonically, so the ray is
    within the radius for t < radius/sin(theta) and within the axial window for
    t in [(z0-z_prod)/uz, (z1-z_prod)/uz].

    Returns (t_in, t_out, valid); t_in = t_out = 0 where the ray misses the pipe.
    Shared by HNL decays (decay probability) and nu interactions (chord length)."""
    dirs = np.asarray(dirs, float)
    uz = dirs[:, 2]
    st = np.sqrt(dirs[:, 0] ** 2 + dirs[:, 1] ** 2)      # sin(theta) to beam axis
    zp = np.asarray(prod_pts, float)[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        t_front = (z0 - zp) / uz                          # reach front cap z0
        t_back = (z1 - zp) / uz                           # reach back cap  z1
        t_side = np.where(st > 0, radius / st, np.inf)    # reach side wall r=R
    t_in = t_front
    t_out = np.minimum(t_back, t_side)                    # leave via back cap OR wall
    valid = (uz > 0) & (t_front >= 0) & (t_out > t_in)
    return np.where(valid, t_in, 0.0), np.where(valid, t_out, 0.0), valid


# --------------------------------------------------------------------------- #
# HNL decay counting
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Branching ratios from SIREN (mirrors balloon_siren.SIRENDimuonGeometry)
# --------------------------------------------------------------------------- #
# The atmospheric analysis gets BR(N4 -> nu_mu mu- mu+) from SIREN's HNLDecay
# partial widths (balloon_siren.SIRENDimuonGeometry._get_decay /
# .dimuon_branching_ratio), NOT from xs_and_decays.HNL_branching_ratios (whose
# channel list is incomplete -- no N -> nu mu+ mu-, no multi-hadron states).  We
# mirror the SIREN calculation here so the tunnel and the balloon quote the SAME
# branching ratios.  SIREN is imported lazily so this module still imports (and
# the inclusive-rate path still runs) in a plain-numpy environment without it.
_DECAY_CACHE = {}   # (m_N, nature) -> (decay, {signature_key: width}, w_tot)


def _siren_decay(m_N, nature="Majorana"):
    """Build (and cache) SIREN's HNLDecay for m_N with muon-dominated mixing.

    Mixing is [Ue4, Umu4, Utau4] = [0, 1, 0]: branching ratios are independent of
    the overall mixing scale for a single-flavour coupling, so a unit Umu4 is
    enough here -- the absolute lifetime is handled by HNL_decay_length(m_N, U2, E).
    Returns (decay, widths_by_signature, total_width)."""
    key = (float(m_N), nature)
    if key in _DECAY_CACHE:
        return _DECAY_CACHE[key]
    from siren import interactions, dataclasses          # lazy: needs the lienv env
    chiral = (interactions.HNLDecay.ChiralNature.Majorana if nature == "Majorana"
              else interactions.HNLDecay.ChiralNature.Dirac)
    decay = interactions.HNLDecay(float(m_N), [0.0, 1.0, 0.0], chiral)
    widths, w_tot = {}, 0.0
    for s in decay.GetPossibleSignatures():
        rec = dataclasses.InteractionRecord()
        rec.signature = s
        w = decay.TotalDecayWidth(rec)
        widths[tuple(str(t) for t in s.secondary_types)] = w
        if w_tot == 0.0:
            # same for every signature; query once
            w_tot = decay.TotalDecayWidthAllFinalStates(rec)
    _DECAY_CACHE[key] = (decay, widths, w_tot)
    return _DECAY_CACHE[key]


def siren_branching_ratios(m_N, nature="Majorana"):
    """{final-state tuple -> branching ratio} from SIREN partial widths."""
    _, widths, w_tot = _siren_decay(m_N, nature)
    if w_tot <= 0:
        return {k: 0.0 for k in widths}
    return {k: w / w_tot for k, w in widths.items()}


def dimuon_BR(m_N, nature="Majorana"):
    """BR(N4 -> nu_mu mu- mu+) from SIREN -- the SAME quantity the atmospheric
    analysis uses (balloon_siren.SIRENDimuonGeometry.dimuon_branching_ratio), so
    tunnel and balloon dimuon rates are directly comparable.  Zero below 2*m_mu."""
    from siren import dataclasses
    PT = dataclasses.Particle.ParticleType
    decay, _, w_tot = _siren_decay(m_N, nature)
    for s in decay.GetPossibleSignatures():
        st = list(s.secondary_types)
        if (s.primary_type == PT.N4 and len(st) == 3
                and st[1] == PT.MuMinus and st[2] == PT.MuPlus):
            rec = dataclasses.InteractionRecord()
            rec.signature = s
            return (decay.TotalDecayWidth(rec) / w_tot) if w_tot > 0 else 0.0
    return 0.0   # channel closed (m_N < 2 m_mu)


def tunnel_decay_counts(geom, m_N, U2, N_samples=20000,
                        z0=TUNNEL_START, z1=None, radius=TUNNEL_RADIUS,
                        nature="Majorana", with_dimuon=True):
    """(N_inclusive, N_dimuon): expected HNL decays inside the tunnel for
    N_muon_decays muons.  HNL production via muon scattering (mu N -> N X),
    identical to the balloon signal; transverse acceptance folded in exactly via
    tunnel_path_bounds.

    Per event the decay-in-tunnel probability is
        P = exp(-t_in / L_dec) - exp(-t_out / L_dec)
    (survival to the pipe mouth times the probability of decaying before exiting
    it), with L_dec the TOTAL-width decay length.  Assumes U2 is muon mixing.

    N_inclusive counts ALL decay channels -- the right observable for a
    background-free volume.  N_dimuon applies the SIREN BR(N4 -> nu mu- mu+),
    matching the atmospheric analysis; set with_dimuon=False to skip the SIREN
    call (and its import) when only the inclusive rate is needed."""
    if z1 is None:
        z1 = z0 + TUNNEL_LENGTH
    prod_pts, N_HNL_per_mu, _, _ = \
        geom.sample_production_points_weighted(m_N, U2, N_samples)   # mode="scattering"
    E_mu_local = muon_energy_in_earth(geom.E_mu, prod_pts[:, -1] + geom.L_target)
    E_N, hnl_dir, _, _ = geom.sample_kinematics(prod_pts[:, -1], E_mu_local, m_N=m_N)
    L_dec = HNL_decay_length(m_N, U2, E_N)                           # m, per event
    t_in, t_out, valid = tunnel_path_bounds(prod_pts, hnl_dir, z0, z1, radius)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        P = np.where(valid & (L_dec > 0),
                     np.exp(-t_in / L_dec) - np.exp(-t_out / L_dec), 0.0)
    N_incl = N_muon_decays * N_HNL_per_mu * float(np.mean(P))
    N_dimu = N_incl * dimuon_BR(m_N, nature=nature) if with_dimuon else np.nan
    return N_incl, N_dimu


def tunnel_significance_grid(geom=None, masses=(5, 7, 10, 14, 20, 30, 40),
                             U2_grid=None, N_samples=20000, channel="inclusive",
                             nature="Majorana", **kw):
    """Background-free significance over (m_N, U2).  Since b = 0, the paper's
    s/sqrt(s+b) reduces to Z = sqrt(N); the Z=2 contour is N = 4 (use N = 2.3 for
    a 90% CL zero-background limit instead).

    ``channel``: "inclusive" (default -- all decay channels, the right observable
    for a background-free volume) or "dimuon" (SIREN BR(N4 -> nu mu- mu+), for a
    like-for-like comparison with the atmospheric dimuon analysis).

    Efficient: the per-event kinematics and tunnel geometry are U2-INDEPENDENT, so
    each mass is sampled ONCE and reweighted analytically across the whole U2 grid
    (N_HNL ~ U2, L_dec ~ 1/U2) -- the same reweighting idea as
    hnl_sensitivity.reweight_hnl, applied to the tunnel's decay-region acceptance.

    Returns dict(masses, U2, Ninc, Ndimuon, Z) with (n_mass, n_U2) arrays.  Feed
    the 'Z' array straight into hnl_sensitivity.SensitivityModel.significance_map
    / _contour_segments to draw the (closed) exclusion band; the upper edge,
    where HNLs decay BEFORE the tunnel, is captured by the exp(-t_in/L) term."""
    if geom is None:
        geom = default_geom()
    if U2_grid is None:
        U2_grid = np.logspace(-14, -8, 61)
    U2_grid = np.asarray(U2_grid, float)
    masses = list(masses)
    z0 = kw.pop("z0", TUNNEL_START)
    z1 = kw.pop("z1", None) or (z0 + TUNNEL_LENGTH)
    radius = kw.pop("radius", TUNNEL_RADIUS)
    U2_ref = 1e-10                     # arbitrary: kinematics are U2-independent

    Ninc = np.zeros((len(masses), len(U2_grid)))
    Ndim = np.zeros_like(Ninc)
    for i, m in enumerate(masses):
        prod, N_HNL_ref, _, _ = \
            geom.sample_production_points_weighted(m, U2_ref, N_samples)
        E_mu_local = muon_energy_in_earth(geom.E_mu, prod[:, -1] + geom.L_target)
        E_N, hnl_dir, _, _ = geom.sample_kinematics(prod[:, -1], E_mu_local, m_N=m)
        t_in, t_out, valid = tunnel_path_bounds(prod, hnl_dir, z0, z1, radius)
        br = dimuon_BR(m, nature=nature) if channel == "dimuon" else 1.0
        for j, U2 in enumerate(U2_grid):
            L_dec = HNL_decay_length(m, U2, E_N)
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                P = np.where(valid & (L_dec > 0),
                             np.exp(-t_in / L_dec) - np.exp(-t_out / L_dec), 0.0)
            Ninc[i, j] = (N_muon_decays * (N_HNL_ref * U2 / U2_ref)
                          * float(np.mean(P)))
            Ndim[i, j] = Ninc[i, j] * br
    N_sig = Ndim if channel == "dimuon" else Ninc
    return dict(masses=np.array(masses, float), U2=U2_grid,
                Ninc=Ninc, Ndimuon=Ndim, Z=np.sqrt(N_sig))


# --------------------------------------------------------------------------- #
# Vacuum pressure requirement (both nu sources: scattering + decay)
# --------------------------------------------------------------------------- #
def required_tunnel_pressure(geom=None, N_int_max=1.0, N_samples=50000,
                             z0=TUNNEL_START, z1=None, radius=TUNNEL_RADIUS,
                             nucleons_per_molecule=NUCLEONS_PER_MOLECULE, T=300.0,
                             modes=("scattering", "decay")):
    """Maximum residual-gas pressure [Pa] for fewer than N_int_max expected beam-
    neutrino CC interactions in the tunnel over the whole run (N_muon_decays).
    HNL-parameter INDEPENDENT -> a single number.

        N_int = N_mu * n_gas * sum_modes (N_nu/mu)_mode * <sigma_CC(E) * L_chord>
        n_gas_max [nucleons/cm^3] = N_int_max / (N_mu * sum_modes ...)
        P = (n_gas_max / nucleons_per_molecule) * 1e6 * k_B * T   [Pa]

    Uses the SAME per-mode flux setup (m_N=0, U2=1) as charm_background, and the
    SAME ray/cylinder geometry as the HNL decays for each nu's chord length.
    sigma_CC_nu is [cm^2]; chords are converted m -> cm.  Returns a dict with the
    pressure in Pa and mbar plus the per-mode breakdown for inspection."""
    if geom is None:
        geom = default_geom()
    if z1 is None:
        z1 = z0 + TUNNEL_LENGTH

    weighted = 0.0        # sum_modes  (N_nu/mu) * <sigma_CC[cm^2] * chord[cm]>
    per_mode = {}
    for mode in modes:
        prod_pts, N_nu_per_mu, _, _ = geom.sample_production_points_weighted(
            m_N=0.0, U2=1.0, N_samples=N_samples, mode=mode)
        E_mu_local = muon_energy_in_earth(geom.E_mu, prod_pts[:, -1] + geom.L_target)
        E_nu, nu_dir, _, _ = geom.sample_kinematics(prod_pts[:, -1], E_mu_local,
                                                    mode=mode)          # m_N=None
        t_in, t_out, valid = tunnel_path_bounds(prod_pts, nu_dir, z0, z1, radius)
        chord_cm = np.where(valid, (t_out - t_in) * 100.0, 0.0)         # m -> cm
        contrib = float(N_nu_per_mu * np.mean(sigma_CC_nu(E_nu) * chord_cm))
        per_mode[mode] = contrib
        weighted += contrib

    n_gas_max = N_int_max / (N_muon_decays * weighted)   # nucleons / cm^3
    n_mol_per_m3 = (n_gas_max / nucleons_per_molecule) * 1e6
    P_pa = n_mol_per_m3 * K_B * T
    return dict(pressure_Pa=P_pa, pressure_mbar=P_pa * 0.01,
                n_gas_per_cm3=n_gas_max, per_mode=per_mode, N_int_max=N_int_max)
