# Charm-production dimuon background for balloon/satellite HNL detection.
#
# Physics
# -------
# The beam produces neutrinos (mu + N -> nu_mu + X, i.e. the same DIS process
# as HNL production with m_N -> 0, U^2 = 1).  Those neutrinos travel upward and
# a tiny fraction undergo CHARM deep-inelastic scattering in the atmosphere:
#
#     nu_mu + N --(CC charm)-->  mu^-  +  Hadrons  +  D            (SIREN)
#                                                   D -> mu^+ nu X  (SIREN)
#
# The primary muon (mu^-) and the semileptonic D-decay muon (mu^+) form an
# OPPOSITE-SIGN DIMUON that mimics the HNL N4 -> nu mu mu signal, accompanied by
# a hadronic shower.  This is the irreducible charm dimuon background.
#
# Relationship to the rest of the codebase
# -----------------------------------------
# This module is the CHARM analogue of src/background.py: it reuses that
# module's beam-neutrino flux (HNLFluxGeometry with m_N=0, U^2=1), atmospheric
# column-depth, and interaction-altitude sampling verbatim, and reuses
# src/balloon_siren.py's pattern of calling SIREN per event to sample a final
# state.  The only physics that changes is:
#
#   * the interaction weight P_int carries the CHARM cross section rather than
#     the total CC cross section (implemented as a unit-free charm/total-CC
#     ratio times background.py's validated sigma_CC_nu, so the absolute
#     normalization is inherited and only the ~few-% charm fraction is new); and
#   * the final state is the SIREN charm chain (two muons + a hadronic shower)
#     rather than a single hand-sampled CC muon.
#
# The per-muon photon array ``mu_photon_counts`` (shape (2, N_det, N_samples))
# matches balloon_siren.compute_dimuon_signal_at_satellite exactly, so the
# output plugs into the same both-muon tagging / summarize machinery and into
# src/hnl_efficiency.py.
#
# Runtime: requires SIREN (pip-installed into the spack ``lienv`` on the FASRC
# cluster) plus the charm-target QuarkDIS splines.  It does NOT need Pythia8 or
# LHAPDF at runtime -- QuarkDISFromSpline is fully analytic once the splines are
# read.  Verified against SIREN 0.1.0 in lienv.
import os

import numpy as np

import siren
from siren import interactions, dataclasses, utilities

from src.constants import N_muon_decays
from src.cherenkov import (cherenkov_photons_multi_detector, cherenkov_transmission,
                          N_AIR)
from src.constants import R_det
from src.balloon import muon_range_in_air, hadronic_shower_cherenkov, muon_energy_in_earth
from src.background import (sigma_CC_nu, atmospheric_column_depth_nucleons,
                           sample_interaction_altitude)
# on-camera imaging (centroid) reused from the HNL imaging module, so the charm
# on-camera muon separation is computed identically to the signal (uniform_n=N_AIR)
from src.muon_image_spread import (image_muons, centroid_to_pixel,
                                    PIXEL_DEG_DEFAULT, IMAGE_MU_KEYS, IMAGE_PAIR_KEYS)

PT = dataclasses.Particle.ParticleType

# --------------------------------------------------------------------------- #
# Charm splines.  The QuarkDIS charm-target .fits splines are large and are not
# bundled with this repo; point SIREN_CHARM_SPLINE_DIR at a directory holding
#     {dsdxdy,sigma}_{nu,nubar}-N-{cc,nc}[-charm]-{PDF}.fits
# (both the charm and the inclusive files are needed -- the latter supplies the
# total-CC denominator for the charm fraction).  The default is the shared set
# used by the IceCube dimuon-charm analysis on FASRC.
# --------------------------------------------------------------------------- #
DEFAULT_SPLINE_DIR = ("/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/"
                      "pzhelnin/DiMuons/Simulation/Resources/Splines/M_Muon")

# Air is ~isoscalar (N2/O2); we use the oxygen (O16) charm splines with the
# O16-matched nuclear PDF as the atmospheric target and the hydrogen splines for
# free protons.  For an isoscalar treatment the O16 spline alone is adequate.
O16_PDF = "EPPS21nlo_CT18Anlo_O16_central"
H_PDF = "PDF4LHC21_mc_central"

ISOSCALAR_MASS = (0.938272 + 0.939565) / 2.0

# D species that a neutrino primary can produce (QuarkDISFromSpline emits the D
# directly as the third secondary).  Charge conjugates for an antineutrino.
_D_SPECIES = (PT.D0, PT.DPlus, PT.DsPlus)
_D_SPECIES_BAR = (PT.D0Bar, PT.DMinus, PT.DsMinus)
_MU_TYPES = (PT.MuMinus, PT.MuPlus)


def _is_antineutrino(nu_type):
    return nu_type in (PT.NuEBar, PT.NuMuBar, PT.NuTauBar)


# --------------------------------------------------------------------------- #
# Model: loads and caches the SIREN charm cross sections and D decays.
# --------------------------------------------------------------------------- #
class CharmDISModel:
    """SIREN charm DIS cross sections + D-meson decays for one neutrino type.

    Loads QuarkDISFromSpline for CC (and optionally NC) charm on an isoscalar
    (O16) target, the matching inclusive-CC spline (for the charm fraction),
    and a CharmMesonDecay per D species, caching everything.

    Parameters
    ----------
    nu_type : siren ParticleType
        Primary neutrino type (e.g. PT.NuMu).
    spline_dir : str or None
        Directory of the charm/inclusive .fits splines (default: env var
        SIREN_CHARM_SPLINE_DIR, else DEFAULT_SPLINE_DIR).
    include_nc : bool
        Also load NC charm (adds to the charm fraction).  Default True.
    target_pdf : str
        PDF tag in the spline filenames (default: the O16 nuclear set).
    seed : int
        Seed for SIREN's RNG used in sampling.
    """

    def __init__(self, nu_type=PT.NuMu, spline_dir=None, include_nc=True,
                 target_pdf=O16_PDF, seed=42):
        self.nu_type = nu_type
        self.include_nc = include_nc
        self.spline_dir = (spline_dir or os.environ.get("SIREN_CHARM_SPLINE_DIR")
                           or DEFAULT_SPLINE_DIR)
        self.target_pdf = target_pdf
        self._rng = utilities.SIREN_random(seed)
        self._prefix = "nubar" if _is_antineutrino(nu_type) else "nu"
        self._target = PT.O16Nucleus
        self._D_species = _D_SPECIES_BAR if _is_antineutrino(nu_type) else _D_SPECIES

        self._xs_charm = {}      # "cc"/"nc" -> QuarkDISFromSpline (charm)
        self._xs_incl = {}       # "cc"/"nc" -> QuarkDISFromSpline (inclusive)
        self._sig_charm = {}     # "cc"/"nc" -> list of signatures
        self._decays = {}        # D type -> (CharmMesonDecay, mu_signature, BR_mu)
        self._frac_grid = None   # (log E grid, total charm fraction) for interpolation
        self._frag = {}          # "cc"/"nc" -> (log E grid, (n_sig, n_E) D-species fractions)
        self._load()
        self._build_fraction_table()
        self._build_fragmentation_table()

    # -- spline loading -------------------------------------------------- #
    def _spline_paths(self, current, charm):
        tag = "charm-" if charm else ""
        base = f"{self._prefix}-N-{current}-{tag}{self.target_pdf}.fits"
        return (os.path.join(self.spline_dir, f"dsdxdy_{base}"),
                os.path.join(self.spline_dir, f"sigma_{base}"))

    def _make_xs(self, current, charm):
        diff, tot = self._spline_paths(current, charm)
        int_type = 1 if current == "cc" else 2
        return interactions.QuarkDISFromSpline(
            diff, tot, int(int_type), ISOSCALAR_MASS, 1,
            [self.nu_type], [self._target], "m")

    def _load(self):
        currents = ["cc"] + (["nc"] if self.include_nc else [])
        for cur in currents:
            self._xs_charm[cur] = self._make_xs(cur, charm=True)
            self._xs_incl[cur] = self._make_xs(cur, charm=False)
            self._sig_charm[cur] = list(
                self._xs_charm[cur].GetPossibleSignaturesFromParents(
                    self.nu_type, self._target))
        # D decays (build lazily-cached muon-channel signature + BR)
        for D in self._D_species:
            self._get_decay(D)

    # -- charm fraction (unit-free interaction-weight multiplier) -------- #
    def charm_fraction(self, E_nu, current="cc"):
        """sigma_charm / sigma_inclusive at energy E_nu [GeV] for this current.

        Unit-free (both from the same TotalCrossSection call), so it multiplies
        background.py's validated total-CC probability to give the charm rate
        without needing the absolute cross-section units.
        """
        E = float(E_nu)
        num = self._xs_charm[current].TotalCrossSection(self.nu_type, E)
        den = self._xs_incl[current].TotalCrossSection(self.nu_type, E)
        return (num / den) if den > 0 else 0.0

    def _total_charm_fraction_exact(self, E_nu):
        """Charm fraction summed over the loaded currents (CC [+ NC]), evaluated
        directly from the splines (slow; used to build the interpolation table)."""
        f = self.charm_fraction(E_nu, "cc")
        if self.include_nc:
            # NC charm relative to the *CC* total (background.py's P_CC baseline)
            f += self.charm_fraction(E_nu, "nc")
        return f

    def _build_fraction_table(self, E_min=5.0, E_max=1e5, n=80):
        """Precompute the total charm fraction on a log-E grid.  The fraction is
        smooth in log E, so per-event evaluation becomes a cheap interpolation
        instead of 2-4 SIREN TotalCrossSection calls (critical at N~1e5 events)."""
        logE = np.linspace(np.log10(E_min), np.log10(E_max), n)
        frac = np.array([self._total_charm_fraction_exact(10.0 ** le)
                         for le in logE])
        self._frac_grid = (logE, frac)

    def total_charm_fraction(self, E_nu):
        """Total charm fraction (CC [+ NC]) at E_nu [GeV], via interpolation of
        the precomputed table (clipped to the grid range)."""
        logE, frac = self._frac_grid
        return float(np.interp(np.log10(max(float(E_nu), 1e-3)), logE, frac))

    def _build_fragmentation_table(self, E_min=5.0, E_max=1e5, n=60):
        """Precompute the D-species fragmentation fractions vs energy for each
        current.  The per-signature TotalCrossSection encodes the fragmentation
        fraction (their sum is the inclusive-over-D charm total), so sampling the
        D-production signature must be weighted by these -- NOT uniform (D0 ~0.61,
        D+ ~0.24, Ds+ ~0.15).  Cached so per-event sampling is a cheap interp."""
        logE = np.linspace(np.log10(E_min), np.log10(E_max), n)
        for cur, sigs in self._sig_charm.items():
            frac = np.zeros((len(sigs), n))
            for j, le in enumerate(logE):
                E = 10.0 ** le
                xsecs = np.array([self._sig_xsec(self._xs_charm[cur], sg, E)
                                  for sg in sigs])
                tot = xsecs.sum()
                frac[:, j] = (xsecs / tot) if tot > 0 else 1.0 / len(sigs)
            self._frag[cur] = (logE, frac)

    @staticmethod
    def _sig_xsec(xs, signature, E):
        """Per-signature (per-D-species) total cross section at energy E [GeV]."""
        r = dataclasses.InteractionRecord()
        r.signature = signature
        r.primary_momentum = [float(E), 0.0, 0.0, float(E)]
        r.primary_mass = 0.0
        r.target_mass = ISOSCALAR_MASS
        try:
            return float(xs.TotalCrossSection(r))
        except Exception:
            return 0.0

    # -- D decay setup --------------------------------------------------- #
    def _get_decay(self, D_type):
        """(CharmMesonDecay, muon-channel signature, BR_mu) for a D species."""
        if D_type in self._decays:
            return self._decays[D_type]
        dec = interactions.CharmMesonDecay(primary_type=D_type)
        mu_sig, br_mu = None, 0.0
        sigs = list(dec.GetPossibleSignaturesFromParent(D_type))
        # build a representative record to evaluate partial/total widths
        rec = dataclasses.InteractionRecord()
        rec.signature = sigs[0]
        rec.primary_momentum = [2.0, 0.0, 0.0, np.sqrt(4.0 - 1.8697**2)]
        rec.primary_mass = 1.8697
        w_tot = dec.TotalDecayWidthAllFinalStates(rec)
        for ds in sigs:
            if any(t in _MU_TYPES for t in ds.secondary_types):
                r = dataclasses.InteractionRecord()
                r.signature = ds
                r.primary_momentum = rec.primary_momentum
                r.primary_mass = rec.primary_mass
                mu_sig = ds
                br_mu = (dec.TotalDecayWidth(r) / w_tot) if w_tot > 0 else 0.0
                break
        self._decays[D_type] = (dec, mu_sig, br_mu)
        return self._decays[D_type]

    def branching_to_muon(self, D_type):
        """Semileptonic BR(D -> mu nu X) for the given D species."""
        return self._get_decay(D_type)[2]

    # -- final-state sampling -------------------------------------------- #
    def sample_event(self, E_nu, nu_dir, current="cc"):
        """Sample one charm DIS event + D->mu decay for a neutrino.

        Parameters
        ----------
        E_nu : float
            Neutrino lab energy [GeV].
        nu_dir : (3,) array
            Neutrino direction (unit vector).
        current : {"cc","nc"}
            Which charm current to sample (default CC).

        Returns
        -------
        dict or None
            None if the event could not be sampled (or no D-muon channel).
            Otherwise:
              mu1_dir, mu1_E   : primary DIS muon (mu^-)   direction, energy
              mu2_dir, mu2_E   : D-decay muon (mu^+)        direction, energy
              had_E, had_dir   : hadronic-shower energy [GeV] and direction
              D_type           : the D species
              decay_length     : lab decay length of the D [m]
              weight           : BR(D -> mu nu X) for this species (the extra
                                 factor turning "a charm event" into "a charm
                                 DIMUON event"); the charm rate itself is carried
                                 by the interaction weight, not here.
        """
        nu_dir = np.asarray(nu_dir, float)
        p = float(E_nu) * nu_dir
        sigs = self._sig_charm[current]
        # pick the D-production signature weighted by the per-species cross
        # section (= physical fragmentation fraction), interpolated in energy.
        logE, frac = self._frag[current]
        w = np.array([np.interp(np.log10(max(float(E_nu), 1e-3)), logE, frac[k])
                      for k in range(len(sigs))])
        w = w / w.sum() if w.sum() > 0 else np.ones(len(sigs)) / len(sigs)
        sig = sigs[np.random.choice(len(sigs), p=w)]

        ir = dataclasses.InteractionRecord()
        ir.signature = sig
        ir.primary_momentum = [float(E_nu), float(p[0]), float(p[1]), float(p[2])]
        ir.primary_mass = 0.0
        ir.target_mass = ISOSCALAR_MASS
        cdr = dataclasses.CrossSectionDistributionRecord(ir)
        try:
            self._xs_charm[current].SampleFinalState(cdr, self._rng)
        except RuntimeError:
            return None
        out = dataclasses.InteractionRecord()
        out.signature = sig
        out.primary_momentum = ir.primary_momentum
        out.primary_mass = 0.0
        out.target_mass = ISOSCALAR_MASS
        cdr.finalize(out)

        secs = list(out.secondary_momenta)
        stypes = list(out.signature.secondary_types)
        smass = list(out.secondary_masses)
        # secondary order is [primary lepton, Hadrons, D] (verified)
        p_lep, p_had, p_D = secs[0], secs[1], secs[2]
        D_type = stypes[2]

        dec, mu_sig, br_mu = self._get_decay(D_type)
        if mu_sig is None or br_mu <= 0:
            return None

        # decay length (lab) of this D
        drec0 = dataclasses.InteractionRecord()
        drec0.signature = mu_sig
        drec0.primary_momentum = [float(p_D[0]), float(p_D[1]),
                                  float(p_D[2]), float(p_D[3])]
        drec0.primary_mass = float(smass[2])
        try:
            decay_length = float(dec.TotalDecayLength(drec0))
        except Exception:
            decay_length = 0.0

        # sample the D -> mu decay kinematics
        dcdr = dataclasses.CrossSectionDistributionRecord(drec0)
        try:
            dec.SampleFinalState(dcdr, self._rng)
        except RuntimeError:
            return None
        dout = dataclasses.InteractionRecord()
        dout.signature = mu_sig
        dout.primary_momentum = drec0.primary_momentum
        dout.primary_mass = drec0.primary_mass
        dcdr.finalize(dout)
        dsecs = list(dout.secondary_momenta)
        dstypes = list(dout.signature.secondary_types)
        mu2 = None
        for t, pv in zip(dstypes, dsecs):
            if t in _MU_TYPES:
                mu2 = np.asarray(pv, float)
                break
        if mu2 is None:
            return None

        def _dir(p4):
            v = np.asarray(p4, float)[1:]
            n = np.linalg.norm(v)
            return v / n if n > 0 else np.array([0.0, 0.0, 1.0])

        return dict(
            mu1_dir=_dir(p_lep), mu1_E=float(p_lep[0]),
            mu2_dir=_dir(mu2), mu2_E=float(mu2[0]),
            had_E=float(p_had[0]), had_dir=_dir(p_had),
            D_type=D_type, decay_length=decay_length, weight=float(br_mu),
        )


# --------------------------------------------------------------------------- #
# Signal-at-satellite: mirrors background.compute_background_at_satellite, but
# the final state is the SIREN charm dimuon chain.
# --------------------------------------------------------------------------- #
def compute_charm_background_at_satellite(flux_geometry, model=None,
                                          detector_positions=None,
                                          N_samples=1000,
                                          max_cherenkov_events=None,
                                          uniform_gen=False,
                                          include_hadronic_shower=True,
                                          include_nc=True,
                                          R_det=R_det,
                                          pixel_deg=PIXEL_DEG_DEFAULT,
                                          mode="scattering"):
    """Charm dimuon background at the detector(s), per (event, detector).

    Mirrors ``background.compute_background_at_satellite``: beam neutrinos
    (m_N=0, U^2=1) are produced along the beam, travel upward, and interact in
    the atmospheric column with probability

        P_int = charm_fraction(E) * sigma_CC_nu(E) * column_depth

    (i.e. background.py's validated total-CC probability scaled by the SIREN
    charm fraction).  For evaluated events the SIREN charm chain provides the
    two muons and the hadronic shower, and Cherenkov photons are accumulated
    per muon per detector.

    Returns
    -------
    out : dict of per-event arrays (length N_samples)
        The mu1/mu2 photon-hit fields are the SAME unified schema the HNL signal
        writes (built by muon_image_spread.image_muons), so signal and background
        record identical per-muon information:
            mu{1,2}_n_ph, mu{1,2}_pix (,2), mu{1,2}_cen (,3), mu{1,2}_rms_deg,
            mu{1,2}_r90_deg, mu{1,2}_peak, mu{1,2}_in_pixel, mu{1,2}_det_z_km,
            mu{1,2}_E, mu{1,2}_dir (,3)
            opening_deg, opening_pixels, oncam_sep_deg, same_detector, both_detected
        Charm ADDS the hadronic-shower fields and charm kinematics:
            had_n_ph, had_pix (,2), had_cen (,3), had_E, had_dir (,3),
            nu_energy, int_pos (,3), D_decay_length, D_code
        plus the normalization: interaction_weights (P_int * BR(D->mu) *
        position_weight), N_nu_per_muon, cherenkov_weight, interaction_altitudes.
    The expected number of tagged background events is
        N_nu_per_muon * N_muon_decays * <interaction_weights * 1(tag)>_MC
    (see ``summarize_charm_background``).
    """
    detector_positions = [np.asarray(p, dtype=float) for p in detector_positions]
    N_det = len(detector_positions)
    max_det_height = max(p[2] for p in detector_positions)

    if model is None:
        model = CharmDISModel(nu_type=PT.NuMu, include_nc=include_nc)

    # --- 1. Neutrino production rate + points (m_N=0, U^2=1) ---
    prod_points, N_nu_per_muon, _, _ = flux_geometry.sample_production_points_weighted(
        m_N=0.0, U2=1.0, N_samples=N_samples, mode=mode)
    E_muon_local = muon_energy_in_earth(
        flux_geometry.E_mu, prod_points[:, -1] + flux_geometry.L_target)

    # --- 2. Neutrino kinematics (no m_N -> background MC kinematics) ---
    nu_energy, nu_dirs, _, _ = flux_geometry.sample_kinematics(
        prod_points[:, -1], E_muon_local, mode=mode)

    interaction_weights = np.zeros(N_samples)
    interaction_altitudes = np.zeros(N_samples)

    # Unified per-event output.  The mu1/mu2 photon-hit fields are the SAME schema
    # the HNL signal writes (built by muon_image_spread.image_muons); charm ADDS
    # the hadronic-shower fields (had_*) and the charm kinematics.
    out = {}
    for key in IMAGE_MU_KEYS:                    # per-muon shared photon-hit fields
        if key.endswith("_pix"):
            out[key] = np.full((N_samples, 2), np.nan)
        elif key.endswith("_cen") or key.endswith("_dir"):
            out[key] = np.full((N_samples, 3), np.nan)
        elif key.endswith("_n_ph"):
            out[key] = np.zeros(N_samples)
        elif key.endswith("_in_pixel"):
            out[key] = np.zeros(N_samples, dtype=bool)
        else:
            out[key] = np.full(N_samples, np.nan)
    out["opening_deg"] = np.full(N_samples, np.nan)
    out["opening_pixels"] = np.full(N_samples, np.nan)
    out["oncam_sep_deg"] = np.full(N_samples, np.nan)
    out["same_detector"] = np.zeros(N_samples, dtype=bool)
    out["both_detected"] = np.zeros(N_samples, dtype=bool)
    # hadronic shower (charm-only): count + compact detected spot (see below)
    out["had_n_ph"] = np.zeros(N_samples)
    out["had_pix"] = np.full((N_samples, 2), np.nan)
    out["had_cen"] = np.full((N_samples, 3), np.nan)
    out["had_E"] = np.zeros(N_samples)
    out["had_dir"] = np.full((N_samples, 3), np.nan)
    # charm kinematics / interaction geometry
    out["nu_energy"] = np.asarray(nu_energy, float).copy()
    out["int_pos"] = np.full((N_samples, 3), np.nan)
    out["D_decay_length"] = np.zeros(N_samples)
    out["D_code"] = np.zeros(N_samples, dtype=int)

    # --- 3. Upward-going neutrinos only ---
    going_up = nu_dirs[:, 2] > 0
    upward_indices = np.where(going_up)[0]
    if len(upward_indices) == 0:
        out["interaction_weights"] = interaction_weights
        out["N_nu_per_muon"] = N_nu_per_muon
        out["cherenkov_weight"] = 1.0
        out["interaction_altitudes"] = interaction_altitudes
        return out

    # --- 4. Interaction probability (charm fraction x total-CC prob) ---
    # Vectorized: charm fraction from the interpolation table, CC baseline and
    # column depth from background.py.
    column_depth = atmospheric_column_depth_nucleons(
        0, max_det_height, flux_geometry.cos_surface_exit_angle)
    E_up = np.asarray(nu_energy, float)[upward_indices]
    logE, frac = model._frac_grid
    f_charm = np.interp(np.log10(np.maximum(E_up, 1e-3)), logE, frac)
    interaction_weights[upward_indices] = (f_charm * sigma_CC_nu(E_up)
                                           * column_depth)

    # --- 5. Cap expensive Cherenkov/SIREN evaluations ---
    N_valid = len(upward_indices)
    if max_cherenkov_events is not None and N_valid > max_cherenkov_events:
        eval_indices = np.random.choice(upward_indices, max_cherenkov_events,
                                        replace=False)
        cherenkov_weight = N_valid / max_cherenkov_events
    else:
        eval_indices = upward_indices
        cherenkov_weight = 1.0

    # --- 6. Interaction altitudes (air-density profile) ---
    z_int_all, pos_weights_eval = sample_interaction_altitude(
        len(eval_indices), z_max_m=max_det_height, uniform_gen=uniform_gen,
        direction_cosine=flux_geometry.cos_surface_exit_angle)
    position_weights = np.ones(N_samples)
    for i_eval, idx in enumerate(eval_indices):
        position_weights[idx] = pos_weights_eval[i_eval]

    for i_eval, idx in enumerate(eval_indices):
        z_int = z_int_all[i_eval]
        interaction_altitudes[idx] = z_int

        # 3D interaction position along the neutrino trajectory
        t_to_z = (z_int - prod_points[idx, 2]) / nu_dirs[idx, 2]
        int_pos = prod_points[idx] + t_to_z * nu_dirs[idx]

        ev = model.sample_event(nu_energy[idx], nu_dirs[idx], current="cc")
        if ev is None:
            continue
        # fold the D->mu branching into this event's interaction weight: only a
        # fraction BR(D->mu) of charm events yield the second muon
        interaction_weights[idx] *= ev["weight"]

        # charm kinematics / interaction geometry
        out["int_pos"][idx] = int_pos
        out["had_E"][idx] = ev["had_E"]
        out["had_dir"][idx] = np.asarray(ev["had_dir"], float)
        out["D_decay_length"][idx] = ev["decay_length"]
        out["D_code"][idx] = int(ev["D_type"])

        # SHARED muon imaging -- identical per-muon photon-hit record to the HNL
        # signal (image_muons: uniform_n=N_AIR + transmission, matching the
        # analysis).  Fills mu1/mu2 counts, pixels, centroids, sizes, plus
        # opening_deg/opening_pixels/oncam_sep_deg/same_detector/both_detected.
        rec = image_muons(int_pos, (ev["mu1_dir"], ev["mu2_dir"]),
                          (ev["mu1_E"], ev["mu2_E"]), detector_positions,
                          R_det=R_det, N_psi=300, pixel_deg=pixel_deg,
                          uniform_n=N_AIR)
        if rec is not None:
            for k in IMAGE_MU_KEYS + IMAGE_PAIR_KEYS:
                out[k][idx] = rec[k]

        # --- Hadronic shower (charm-only) ---
        # sigma_had=10deg is only the EMISSION spread (sets the Gaussian
        # acceptance / the COUNT); the detected light comes from the ~point-like
        # shower, so it images to ~one beam-axis pixel at direction (det-int_pos).
        # Record the count + spot at the brightest hadronic camera.
        if include_hadronic_shower and ev["had_E"] > 1.0:
            best_cnt, best_det = 0.0, None
            for dp in detector_positions:
                dp = np.asarray(dp, float)
                if z_int >= dp[2]:
                    continue
                cnt = hadronic_shower_cherenkov(ev["had_E"], ev["had_dir"],
                                                int_pos - dp, z_int)
                if cnt > best_cnt:
                    best_cnt, best_det = cnt, dp
            if best_det is not None and best_cnt > 0:
                out["had_n_ph"][idx] = best_cnt
                d_arr = best_det - int_pos
                nrm = np.linalg.norm(d_arr)
                if nrm > 0:
                    d_hat = d_arr / nrm
                    out["had_cen"][idx] = d_hat
                    out["had_pix"][idx] = centroid_to_pixel(d_hat, pixel_deg)

    out["interaction_weights"] = interaction_weights * position_weights
    out["N_nu_per_muon"] = N_nu_per_muon
    out["cherenkov_weight"] = cherenkov_weight
    out["interaction_altitudes"] = interaction_altitudes
    return out


def summarize_charm_background(out, N_samples, min_photons=10, both_muon_tag=True):
    """Expected charm dimuon background events (scalar).

        N_bg = N_nu_per_muon * N_muon_decays * <interaction_weight * 1(tag)>_MC

    Uses the unified per-muon best-camera counts (same tag as the HNL signal):
    both muons above ``min_photons`` on the SAME camera (both_muon_tag=True), or
    at least one muon above threshold (both_muon_tag=False).  ``out`` is the dict
    returned by compute_charm_background_at_satellite.
    """
    m0 = np.asarray(out["mu1_n_ph"], float)
    m1 = np.asarray(out["mu2_n_ph"], float)
    if both_muon_tag:
        tagged = (m0 >= min_photons) & (m1 >= min_photons) \
            & np.asarray(out["same_detector"], bool)
    else:
        tagged = np.maximum(m0, m1) >= min_photons
    weighted = np.sum(np.asarray(out["interaction_weights"], float)[tagged]) \
        * float(out["cherenkov_weight"])
    return float(out["N_nu_per_muon"]) * N_muon_decays * weighted / N_samples
