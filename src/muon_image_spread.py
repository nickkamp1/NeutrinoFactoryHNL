"""Angular spread of the Cherenkov light from each muon of an HNL dimuon decay.

This module quantifies, event by event, how the Cherenkov photons of the two
muons from  N4 -> nu_mu mu- mu+  map onto the focal plane of an imaging air
Cherenkov telescope (Trinity-style, ~0.3 deg pixels).  It is meant to produce
the data behind two paper figures:

  (1) SINGLE-MUON COMPACTNESS.  For each muon, the Cherenkov photons that reach
      the detector arrive from a very narrow range of directions (a bare track
      viewed from tens of km through a 2 m aperture is nearly point-like), so a
      muon's whole image sits inside one pixel.  We report the angular RMS and
      the 68/90/95% containment radius of the arriving photon directions, plus a
      phase-averaged "fraction of light in the brightest pixel".

  (2) TWO-MUON SEPARATION.  The two muons are boosted apart by ~m_N/E_N, so
      their images land in DIFFERENT pixels.  We report the angular separation
      between the two muon image centroids (in degrees and in pixels).

Both are computed as a function of HNL mass and mixing: mass sets the decay
kinematics (opening angle), while mixing shifts the *detected* population
through the decay-length weighting, so the distributions move with U2.

The imaging geometry mirrors ``cherenkov.cherenkov_photons_multi_detector``
exactly (same altitude-dependent index of refraction, Cherenkov angle, and
disk-hit test); the only addition is that we record the photon ARRIVAL
DIRECTION for every contributing track step.  In the small-aperture limit
(R_det / distance ~ 3e-5 rad) the arrival direction of the photons that hit the
2 m disk equals, to ~0.003 deg, the direction from the emission point to the
detector -- far below a pixel -- so we use that direction per track step and
weight it by the photon yield of the step.

The heavy SIREN machinery is only needed for ``sample_dimuon_kinematics`` and
the scan drivers; the imaging core (``muon_image_at_detector``) and the
``selftest`` need only numpy + scipy and can run without SIREN.

Typical use
-----------
    from src.muon_image_spread import scan
    scan(masses=[5, 10, 20, 50], U2_grid=[1e-10, 1e-9],
         output="data/muon_image_spread.npz", N_samples=2000)

then histogram the saved per-event arrays (weighted by ``weight``) versus
``m_N`` / ``U2`` to make the figures.
"""
import numpy as np

from src.cherenkov import (orthonormal_basis, n_air_at_altitude,
                           cherenkov_transmission, ALPHA, LAMBDA_MIN, LAMBDA_MAX)
from src.balloon import muon_range_in_air
from src.constants import R_det as R_DET_DEFAULT

# Trinity design pixel (deg).  Demonstrator achieved 0.24 deg; see balloon_siren.
PIXEL_DEG_DEFAULT = 0.3


# --------------------------------------------------------------------------- #
# Imaging core (no SIREN dependency)
# --------------------------------------------------------------------------- #
def muon_image_at_detector(decay_pos, mu_dir, mu_energy, det_pos,
                           decay_altitude=None, R_det=R_DET_DEFAULT,
                           N_psi=300, N_track=None, pixel_deg=PIXEL_DEG_DEFAULT,
                           n_phase=300, apply_transmission=True, uniform_n=None):
    """Image statistics of ONE muon's Cherenkov light at ONE detector.

    Parameters
    ----------
    decay_pos : (3,) array
        Muon (HNL decay) start position [m], absolute coords (z = altitude).
    mu_dir : (3,) array
        Muon direction (unit vector; need not be normalised).
    mu_energy : float
        Muon lab energy [GeV] (sets the track length in air).
    det_pos : (3,) array
        Detector centre [m]; disk of radius R_det, normal along -z.
    decay_altitude : float or None
        Altitude of ``decay_pos`` [m]; defaults to ``decay_pos[2]`` clipped at 0.
    R_det, N_psi, N_track, pixel_deg : see module docstring.
        N_track defaults to the same adaptive rule as the signal calculation.
    n_phase : int
        Number of pixel-grid phase offsets per axis used to average the
        peak-pixel fraction (the image centroid falls at a random spot within a
        pixel; we average over that).
    apply_transmission : bool
        Multiply the photon count by the atmospheric transmission (matches the
        signal calculation).  Does not affect the angular statistics.
    uniform_n : float or None
        If None (default), use the altitude-dependent index of refraction
        n_air_at_altitude(z) -- the physically correct thin-air Cherenkov at
        altitude.  If set to a float (e.g. cherenkov.N_AIR = 1.0003), use that
        UNIFORM index for the whole track instead.  Pass N_AIR to make the
        photon count consistent with balloon_siren / charm_background, which call
        cherenkov_photons_multi_detector in its default uniform-sea-level-n mode
        (the two calculations are otherwise identical).  Only affects the photon
        yield / Cherenkov angle, not the imaging geometry.

    Returns
    -------
    dict or None
        None if no photons reach the detector.  Otherwise:
          n_photons     : total photons on the disk (after transmission)
          centroid      : (3,) unit vector, photon-weighted mean arrival dir
          rms_deg       : photon-weighted RMS angular spread [deg]
          r68/r90/r95_deg : containment radii [deg]
          peak_pixel_frac : phase-averaged fraction of light in brightest pixel
          diam90_deg    : 2*r90 (image diameter, for the "fits in a pixel" test)
          in_one_pixel  : bool, diam90_deg < pixel_deg
    """
    decay_pos = np.asarray(decay_pos, float)
    det_pos = np.asarray(det_pos, float)
    p_hat = np.asarray(mu_dir, float)
    nrm = np.linalg.norm(p_hat)
    if nrm == 0 or p_hat[2] <= 0:          # muon must go upward to reach detector
        return None
    p_hat = p_hat / nrm
    if decay_altitude is None:
        decay_altitude = max(0.0, decay_pos[2])
    if det_pos[2] <= decay_pos[2]:         # detector must be above the decay
        return None

    track_length = muon_range_in_air(mu_energy, decay_altitude,
                                     direction_cosine=p_hat[2])
    if N_track is None:
        N_track = min(1000, max(300, int(track_length / 100)))

    psi = np.linspace(0, 2 * np.pi, N_psi, endpoint=False)
    s = np.linspace(0, track_length, N_track)
    ds = track_length / (N_track - 1) if N_track > 1 else track_length

    r_emit = decay_pos[:, None] + s[None, :] * p_hat[:, None]           # (3,N_track)
    alt = np.maximum(decay_altitude + s * p_hat[2], 0.0)
    n_arr = (np.full(N_track, float(uniform_n)) if uniform_n is not None
             else n_air_at_altitude(alt))
    theta_C = np.arccos(1.0 / n_arr)
    dN_dx = 2 * np.pi * ALPHA * np.sin(theta_C) ** 2 * (1 / LAMBDA_MIN - 1 / LAMBDA_MAX)

    e1, e2 = orthonormal_basis(p_hat)
    # Cherenkov photon directions, (3, N_track, N_psi)
    k_hat = (np.cos(theta_C)[None, :, None] * p_hat[:, None, None]
             + np.sin(theta_C)[None, :, None]
             * (np.cos(psi)[None, None, :] * e1[:, None, None]
                + np.sin(psi)[None, None, :] * e2[:, None, None]))
    r_rel = (r_emit - det_pos[:, None])[:, :, None]                     # (3,N_track,1)
    with np.errstate(divide='ignore', invalid='ignore'):
        t = -r_rel[2] / k_hat[2]
    x_int = r_rel[0] + t * k_hat[0]
    y_int = r_rel[1] + t * k_hat[1]
    hit = (t > 0) & (x_int ** 2 + y_int ** 2 <= R_det ** 2)            # (N_track,N_psi)

    ti, pj = np.where(hit)
    if ti.size == 0:
        return None
    # ARRIVAL DIRECTION of each photon that reaches the disk = its own k_hat.
    # (Using the direction to the detector centre instead would drop the
    # intra-aperture spread ~R_det/distance, which dominates the tiny image.)
    k_hits = k_hat[:, ti, pj]                                          # (3,M)
    w = ds * dN_dx[ti] / N_psi                                        # photons per hit ray

    transmission = 1.0
    if apply_transmission:
        zenith = np.arccos(np.clip(p_hat[2], -1.0, 1.0))
        transmission = float(cherenkov_transmission(decay_altitude, zenith))
    n_photons = float(w.sum()) * transmission

    centroid = (k_hits * w).sum(axis=1)
    centroid /= np.linalg.norm(centroid)
    ang = np.degrees(np.arccos(np.clip(k_hits.T @ centroid, -1.0, 1.0)))  # (M,)
    rms = float(np.sqrt((w * ang ** 2).sum() / w.sum()))
    order = np.argsort(ang)
    cw = np.cumsum(w[order]) / w.sum()
    a_sorted = ang[order]

    def contain(frac):
        return float(a_sorted[min(np.searchsorted(cw, frac), len(a_sorted) - 1)])

    r68, r90, r95 = contain(0.68), contain(0.90), contain(0.95)
    peak = _peak_pixel_fraction(k_hits, w, centroid, pixel_deg, n_phase)
    return dict(n_photons=n_photons, centroid=centroid, rms_deg=rms,
                r68_deg=r68, r90_deg=r90, r95_deg=r95,
                peak_pixel_frac=peak, diam90_deg=2 * r90,
                in_one_pixel=bool(2 * r90 < pixel_deg))


def _peak_pixel_fraction(k_hits, w, centroid, pixel_deg, n_phase, max_hits=5000):
    """Fraction of photon weight in the single brightest pixel, averaged over a
    grid of pixel-grid phase offsets (image centroid is random within a pixel).

    k_hits : (3, M) photon arrival directions; w : (M,) photon weights."""
    if k_hits.shape[1] > max_hits:                  # cap cost; uniform thinning
        sel = np.linspace(0, k_hits.shape[1] - 1, max_hits).astype(np.int64)
        k_hits, w = k_hits[:, sel], w[sel]
    up = np.array([0.0, 0.0, 1.0])
    ex = np.cross(up, centroid)
    if np.linalg.norm(ex) < 1e-9:
        ex = np.array([1.0, 0.0, 0.0])
    ex /= np.linalg.norm(ex)
    ey = np.cross(centroid, ex)
    ey /= np.linalg.norm(ey)
    dx = np.degrees(k_hits.T @ ex)                  # tangent-plane offsets [deg]
    dy = np.degrees(k_hits.T @ ey)
    tot = w.sum()
    n_side = max(1, int(np.sqrt(n_phase)))
    offs = np.linspace(0.0, 1.0, n_side, endpoint=False)
    best = []
    for ox in offs:
        for oy in offs:
            ix = np.floor(dx / pixel_deg + ox).astype(np.int64)
            iy = np.floor(dy / pixel_deg + oy).astype(np.int64)
            key = ix * 1_000_003 + iy
            uk, inv = np.unique(key, return_inverse=True)
            acc = np.zeros(len(uk))
            np.add.at(acc, inv, w)
            best.append(acc.max() / tot)
    return float(np.mean(best))


def two_muon_separation_deg(centroid1, centroid2):
    """Angular separation [deg] between the two muon image centroids."""
    if centroid1 is None or centroid2 is None:
        return np.nan
    c = np.clip(np.dot(centroid1, centroid2), -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def centroid_to_pixel(centroid, pixel_deg=PIXEL_DEG_DEFAULT):
    """Camera pixel index (float) of an imaged arrival-direction unit vector.

    Every station is a downward-looking telescope (disk normal along -z), so the
    camera optical axis is the beam axis +z.  The beam axis therefore maps to
    pixel (0, 0), and "clustering around the beam axis" means small |pixel| --
    the handle for rejecting beam-collimated muon-decay-neutrino backgrounds.
    Camera axes are ex=[1,0,0], ey=[0,1,0]; pixel = degrees(atan2(c.e, c.z))/pixel_deg
    (exact gnomonic; linear for the <2 deg offsets here).  Returns (px_x, px_y)
    in pixel units, or (nan, nan) if the direction is missing or not upward.
    """
    if centroid is None:
        return (np.nan, np.nan)
    c = np.asarray(centroid, float)
    if c[2] <= 0:
        return (np.nan, np.nan)
    dx_deg = np.degrees(np.arctan2(c[0], c[2]))
    dy_deg = np.degrees(np.arctan2(c[1], c[2]))
    return (dx_deg / pixel_deg, dy_deg / pixel_deg)


# --------------------------------------------------------------------------- #
# SIREN front-half: sample a set of dimuon events (mirrors
# SIRENDimuonGeometry.compute_dimuon_signal_at_satellite up to the Cherenkov step)
# --------------------------------------------------------------------------- #
def sample_dimuon_kinematics(geom, m_N, U2, detector_positions, N_samples=2000,
                             seed=None, uniform_gen=False, sampling=None,
                             mix_frac_log=0.5, dmin_frac=1e-4):
    """Sample production + decay for N_samples HNL dimuon events.

    Returns a dict of arrays (length N_samples):
        decay_points (N,3), dir_mu1 (N,3), dir_mu2 (N,3), E_mu1, E_mu2,
        hnl_energy, decay_probability, decay_pos_probability, decay_dist,
        decay_length, d_max, valid_decay (bool), interaction_probability
        (scalar), BR_mumu (scalar).
    The physics is identical to compute_dimuon_signal_at_satellite.

    sampling : {"trunc_exp", "uniform", "mixture"} or None
        Decay-distance proposal (importance sampling).  None falls back to
        uniform_gen (True->"uniform", False->"trunc_exp").
          "trunc_exp" -- physical truncated exponential (no reweight coverage).
          "uniform"   -- uniform on (0, d_max); covers the LONG-lived regime.
          "mixture"   -- fraction ``mix_frac_log`` LOG-uniform on (dmin, d_max)
                         (dense at short decay distance -> covers the SHORT-lived
                         / high-U2 regime, i.e. the upper sensitivity edge) plus
                         the rest uniform (long-lived regime).  Best coverage for
                         reweighting across the whole (m_N, U2) plane.
        The returned ``sampling_pdf`` q(decay_dist) lets any target (m_N, U2) be
        reweighted analytically as decay_pos_probability = p_target/q (see
        src/hnl_sensitivity.py); decay_dist/decay_length/d_max are also returned.
    mix_frac_log, dmin_frac : float
        Log-uniform fraction and its lower edge dmin = max(d_max*dmin_frac, 1 m).
    """
    from src.xs_and_decays import HNL_decay_length
    from src.muon_beam_dump_helpers import muon_energy_in_earth, d_max_curved_earth

    if seed is not None:
        np.random.seed(seed)

    detector_positions = [np.asarray(p, float) for p in detector_positions]
    max_det_height = max(p[2] for p in detector_positions)
    BR_mumu = geom.dimuon_branching_ratio(m_N)

    prod_points, interaction_probability, _, _ = \
        geom.sample_production_points_weighted(m_N, U2, N_samples)
    E_muon_local = muon_energy_in_earth(geom.E_mu, prod_points[:, -1] + geom.L_target)
    hnl_energy, hnl_dirs, _, _ = geom.sample_kinematics(
        prod_points[:, -1], E_muon_local, m_N=m_N)
    E_mu1, dir_mu1, E_mu2, dir_mu2 = geom.sample_dimuon_decay(
        hnl_energy, hnl_dirs, m_N)

    decay_length = HNL_decay_length(m_N, U2, hnl_energy)
    cos_z = hnl_dirs[:, 2]
    upward = cos_z > 0
    d_max = np.where(upward, d_max_curved_earth(cos_z, max_det_height), 0.0)
    d_max = np.maximum(d_max, 0.0)
    mode = sampling if sampling is not None else ("uniform" if uniform_gen else "trunc_exp")
    dmin = np.maximum(d_max * dmin_frac, 1.0)          # log-uniform lower edge
    ln_ratio = np.log(np.maximum(d_max, dmin * 1.0001) / dmin)  # ln(d_max/dmin)
    with np.errstate(divide='ignore', invalid='ignore'):
        decay_probability = np.where(d_max > 0, 1.0 - np.exp(-d_max / decay_length), 0.0)
        if mode == "trunc_exp":
            # sample from the physical truncated exponential; q = p
            u = np.random.uniform(0, 1, N_samples)
            exp_term = np.exp(-d_max / decay_length)
            decay_dist = np.where(decay_probability > 0,
                                  -decay_length * np.log(1.0 - u * (1.0 - exp_term)), 0.0)
            sampling_pdf = np.where(
                decay_probability > 0,
                np.exp(-decay_dist / decay_length)
                / (decay_length * (1.0 - np.exp(-d_max / decay_length))), 1.0)
        elif mode == "uniform":
            decay_dist = np.random.uniform(0, d_max, N_samples)
            sampling_pdf = np.where(d_max > 0, 1.0 / d_max, 1.0)
        elif mode == "mixture":
            # MIXTURE proposal: fraction (1-mix_frac_log) uniform on (0, d_max)
            # for the long-lived regime, mix_frac_log log-uniform on (dmin, d_max)
            # for the short-lived regime.  Stored sampling_pdf = q(decay_dist).
            pick_log = np.random.uniform(0, 1, N_samples) < mix_frac_log
            uu = np.random.uniform(0, 1, N_samples)
            d_unif = uu * d_max
            d_log = dmin * (d_max / dmin) ** uu
            decay_dist = np.where(pick_log, d_log, d_unif)
            q_u = np.where(d_max > 0, 1.0 / d_max, 0.0)
            q_l = np.where(decay_dist >= dmin, 1.0 / (decay_dist * ln_ratio), 0.0)
            sampling_pdf = (1.0 - mix_frac_log) * q_u + mix_frac_log * q_l
            sampling_pdf = np.where(sampling_pdf > 0, sampling_pdf, 1.0)
        else:
            raise ValueError(f"unknown sampling mode {mode!r}")
        # decay_pos_probability = physical_pdf / sampling_pdf (importance weight)
        p_ref = np.where(decay_probability > 0,
                         np.exp(-decay_dist / decay_length)
                         / (decay_length * (1.0 - np.exp(-d_max / decay_length))), 0.0)
        decay_pos_probability = np.where(decay_probability > 0,
                                         p_ref / sampling_pdf, 0.0)
    decay_points = prod_points + decay_dist[:, None] * hnl_dirs
    valid_decay = (decay_points[:, 2] > 0) & (decay_probability > 0)

    return dict(decay_points=decay_points, dir_mu1=dir_mu1, dir_mu2=dir_mu2,
                E_mu1=E_mu1, E_mu2=E_mu2, hnl_energy=hnl_energy,
                decay_probability=decay_probability,
                decay_pos_probability=decay_pos_probability,
                sampling_pdf=sampling_pdf,
                decay_dist=decay_dist, decay_length=decay_length, d_max=d_max,
                valid_decay=valid_decay,
                interaction_probability=float(interaction_probability),
                BR_mumu=float(BR_mumu))


# --------------------------------------------------------------------------- #
# Shared muon-pair imaging (used by BOTH the HNL signal and the charm bkg so
# they record identical per-muon photon-hit information)
# --------------------------------------------------------------------------- #
# Unified per-muon photon-hit fields (N in {1,2}) + two-muon pair quantities.
# Both analyze_mass_mixing (HNL) and charm_background.compute_charm_background_at_satellite
# build their per-event records from these exact keys.
IMAGE_MU_FIELDS = ("n_ph", "pix", "cen", "rms_deg", "r90_deg", "peak",
                   "in_pixel", "det_z_km", "E", "dir")
IMAGE_MU_KEYS = tuple(f"mu{N}_{f}" for N in (1, 2) for f in IMAGE_MU_FIELDS)
IMAGE_PAIR_KEYS = ("opening_deg", "opening_pixels", "oncam_sep_deg",
                   "same_detector", "both_detected")
IMAGE_KEYS = IMAGE_MU_KEYS + IMAGE_PAIR_KEYS
# HNL-signal-specific per-event context stored alongside the shared image fields
# (weight + reweighting inputs).  Charm stores its own context instead.
HNL_CONTEXT_KEYS = ("weight", "hnl_energy", "decay_altitude", "decay_dist",
                    "decay_length", "d_max", "decay_pos_probability", "sampling_pdf")


def image_muons(vertex, mu_dirs, mu_energies, detector_positions,
                R_det=R_DET_DEFAULT, N_psi=300, pixel_deg=PIXEL_DEG_DEFAULT,
                uniform_n=None, N_track=None, E_min=0.1):
    """Image a muon PAIR emitted from ``vertex`` and return the unified record.

    Each muon is imaged with ``muon_image_at_detector`` (transmission applied) at
    every detector above the vertex and assigned to its BRIGHTEST detector.  This
    is the single source of truth for the per-muon photon-hit observables shared
    by the HNL signal and the charm background, so the two are guaranteed
    consistent (verified: the previous separate code paths agreed to machine
    precision).  ``apply_transmission=True`` here scales only ``n_ph`` (the
    centroid/rms/pixel are transmission-independent).

    Returns None if NEITHER muon reaches a detector.  Otherwise a dict with, for
    N in {1,2}:
        muN_n_ph      photons on the disk (transmission applied); 0.0 if unimaged
        muN_pix       (2,) camera pixel, beam-axis referenced; [nan,nan] if unimaged
        muN_cen       (3,) photon-image centroid unit vector;   [nan]*3 if unimaged
        muN_rms_deg, muN_r90_deg, muN_peak, muN_in_pixel   image compactness
        muN_det_z_km  chosen detector beam-frame z [km];        nan if unimaged
        muN_E         muon energy [GeV]
        muN_dir       (3,) muon momentum unit vector
    plus two-muon pair quantities:
        opening_deg, opening_pixels, oncam_sep_deg, same_detector, both_detected
    """
    vertex = np.asarray(vertex, float)
    alt = max(0.0, vertex[2])
    dets_above = [np.asarray(d, float) for d in detector_positions
                  if np.asarray(d, float)[2] > vertex[2]]
    ims = [None, None]
    best = [None, None]
    for k in (0, 1):
        E = float(mu_energies[k])
        mu = np.asarray(mu_dirs[k], float)
        if E < E_min:
            continue
        bim, bd = None, None
        for d in dets_above:
            im = muon_image_at_detector(vertex, mu, E, d, decay_altitude=alt,
                                        R_det=R_det, N_psi=N_psi, N_track=N_track,
                                        pixel_deg=pixel_deg, uniform_n=uniform_n)
            if im is None:
                continue
            if bim is None or im["n_photons"] > bim["n_photons"]:
                bim, bd = im, d
        ims[k], best[k] = bim, bd
    if ims[0] is None and ims[1] is None:
        return None

    rec = {}
    for k in (0, 1):
        im, bd, N = ims[k], best[k], k + 1
        if im is not None:
            cen = np.asarray(im["centroid"], float)
            rec[f"mu{N}_n_ph"] = float(im["n_photons"])
            rec[f"mu{N}_pix"] = np.asarray(centroid_to_pixel(cen, pixel_deg), float)
            rec[f"mu{N}_cen"] = cen
            rec[f"mu{N}_rms_deg"] = float(im["rms_deg"])
            rec[f"mu{N}_r90_deg"] = float(im["r90_deg"])
            rec[f"mu{N}_peak"] = float(im["peak_pixel_frac"])
            rec[f"mu{N}_in_pixel"] = bool(im["in_one_pixel"])
            rec[f"mu{N}_det_z_km"] = float(bd[2] / 1e3)
        else:
            rec[f"mu{N}_n_ph"] = 0.0
            rec[f"mu{N}_pix"] = np.array([np.nan, np.nan])
            rec[f"mu{N}_cen"] = np.array([np.nan, np.nan, np.nan])
            rec[f"mu{N}_rms_deg"] = np.nan
            rec[f"mu{N}_r90_deg"] = np.nan
            rec[f"mu{N}_peak"] = np.nan
            rec[f"mu{N}_in_pixel"] = False
            rec[f"mu{N}_det_z_km"] = np.nan
        mu = np.asarray(mu_dirs[k], float)
        nrm = np.linalg.norm(mu)
        rec[f"mu{N}_dir"] = mu / nrm if nrm > 0 else mu
        rec[f"mu{N}_E"] = float(mu_energies[k])

    n1, n2 = rec["mu1_dir"], rec["mu2_dir"]
    both = (ims[0] is not None) and (ims[1] is not None)
    same = both and (best[0] is best[1])
    rec["opening_deg"] = float(np.degrees(np.arccos(np.clip(n1 @ n2, -1.0, 1.0))))
    rec["opening_pixels"] = rec["opening_deg"] / pixel_deg
    rec["same_detector"] = bool(same)
    rec["both_detected"] = bool(both)
    rec["oncam_sep_deg"] = (two_muon_separation_deg(ims[0]["centroid"], ims[1]["centroid"])
                            if same else np.nan)
    return rec


# --------------------------------------------------------------------------- #
# Per-(mass, mixing) analysis
# --------------------------------------------------------------------------- #
def analyze_mass_mixing(geom, m_N, U2, detector_positions, N_samples=2000,
                        max_events=None, R_det=R_DET_DEFAULT,
                        pixel_deg=PIXEL_DEG_DEFAULT, N_psi=300, seed=None,
                        verbose=True, uniform_n=None, uniform_gen=False,
                        sampling=None):
    """Per-event image statistics for one (m_N, U2).

    For every valid decay, each muon is imaged at the detector it illuminates
    most brightly (its Cherenkov beam is a narrow searchlight, so the two muons
    usually light up DIFFERENT detectors).  We record per-muon compactness and
    the two-muon angular separation.  The separation reported is the lab opening
    angle between the two muon directions -- which equals the focal-plane
    separation the two spots would have on a camera that saw both, and is
    defined whether or not they share a detector.

    A record is kept for every valid decay in which AT LEAST ONE muon reaches a
    detector.  Muon-2 fields are NaN/False when only muon 1 was imaged (and vice
    versa); ``both_detected`` flags the events where both were imaged.

    Returns a dict of 1-D per-event arrays:
        weight        detection weight (decay_probability * cherenkov_weight);
                      multiply by interaction_probability*BR_mumu*N_muon_decays
                      for an absolute event count.
        m_N, U2       broadcast scalars (for concatenating across a scan)
    The per-muon photon-hit fields are the SHARED unified schema (image_muons),
    identical to what the charm background records (N in {1,2}):
        muN_n_ph              photons per muon (transmission applied)
        muN_pix (,2), muN_cen (,3)  camera pixel + centroid dir (beam-axis ref)
        muN_rms_deg, muN_r90_deg, muN_peak, muN_in_pixel   image compactness
        muN_det_z_km, muN_E, muN_dir (,3)   chosen detector, energy, momentum dir
        opening_deg, opening_pixels, oncam_sep_deg, same_detector, both_detected
    plus the HNL context: weight, hnl_energy, decay_altitude, and the reweight
    inputs decay_dist/decay_length/d_max/decay_pos_probability/sampling_pdf.
    Plus scalars: interaction_probability, BR_mumu, cherenkov_weight, n_events.
    """
    kin = sample_dimuon_kinematics(geom, m_N, U2, detector_positions,
                                   N_samples=N_samples, seed=seed,
                                   uniform_gen=uniform_gen, sampling=sampling)
    det_pos = [np.asarray(p, float) for p in detector_positions]
    valid_idx = np.where(kin["valid_decay"])[0]

    cherenkov_weight = 1.0
    if max_events is not None and len(valid_idx) > max_events:
        valid_idx = np.random.choice(valid_idx, max_events, replace=False)
        cherenkov_weight = len(np.where(kin["valid_decay"])[0]) / max_events

    keys = IMAGE_KEYS + HNL_CONTEXT_KEYS
    cols = {k: [] for k in keys}

    d1 = kin["dir_mu1"]; d2 = kin["dir_mu2"]
    for n_i, i in enumerate(valid_idx):
        if verbose and n_i % 500 == 0:      # throttle: '\r' floods redirected logs
            print(f"Processing event {n_i} out of {len(valid_idx)}...", end='\r')
        dp = kin["decay_points"][i]
        # SHARED imaging: identical per-muon photon-hit record to the charm bkg.
        rec = image_muons(dp, (d1[i], d2[i]),
                          (kin["E_mu1"][i], kin["E_mu2"][i]), detector_positions,
                          R_det=R_det, N_psi=N_psi, pixel_deg=pixel_deg,
                          uniform_n=uniform_n)
        if rec is None:                          # neither muon reached a detector
            continue
        for k in IMAGE_KEYS:
            cols[k].append(rec[k])
        cols["weight"].append(kin["decay_probability"][i]
                              * kin["decay_pos_probability"][i] * cherenkov_weight)
        cols["hnl_energy"].append(kin["hnl_energy"][i])
        cols["decay_altitude"].append(max(0.0, dp[2]))
        cols["decay_dist"].append(kin["decay_dist"][i])
        cols["decay_length"].append(kin["decay_length"][i])
        cols["d_max"].append(kin["d_max"][i])
        cols["decay_pos_probability"].append(kin["decay_pos_probability"][i])
        cols["sampling_pdf"].append(kin["sampling_pdf"][i])

    out = {k: np.asarray(v) for k, v in cols.items()}
    n_ev = len(out["weight"])
    out["m_N"] = np.full(n_ev, m_N, float)
    out["U2"] = np.full(n_ev, U2, float)
    out["interaction_probability"] = kin["interaction_probability"]
    out["BR_mumu"] = kin["BR_mumu"]
    out["cherenkov_weight"] = cherenkov_weight
    out["n_events"] = n_ev
    if verbose:
        rms_all = np.concatenate([out["mu1_rms_deg"], out["mu2_rms_deg"]]) if n_ev else np.array([])
        med_rms = np.nanmedian(rms_all) if rms_all.size else np.nan
        med_open = np.nanmedian(out["opening_deg"]) if n_ev else np.nan
        fpix = np.nanmean(out["opening_pixels"] > 1) if n_ev else np.nan
        fboth = np.mean(out["both_detected"]) if n_ev else np.nan
        print(f"m_N={m_N:6.2f} U2={U2:.1e}  events={n_ev:5d}  "
              f"median muon RMS={med_rms:.4f} deg  median opening={med_open:.3f} deg  "
              f"frac(open>1pix)={fpix:.2f}  frac(both det)={fboth:.2f}")
    return out


def scan(masses, U2_grid, detector_positions=None, output=None,
         N_samples=2000, max_events=2000, geom=None, E_mu=5000,
         dump_depth=100, dump_angle=1.53, nature="Majorana", seed=12345,
         pixel_deg=PIXEL_DEG_DEFAULT, R_det=R_DET_DEFAULT, N_psi=300,
         uniform_n=None, uniform_gen=False, sampling=None):
    """Scan (m_N, U2) and concatenate the per-event image statistics.

    If ``output`` is given, saves a flat npz whose per-event arrays are the
    concatenation over the grid, tagged by the per-event ``m_N`` and ``U2``
    columns (so you can filter/weight for any figure).  Returns the same dict.
    """
    if detector_positions is None:
        detector_positions = [
            #np.array([500, 0, 20000.0]), np.array([-500, 0, 20000.0]), np.array([0, 0, 20000.0]),
            #np.array([500, 0, 50000.0]), np.array([-500, 0, 50000.0]), np.array([0, 0, 50000.0]),
            #np.array([500, 0, 100000.0]),
            #np.array([-500, 0, 100000.0]),
            np.array([0, 0, 100000.0]),
        ]
    if geom is None:
        from src.balloon_siren import SIRENDimuonGeometry
        geom = SIRENDimuonGeometry(E_mu=E_mu, dump_depth=dump_depth,
                                   dump_angle=dump_angle, nature=nature, seed=seed)

    per_event_keys = ("m_N", "U2") + IMAGE_KEYS + HNL_CONTEXT_KEYS
    acc = {k: [] for k in per_event_keys}
    meta = {}
    for m_N in masses:
        for U2 in U2_grid:
            res = analyze_mass_mixing(geom, m_N, U2, detector_positions,
                                      N_samples=N_samples, max_events=max_events,
                                      R_det=R_det, pixel_deg=pixel_deg, N_psi=N_psi,
                                      uniform_n=uniform_n, uniform_gen=uniform_gen,
                                      sampling=sampling)
            for k in per_event_keys:
                acc[k].append(res[k])
            meta[(m_N, U2)] = dict(interaction_probability=res["interaction_probability"],
                                   BR_mumu=res["BR_mumu"],
                                   cherenkov_weight=res["cherenkov_weight"],
                                   n_events=res["n_events"])
    flat = {k: (np.concatenate(v) if len(v) else np.array([])) for k, v in acc.items()}
    flat["pixel_deg"] = pixel_deg
    # Camera pixel convention (see centroid_to_pixel): optical axis = beam axis
    # = +z, so the beam axis is at pixel (0,0) for every station -- the origin
    # for beam-axis-clustering cuts on the muon-decay-neutrino background.
    flat["beam_axis_pixel"] = np.array([0.0, 0.0])
    flat["camera_optical_axis"] = np.array([0.0, 0.0, 1.0])
    flat["masses"] = np.asarray(masses, float)
    flat["U2_grid"] = np.asarray(U2_grid, float)
    # meta as parallel arrays keyed by (m_N,U2)
    mk = list(meta.keys())
    flat["meta_m_N"] = np.array([k[0] for k in mk], float)
    flat["meta_U2"] = np.array([k[1] for k in mk], float)
    flat["meta_interaction_probability"] = np.array([meta[k]["interaction_probability"] for k in mk])
    flat["meta_BR_mumu"] = np.array([meta[k]["BR_mumu"] for k in mk])
    flat["meta_cherenkov_weight"] = np.array([meta[k]["cherenkov_weight"] for k in mk])
    flat["meta_n_events"] = np.array([meta[k]["n_events"] for k in mk])
    # N_samples = number of beam MC events THROWN per grid point (not just the
    # imaged/valid subset in meta_n_events).  It is the correct outer normalization
    # for the reweighted event rate (see hnl_sensitivity.reweight_hnl); without it,
    # dividing by meta_n_events over-counts by N_samples/n_events (~25x).
    flat["meta_N_samples"] = np.full(len(mk), float(N_samples))
    flat["uniform_gen"] = bool(uniform_gen)
    flat["sampling"] = (sampling if sampling is not None
                        else ("uniform" if uniform_gen else "trunc_exp"))
    if output is not None:
        np.savez(output, **flat)
        print(f"Saved per-event image statistics to {output}  "
              f"({len(flat['weight'])} events over {len(mk)} grid points)")
    return flat


# --------------------------------------------------------------------------- #
# Self-test of the imaging core (no SIREN needed)
# --------------------------------------------------------------------------- #
def selftest():
    """Sanity checks on the imaging core using synthetic muon tracks."""
    det = np.array([0.0, 0.0, 100000.0])
    print("single-muon image (should be << 0.3 deg pixel):")
    print(f"  {'h_dec[km]':>9} {'thetaC[deg]':>11} {'rms[deg]':>9} "
          f"{'r90[deg]':>9} {'peakfrac':>9} {'in_pixel':>8} {'N_ph':>10}")
    for h_km in (10, 30, 60, 90):
        h = h_km * 1e3
        u = det - np.array([0, 0, h]); u = u / np.linalg.norm(u)
        thC = np.degrees(np.arccos(1.0 / n_air_at_altitude(h)))
        im = muon_image_at_detector(np.array([0, 0, h]), u, 100.0, det,
                                    decay_altitude=h, apply_transmission=False)
        print(f"  {h_km:9d} {thC:11.3f} {im['rms_deg']:9.4f} {im['r90_deg']:9.4f} "
              f"{im['peak_pixel_frac']:9.3f} {str(im['in_one_pixel']):>8} "
              f"{im['n_photons']:10.1f}")
    # Two muons imaged at their OWN (different) detectors, tilted by +/- theta_C
    # so each actually reaches a detector; recover the opening angle from the
    # image centroids.  (A single 2 m detector can rarely see both muons: their
    # Cherenkov beams are only ~theta_C wide, so they light up different
    # detectors -- which is why analyze_mass_mixing reports the direction
    # opening angle, not a shared-detector separation.)
    print("\ntwo-muon opening angle recovered from separate-detector images:")
    h = 30e3
    thC = np.arccos(1.0 / n_air_at_altitude(h))
    for det_a, det_b, tag in (
            (np.array([0.0, 0.0, 100000.0]), np.array([0.0, 0.0, 100000.0]), "same det"),
            (np.array([300.0, 0.0, 100000.0]), np.array([-300.0, 0.0, 100000.0]), "two dets")):
        ua = det_a - np.array([0, 0, h]); ua /= np.linalg.norm(ua)
        ub = det_b - np.array([0, 0, h]); ub /= np.linalg.norm(ub)
        im1 = muon_image_at_detector(np.array([0, 0, h]), ua, 100.0, det_a, h)
        im2 = muon_image_at_detector(np.array([0, 0, h]), ub, 100.0, det_b, h)
        if im1 is None or im2 is None:
            print(f"  [{tag}] a muon missed its detector"); continue
        sep = two_muon_separation_deg(im1["centroid"], im2["centroid"])
        true_open = np.degrees(np.arccos(np.clip(ua @ ub, -1, 1)))
        print(f"  [{tag}] centroid sep = {sep:.3f} deg  (true opening {true_open:.3f} deg, "
              f"theta_C={np.degrees(thC):.3f} deg) = {sep / PIXEL_DEG_DEFAULT:.2f} pixels")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selftest", action="store_true",
                   help="Run the SIREN-free imaging-core sanity checks and exit.")
    p.add_argument("--masses", type=float, nargs="+",
                   default=[5, 10, 20, 50])
    p.add_argument("--U2", type=float, nargs="+", default=[1e-10, 1e-9])
    p.add_argument("--n-samples", type=int, default=2000)
    p.add_argument("--max-events", type=int, default=2000)
    p.add_argument("--output", type=str, default="data/muon_image_spread.npz")
    p.add_argument("--pixel-deg", type=float, default=PIXEL_DEG_DEFAULT)
    args = p.parse_args()

    if args.selftest:
        selftest()
    else:
        scan(args.masses, args.U2, output=args.output,
             N_samples=args.n_samples, max_events=args.max_events,
             pixel_deg=args.pixel_deg)
