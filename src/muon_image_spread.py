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
        rms1_deg, rms2_deg, r90_1_deg, r90_2_deg   per-muon image size [deg]
        peak1, peak2          brightest-pixel light fraction per muon
        in_pixel1, in_pixel2  bool, muon image fits in one pixel
        opening_deg, opening_pixels  two-muon lab opening angle (Fig 2 metric)
        oncam_sep_deg         centroid separation when both share a detector (else NaN)
        same_detector, both_detected  bools
        n_ph1, n_ph2          photons per muon (after transmission)
        E_mu1, E_mu2, hnl_energy, decay_altitude
        det1_alt_km, det2_alt_km  altitude of each muon's chosen detector
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

    keys = ("weight", "rms1_deg", "rms2_deg", "r90_1_deg", "r90_2_deg",
            "peak1", "peak2", "in_pixel1", "in_pixel2",
            "opening_deg", "opening_pixels", "oncam_sep_deg",
            "same_detector", "both_detected", "n_ph1", "n_ph2",
            # camera pixel index of each muon spot, referenced to the beam axis
            # (beam axis -> pixel (0,0)); NaN when that muon was not imaged.
            "mu1_pix_x", "mu1_pix_y", "mu2_pix_x", "mu2_pix_y",
            # photon-image centroid DIRECTION (unit vector) per muon, so pixels
            # can be re-derived at ANY density via centroid_to_pixel; NaN if unimaged.
            "mu1_cen_x", "mu1_cen_y", "mu1_cen_z",
            "mu2_cen_x", "mu2_cen_y", "mu2_cen_z",
            "E_mu1", "E_mu2", "hnl_energy", "decay_altitude",
            "det1_alt_km", "det2_alt_km",
            # reweighting inputs: sampling_pdf q(decay_dist) lets hnl_sensitivity
            # reweight to any (m_N, U2) analytically as p_target/q
            "decay_dist", "decay_length", "d_max", "decay_pos_probability",
            "sampling_pdf")
    cols = {k: [] for k in keys}

    def best_detector(mu_dir, E_mu, dp, alt, dets):
        """Image mu at each detector above the decay; return (image, det) for
        the detector with the most photons, or (None, None)."""
        best_im, best_d = None, None
        for d in dets:
            im = muon_image_at_detector(dp, mu_dir, E_mu, d, alt, R_det, N_psi,
                                        pixel_deg=pixel_deg, uniform_n=uniform_n)
            if im is None:
                continue
            if best_im is None or im["n_photons"] > best_im["n_photons"]:
                best_im, best_d = im, d
        return best_im, best_d

    d1 = kin["dir_mu1"]; d2 = kin["dir_mu2"]
    for n_i, i in enumerate(valid_idx):
        if verbose and n_i % 500 == 0:      # throttle: '\r' floods redirected logs
            print(f"Processing event {n_i} out of {len(valid_idx)}...", end='\r')
        dp = kin["decay_points"][i]
        alt = max(0.0, dp[2])
        dets_above = [d for d in det_pos if d[2] > dp[2]]
        if not dets_above:
            continue
        im1, det1 = best_detector(d1[i], kin["E_mu1"][i], dp, alt, dets_above)
        im2, det2 = best_detector(d2[i], kin["E_mu2"][i], dp, alt, dets_above)
        if im1 is None and im2 is None:
            continue
        # Lab opening angle between the two muon directions (Fig 2 metric).
        n1 = d1[i] / np.linalg.norm(d1[i]); n2 = d2[i] / np.linalg.norm(d2[i])
        opening = float(np.degrees(np.arccos(np.clip(n1 @ n2, -1.0, 1.0))))
        both = (im1 is not None) and (im2 is not None)
        same_det = both and det1 is det2
        oncam = (two_muon_separation_deg(im1["centroid"], im2["centroid"])
                 if same_det else np.nan)

        cols["weight"].append(kin["decay_probability"][i]
                              * kin["decay_pos_probability"][i] * cherenkov_weight)
        cols["rms1_deg"].append(im1["rms_deg"] if im1 else np.nan)
        cols["rms2_deg"].append(im2["rms_deg"] if im2 else np.nan)
        cols["r90_1_deg"].append(im1["r90_deg"] if im1 else np.nan)
        cols["r90_2_deg"].append(im2["r90_deg"] if im2 else np.nan)
        cols["peak1"].append(im1["peak_pixel_frac"] if im1 else np.nan)
        cols["peak2"].append(im2["peak_pixel_frac"] if im2 else np.nan)
        cols["in_pixel1"].append(im1["in_one_pixel"] if im1 else False)
        cols["in_pixel2"].append(im2["in_one_pixel"] if im2 else False)
        cols["opening_deg"].append(opening)
        cols["opening_pixels"].append(opening / pixel_deg)
        cols["oncam_sep_deg"].append(oncam)
        cols["same_detector"].append(bool(same_det))
        cols["both_detected"].append(bool(both))
        cols["n_ph1"].append(im1["n_photons"] if im1 else 0.0)
        cols["n_ph2"].append(im2["n_photons"] if im2 else 0.0)
        px1 = centroid_to_pixel(im1["centroid"], pixel_deg) if im1 else (np.nan, np.nan)
        px2 = centroid_to_pixel(im2["centroid"], pixel_deg) if im2 else (np.nan, np.nan)
        cols["mu1_pix_x"].append(px1[0]); cols["mu1_pix_y"].append(px1[1])
        cols["mu2_pix_x"].append(px2[0]); cols["mu2_pix_y"].append(px2[1])
        c1 = im1["centroid"] if im1 else (np.nan, np.nan, np.nan)
        c2 = im2["centroid"] if im2 else (np.nan, np.nan, np.nan)
        cols["mu1_cen_x"].append(c1[0]); cols["mu1_cen_y"].append(c1[1]); cols["mu1_cen_z"].append(c1[2])
        cols["mu2_cen_x"].append(c2[0]); cols["mu2_cen_y"].append(c2[1]); cols["mu2_cen_z"].append(c2[2])
        cols["E_mu1"].append(kin["E_mu1"][i])
        cols["E_mu2"].append(kin["E_mu2"][i])
        cols["hnl_energy"].append(kin["hnl_energy"][i])
        cols["decay_altitude"].append(alt)
        cols["det1_alt_km"].append(det1[2] / 1e3 if det1 is not None else np.nan)
        cols["det2_alt_km"].append(det2[2] / 1e3 if det2 is not None else np.nan)
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
        rms_all = np.concatenate([out["rms1_deg"], out["rms2_deg"]]) if n_ev else np.array([])
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

    per_event_keys = ("weight", "m_N", "U2", "rms1_deg", "rms2_deg",
                      "r90_1_deg", "r90_2_deg", "peak1", "peak2",
                      "in_pixel1", "in_pixel2", "opening_deg", "opening_pixels",
                      "oncam_sep_deg", "same_detector", "both_detected",
                      "n_ph1", "n_ph2",
                      "mu1_pix_x", "mu1_pix_y", "mu2_pix_x", "mu2_pix_y",
                      "mu1_cen_x", "mu1_cen_y", "mu1_cen_z",
                      "mu2_cen_x", "mu2_cen_y", "mu2_cen_z",
                      "E_mu1", "E_mu2", "hnl_energy",
                      "decay_altitude", "det1_alt_km", "det2_alt_km",
                      "decay_dist", "decay_length", "d_max", "decay_pos_probability",
                      "sampling_pdf")
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
