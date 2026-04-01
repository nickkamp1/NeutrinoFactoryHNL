# Beam-induced neutrino background for balloon/satellite HNL detection
#
# Physics: The same DIS process that produces HNLs also produces neutrinos:
#   μ + N → ν_μ + X  (identical to HNL production with m_N → 0, U² = 1)
#
# These neutrinos travel upward and can interact in the atmosphere via CC:
#   ν_μ + N → μ + X
# producing muons that emit Cherenkov light — mimicking the HNL signal.
#
# Key difference from signal: the neutrino does NOT decay. Instead, it has
# a tiny probability P_CC of interacting in the atmospheric column. Each MC
# event carries this weight.
#
# The background rate is:
#   N_bg = N_ν_per_muon × N_muons × <P_CC × P_detect>_MC
#
# The altitude distribution of background events follows n_air(z) ~ exp(-z/H),
# while HNL decays follow exp(-z/L_decay). With L_decay >> H, the signal
# is flatter in z than the background — exploitable with two balloons at
# different heights.

import numpy as np

from src.constants import *
from src.xs_and_decays import sigma
from src.cherenkov import (cherenkov_photons_detected_vectorized,
                          cherenkov_photons_multi_detector,
                          cherenkov_transmission)
from src.balloon import air_density, muon_range_in_air, hadronic_shower_cherenkov


# Neutrino interaction constants
SIGMA_CC_PER_NUCLEON_PER_GEV = 0.67e-38  # cm^2 / nucleon / GeV (isoscalar target)
N_AVOGADRO = 6.022e23  # nucleons per mol
A_AIR = 14.5  # effective atomic mass of air (N2/O2 mix)

# Atmosphere (matching balloon.py)
RHO_AIR_0 = 1.225e-3  # sea level density [g/cm³]
H_SCALE_CM = 8500.0 * 100  # scale height [cm]


def sigma_CC_nu(E_nu_GeV):
    """Total CC neutrino-nucleon cross section [cm² per nucleon]."""
    return SIGMA_CC_PER_NUCLEON_PER_GEV * np.asarray(E_nu_GeV, dtype=float)


def atmospheric_column_depth_nucleons(z_start_m, z_end_m, direction_cosine):
    """
    Nucleon column depth [nucleons/cm²] along a slant path through atmosphere.

    Parameters
    ----------
    z_start_m, z_end_m : float
        Start and end altitude [m] (z_end > z_start)
    direction_cosine : float
        cos(zenith angle) of the path (1 = vertical)

    Returns
    -------
    column : float
        Nucleon column depth [nucleons/cm²]
    """
    z0 = z_start_m * 100  # cm
    z1 = z_end_m * 100    # cm
    cos_z = np.maximum(np.abs(direction_cosine), 0.05)
    H_slant = H_SCALE_CM / cos_z

    # Vertical grammage from z0 to z1: integral rho(z) dz
    grammage = RHO_AIR_0 * H_slant * (np.exp(-z0 / H_slant)
                                          - np.exp(-z1 / H_slant))

    # Slant correction and convert to nucleons
    return grammage * N_AVOGADRO / A_AIR


def sample_interaction_altitude(N, z_max_m, z_min_m=0., direction_cosine=1.0, uniform_gen=False):
    """
    Sample neutrino interaction altitudes from the air density profile.

    The interaction probability density is proportional to n_air(z) ~ exp(-z cos(θ)/H).
    We sample from the truncated exponential on [z_min, z_max].

    Parameters
    ----------
    N : int
        Number of samples
    z_max_m : float
        Maximum altitude [m] (satellite height)
    z_min_m : float
        Minimum altitude [m] (default 0, sea level)
    direction_cosine : float
        cos(zenith angle) of the neutrino path (default 1 = vertical)
    uniform_gen : bool
        If True, sample uniformly in [z_min, z_max] and weight by exp(-z/H).
         This can reduce variance if many events are near the satellite height.

    Returns
    -------
    z : ndarray, shape (N,)
        Sampled altitudes [m]
    w : ndarray, shape (N,)
        Sample weights
    """
    H = 8500.0  # m
    cos_z = np.maximum(np.abs(direction_cosine), 0.05) # Avoid too small cosines
    Hslant = H / cos_z # effective scale height along the slant path

    if uniform_gen:
        z = np.random.uniform(z_min_m,z_max_m, N)
        w = np.exp(-z/Hslant)*(z_max_m-z_min_m) / (Hslant*(np.exp(-z_min_m/Hslant) - np.exp(-z_max_m/Hslant)))
        return z,w
    else:
        u = np.random.uniform(0, 1, N)
        exp_min = np.exp(-z_min_m / Hslant)
        exp_max = np.exp(-z_max_m / Hslant)
        return -Hslant * np.log(exp_min - u * (exp_min - exp_max)), np.ones(N)


def compute_background_at_satellite(flux_geometry,
                                    detector_positions=None,
                                    N_samples=1000,
                                    max_cherenkov_events=None,
                                    uniform_gen=False,
                                    include_hadronic_shower=True):
    """
    Compute neutrino background using the same DIS framework as the HNL signal.

    Neutrinos produced by μ + N → ν_μ + X (m_N=0, U²=1) interact in the
    atmosphere via CC, producing muons whose Cherenkov light reaches the
    detector(s). Each event is weighted by P_CC, the probability that the
    neutrino interacts in the atmospheric column.

    Parameters
    ----------
    flux_geometry : HNLFluxGeometry
        Geometry configuration (provides production points, MC kinematics, etc.)
    N_samples : int
        Number of MC neutrino events
    max_cherenkov_events : int or None
        Cap on expensive Cherenkov evaluations
    detector_positions : list of array-like
        List of 3D detector positions [m].

    Returns
    -------
    photon_counts : ndarray
        Cherenkov photons per event. Shape (N_samples,) for single detector,
        or (N_det, N_samples) for multiple detectors.
    interaction_weights : ndarray, shape (N_samples,)
        P_CC weight for each event
    N_nu_per_muon : float
        Neutrino production rate per muon traversal
    cherenkov_weight : float
        Reweight factor if Cherenkov evaluations were capped
    interaction_altitudes : ndarray, shape (N_samples,)
        Sampled interaction altitude for each event [m] (0 if not evaluated)
    """
    # Set up detector positions
    detector_positions = [np.asarray(p, dtype=float) for p in detector_positions]
    N_det = len(detector_positions)
    max_det_height = max(p[2] for p in detector_positions)

    # --- 1. Neutrino production rate ---
    # Same as HNL with m_N=0, U²=1: sigma(E_mu, 0, 1) integrated over depth
    N_nu_per_muon, _, _ = \
        flux_geometry.compute_weighted_production_rate(m_N=0.0, U2=1.0)

    # --- 2. Sample neutrino kinematics ---
    # Passing no m_N value uses the neutrino background MC kinematics (no HNL mass, U²=1)
    nu_energy, nu_dirs, _, _ = flux_geometry.sample_kinematics(N_samples)

    # --- 3. Sample production points (weighted by local cross section) ---
    prod_points, _ = flux_geometry.sample_production_points_weighted(
        m_N=0.0, U2=1.0, N_samples=N_samples
    )

    # --- 4. Filter upward-going neutrinos ---
    going_up = nu_dirs[:, 2] > 0
    upward_indices = np.where(going_up)[0]

    photon_counts = np.zeros((N_det, N_samples))
    interaction_weights = np.zeros(N_samples)
    interaction_altitudes = np.zeros(N_samples)
    int_pos_all = np.zeros((N_samples, 3))

    if len(upward_indices) == 0:
        return photon_counts, interaction_weights, N_nu_per_muon, 1.0, int_pos_all

    # --- 5. Compute P_CC for each upward neutrino ---
    # Use max detector height for the atmospheric column
    # consider an approximate column depth to the satellite for all events, since most interactions will be near the balloon
    column_depth = atmospheric_column_depth_nucleons(0, max_det_height,flux_geometry.cos_surface_exit_angle)
    for idx in upward_indices:
        #cos_z = nu_dirs[idx, 2]
        #column = atmospheric_column_depth_nucleons(0, max_det_height, cos_z)
        interaction_weights[idx] = sigma_CC_nu(nu_energy[idx]) * column_depth

    # --- 6. Sample interaction altitudes and compute Cherenkov ---
    N_valid = len(upward_indices)
    if max_cherenkov_events is not None and N_valid > max_cherenkov_events:
        eval_indices = np.random.choice(upward_indices, max_cherenkov_events, replace=False)
        cherenkov_weight = N_valid / max_cherenkov_events
    else:
        eval_indices = upward_indices
        cherenkov_weight = 1.0

    # Pre-sample all interaction altitudes at once
    z_int_all, pos_weights_eval = sample_interaction_altitude(len(eval_indices), z_max_m=max_det_height, uniform_gen=uniform_gen, direction_cosine=flux_geometry.cos_surface_exit_angle)
    # Map position weights back to full N_samples array (ones for non-evaluated)
    position_weights = np.ones(N_samples)
    for i_eval, idx in enumerate(eval_indices):
        position_weights[idx] = pos_weights_eval[i_eval]

    for i_eval, idx in enumerate(eval_indices):
        z_int = z_int_all[i_eval]
        interaction_altitudes[idx] = z_int

        # Compute 3D interaction position along the neutrino trajectory
        # Neutrino starts at prod_points[idx] (z < 0) and travels along nu_dirs[idx]
        # Time to reach z = z_int:
        t_to_z = (z_int - prod_points[idx, 2]) / nu_dirs[idx, 2]
        int_pos = prod_points[idx] + t_to_z * nu_dirs[idx]
        int_pos_all[idx] = int_pos

        # --- Outgoing muon from CC interaction ---
        # Inelasticity y: dsigma/dy ~ 1 + (1-y)^2 for neutrinos
        while True:
            y = np.random.uniform(0, 1)
            if np.random.uniform(0, 2) < 1 + (1 - y)**2:
                break

        E_mu_out = (1 - y) * nu_energy[idx]
        if E_mu_out < 0.1:
            continue

        # Muon direction: nearly collinear with neutrino at high energy
        # Characteristic scattering angle ~ m_mu / E_mu_out
        theta_scat = min(m_mu / E_mu_out, 0.3)
        phi_scat = np.random.uniform(0, 2 * np.pi)

        # Rotate neutrino direction by small angle
        nu_hat = nu_dirs[idx]
        if abs(nu_hat[2]) < 0.9:
            perp1 = np.cross(nu_hat, [0, 0, 1])
        else:
            perp1 = np.cross(nu_hat, [1, 0, 0])
        perp1 = perp1 / np.linalg.norm(perp1)
        perp2 = np.cross(nu_hat, perp1)

        mu_hat = (np.cos(theta_scat) * nu_hat
                  + np.sin(theta_scat) * (np.cos(phi_scat) * perp1
                                          + np.sin(phi_scat) * perp2))
        mu_hat = mu_hat / np.linalg.norm(mu_hat)

        if mu_hat[2] <= 0:
            continue

        # --- Cherenkov from outgoing muon at each detector ---
        dir_cosine = max(mu_hat[2], 0.1)
        track_length = muon_range_in_air(E_mu_out, z_int, direction_cosine=dir_cosine)
        if track_length <= 0:
            continue

        # Atmospheric transmission (same for all detectors)
        zenith_angle = np.arccos(dir_cosine)
        transmission = cherenkov_transmission(z_int, zenith_angle)

        N_track = min(1000, max(300, int(track_length / 100)))

        # Filter detectors above the interaction point
        valid_dets = [i for i, dp in enumerate(detector_positions)
                      if z_int < dp[2]]
        if not valid_dets:
            continue

        # --- Muon Cherenkov (computed once for all detectors) ---
        valid_det_pos = [detector_positions[i] for i in valid_dets]
        try:
            N_ph_mu_all = cherenkov_photons_multi_detector(
                int_pos, mu_hat, track_length, R_det,
                valid_det_pos, N_psi=300, N_track=N_track
            )
            N_ph_mu_all *= transmission
        except Exception:
            N_ph_mu_all = np.zeros(len(valid_dets))

        # --- Hadronic shower + combine ---
        for j, i_det in enumerate(valid_dets):
            N_ph_had = 0.0
            if include_hadronic_shower:
                E_had = y * nu_energy[idx]
                if E_had > 1.0:
                    r_rel = int_pos - detector_positions[i_det]
                    N_ph_had = hadronic_shower_cherenkov(
                        E_had, nu_dirs[idx], r_rel, z_int)

            photon_counts[i_det, idx] = N_ph_mu_all[j] + N_ph_had

    return (photon_counts, interaction_weights*position_weights, N_nu_per_muon,
            cherenkov_weight, int_pos_all)


def summarize_background(photon_counts, interaction_weights, N_nu_per_muon,
                         N_samples, min_photons=10, cherenkov_weight=1.0):
    """
    Compute expected number of background events from MC output.

    N_bg = N_ν_per_muon × N_muon_total × <P_CC × 1(N_ph ≥ threshold)>_MC

    Parameters
    ----------
    photon_counts : ndarray
        Cherenkov photons per MC event. Shape (N_samples,) or (N_det, N_samples).
    interaction_weights : ndarray, shape (N_samples,)
        P_CC weight per event
    N_nu_per_muon : float
        Neutrino production rate per muon
    N_samples : int
        Total MC samples
    min_photons : float
        Detection threshold
    cherenkov_weight : float
        Reweight factor from Cherenkov cap

    Returns
    -------
    N_background : float or array
        Expected background events. Array of length N_det if multi-detector.
    detection_info : dict or list of dict
        Diagnostic information. List if multi-detector.
    """
    photon_counts = np.asarray(photon_counts)

    # Handle multi-detector case: recurse over detectors
    if photon_counts.ndim == 2:
        results = [summarize_background(photon_counts[i], interaction_weights,
                                        N_nu_per_muon, N_samples, min_photons,
                                        cherenkov_weight)
                   for i in range(photon_counts.shape[0])]
        return (np.array([r[0] for r in results]),
                [r[1] for r in results])

    # Single-detector case
    detected = photon_counts >= min_photons

    # Weighted detection: sum P_CC over detected events, average over all
    weighted_sum = np.sum(interaction_weights[detected]) * cherenkov_weight
    avg_P_CC_times_detect = weighted_sum / N_samples

    N_background = N_nu_per_muon * N_muon_decays * avg_P_CC_times_detect

    # Diagnostics
    if np.any(detected):
        mean_photons = np.average(
            photon_counts[detected],
            weights=interaction_weights[detected]
        )
    else:
        mean_photons = 0.0

    detection_info = {
        'N_background': N_background,
        'N_nu_per_muon': N_nu_per_muon,
        'avg_P_CC': np.mean(interaction_weights),
        'avg_P_CC_times_detect': avg_P_CC_times_detect,
        'n_detected_unweighted': int(np.sum(detected)),
        'mean_photons_detected': mean_photons,
    }

    return N_background, detection_info


def background_altitude_distribution(photon_counts, interaction_weights,
                                     interaction_altitudes, min_photons=10,
                                     z_bins=None):
    """
    Compute the altitude distribution of detectable background events.

    Useful for comparing signal vs background z-profiles with two balloons.

    Parameters
    ----------
    photon_counts, interaction_weights, interaction_altitudes : ndarray
        Output from compute_background_at_satellite
    min_photons : float
        Detection threshold
    z_bins : ndarray or None
        Altitude bin edges [m]. If None, uses 50 bins from 0 to 50 km.

    Returns
    -------
    z_centers : ndarray
        Bin centers [m]
    weighted_hist : ndarray
        P_CC-weighted count of detected events per bin (unnormalized)
    """
    if z_bins is None:
        z_bins = np.linspace(0, 50000, 51)

    detected = photon_counts >= min_photons
    z_det = interaction_altitudes[detected]
    w_det = interaction_weights[detected]

    weighted_hist, _ = np.histogram(z_det, bins=z_bins, weights=w_det)
    z_centers = 0.5 * (z_bins[:-1] + z_bins[1:])

    return z_centers, weighted_hist
