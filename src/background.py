# Beam-induced neutrino background for balloon/satellite HNL detection
#
# The dominant background is NOT atmospheric neutrinos but beam-induced
# neutrinos: the muon beam produces neutrinos (from muon decay and DIS
# in the target) that travel upward alongside the HNLs. These neutrinos
# interact in the atmosphere via CC, producing charged leptons that emit
# Cherenkov light mimicking the signal.

import numpy as np
from scipy.integrate import quad

from src.constants import *
from src.flux import numu_flux_baseline, nue_flux_baseline
from src.cherenkov import cherenkov_photons_detected_vectorized, cherenkov_transmission
from src.balloon import (air_density, muon_range_in_air,
                         electron_shower_length_in_air, particle_track_length)


# Physical constants for neutrino interactions
SIGMA_CC_PER_NUCLEON_PER_GEV = 0.7e-38  # cm^2 / nucleon / GeV (linear in E_nu)
N_AVOGADRO = 6.022e23  # nucleons per mol
A_AIR = 14.5  # effective atomic mass of air (N2/O2 mix)


def sigma_CC(E_nu_GeV):
    """
    Total CC neutrino-nucleon cross section.

    sigma_CC(E_nu) ~ 0.7e-38 * E_nu [cm^2/nucleon]

    Parameters
    ----------
    E_nu_GeV : float or array
        Neutrino energy [GeV]

    Returns
    -------
    sigma : float or array
        Cross section [cm^2 per nucleon]
    """
    return SIGMA_CC_PER_NUCLEON_PER_GEV * E_nu_GeV


def air_nucleon_density(altitude_m):
    """
    Number density of nucleons in air at given altitude [nucleons/cm^3].

    Parameters
    ----------
    altitude_m : float or array
        Altitude [m]

    Returns
    -------
    n : float or array
        Nucleon number density [cm^-3]
    """
    rho = air_density(altitude_m)  # kg/m^3
    rho_gcm3 = rho * 1e-3  # g/cm^3
    return N_AVOGADRO * rho_gcm3 / A_AIR  # nucleons/cm^3


def beam_neutrino_flux_spectrum(E_nu_arr, E_mu_beam, baseline_m,
                                Pmuon=-1.0, cos_theta=1.0):
    """
    Compute the beam neutrino flux (nu_mu + nu_e) as a function of energy.

    Uses the muon-decay neutrino flux from src/flux.py.
    The neutrinos travel along the beam axis (upward toward satellite).

    Parameters
    ----------
    E_nu_arr : array
        Neutrino energies [GeV]
    E_mu_beam : float
        Muon beam energy [GeV]
    baseline_m : float
        Distance from source to observation point [m]
    Pmuon : float
        Muon polarization (-1 for mu-, +1 for mu+)
    cos_theta : float
        Cosine of angle between neutrino direction and beam axis

    Returns
    -------
    flux_numu : array
        nu_mu flux [neutrinos / GeV / m^2] at each energy
    flux_nue : array
        nu_e flux [neutrinos / GeV / m^2] at each energy
    """
    flux_numu = np.zeros_like(E_nu_arr)
    flux_nue = np.zeros_like(E_nu_arr)

    for i, enu in enumerate(E_nu_arr):
        try:
            flux_numu[i] = numu_flux_baseline(E_mu_beam, Pmuon, enu, cos_theta, baseline_m)
        except Exception:
            flux_numu[i] = 0.0
        try:
            flux_nue[i] = nue_flux_baseline(E_mu_beam, Pmuon, enu, cos_theta, baseline_m)
        except Exception:
            flux_nue[i] = 0.0

    # flux.py returns flux in units of [neutrinos / GeV / m^2]
    return flux_numu, flux_nue


def compute_background_rate(E_mu_beam, satellite_height_m, target_depth_m,
                            A_eff_m2=None, R_det_m=None,
                            N_E_bins=50, N_z_bins=100,
                            E_nu_min=1.0, E_nu_max=None,
                            Pmuon=-1.0):
    """
    Compute the expected number of beam-induced neutrino background events.

    For each altitude bin, integrates the neutrino interaction rate:
        dN/dz = integral Phi_nu(E) * sigma_CC(E) * n_air(z) * A_eff * dE

    The outgoing charged lepton produces Cherenkov light detectable by
    the balloon/satellite.

    Parameters
    ----------
    E_mu_beam : float
        Muon beam energy [GeV]
    satellite_height_m : float
        Satellite/balloon altitude [m]
    target_depth_m : float
        Depth of production target below surface [m]
    A_eff_m2 : float or None
        Effective area for neutrino interactions [m^2]. If None, computed from
        the angular spread of the beam neutrino flux and the altitude.
    R_det_m : float or None
        Detector radius [m]. Used for Cherenkov calculation. Defaults to R_det.
    N_E_bins : int
        Number of energy bins for neutrino spectrum
    N_z_bins : int
        Number of altitude bins
    E_nu_min : float
        Minimum neutrino energy [GeV]
    E_nu_max : float or None
        Maximum neutrino energy [GeV]. Defaults to E_mu_beam.
    Pmuon : float
        Muon polarization

    Returns
    -------
    N_background : float
        Expected number of background events (above Cherenkov threshold)
    background_info : dict
        Diagnostic information including altitude distribution, etc.
    """
    if R_det_m is None:
        R_det_m = R_det
    if E_nu_max is None:
        E_nu_max = E_mu_beam

    # The baseline for the neutrino flux: distance from muon decay
    # region to the atmosphere. The muons decay throughout the accelerator
    # straight section; the neutrinos then travel to the atmosphere.
    # For the balloon geometry, the baseline is approximately
    # target_depth + altitude.
    # But the flux functions from flux.py already handle the baseline.

    # Energy bins
    E_nu_arr = np.logspace(np.log10(max(E_nu_min, 0.1)), np.log10(E_nu_max), N_E_bins)
    dE = np.diff(E_nu_arr)
    E_centers = 0.5 * (E_nu_arr[:-1] + E_nu_arr[1:])

    # Altitude bins (from sea level to satellite height)
    z_arr = np.linspace(0, satellite_height_m, N_z_bins + 1)
    z_centers = 0.5 * (z_arr[:-1] + z_arr[1:])
    dz = z_arr[1] - z_arr[0]  # m

    # Effective area for neutrino interactions
    # The beam neutrinos are highly forward-peaked (angle ~ 1/gamma_mu).
    # At altitude z, the beam cone has radius ~ z * theta_beam.
    gamma_mu = E_mu_beam / m_mu
    theta_beam = 1.0 / gamma_mu  # characteristic opening angle

    if A_eff_m2 is None:
        # Use a characteristic area: the beam footprint at mid-atmosphere,
        # clipped to a reasonable size
        z_mid = satellite_height_m / 2
        r_beam = max(z_mid * theta_beam, 10.0)  # at least 10 m
        A_eff_m2 = np.pi * r_beam**2

    # Compute interaction rates per altitude bin
    interaction_rate_per_z = np.zeros(N_z_bins)
    avg_E_lepton_per_z = np.zeros(N_z_bins)

    for iz, z in enumerate(z_centers):
        # Baseline from source to this altitude
        baseline = target_depth_m + z  # m

        # Get flux at this baseline
        flux_numu, flux_nue = beam_neutrino_flux_spectrum(
            E_centers, E_mu_beam, baseline, Pmuon=Pmuon
        )

        # Total flux (nu_mu + nu_e)
        total_flux = flux_numu + flux_nue  # per GeV per m^2

        # Air nucleon density at this altitude
        n_air = air_nucleon_density(z)  # cm^-3

        # Interaction rate per unit volume per unit area
        # dN/(dz * dA) = integral Phi(E) * sigma_CC(E) * n_air(z) * dE
        # Units: [1/GeV/m^2] * [cm^2] * [1/cm^3] * [GeV] = [1/cm^3/m^2 * cm^2]
        # Need to be careful with units.

        # sigma_CC in cm^2, n_air in cm^-3, flux in m^-2 GeV^-1
        # Rate density = sum(flux * sigma * n_air * dE) * dz_cm * A_eff_m2
        # where dz is in cm for consistency with cm^2 and cm^-3

        dz_cm = dz * 100  # convert m to cm

        integrand = total_flux * sigma_CC(E_centers) * n_air  # per GeV per m^2 per cm
        rate = np.sum(integrand * dE) * dz_cm * A_eff_m2

        interaction_rate_per_z[iz] = rate

        # Average outgoing lepton energy (weighted by rate)
        weighted_E = np.sum(integrand * dE * E_centers * 0.75)  # <y> ~ 0.25, so E_lepton ~ 0.75 * E_nu
        if rate > 0:
            avg_E_lepton_per_z[iz] = weighted_E * dz_cm * A_eff_m2 / rate
        else:
            avg_E_lepton_per_z[iz] = 0.0

    total_interactions = np.sum(interaction_rate_per_z)

    background_info = {
        'z_centers': z_centers,
        'interaction_rate_per_z': interaction_rate_per_z,
        'avg_E_lepton_per_z': avg_E_lepton_per_z,
        'total_interactions': total_interactions,
        'A_eff_m2': A_eff_m2,
        'N_muon_decays': N_muon_decays,
    }

    # The total background is the number of interactions that produce
    # detectable Cherenkov light. For a rough estimate, multiply by N_muon_decays.
    N_background = total_interactions * N_muon_decays

    return N_background, background_info


def sample_background_events(E_mu_beam, satellite_height_m, target_depth_m,
                             N_samples=10000, R_det_m=None,
                             E_nu_min=1.0, E_nu_max=None,
                             Pmuon=-1.0, min_photons=10):
    """
    Monte Carlo sampling of beam-induced neutrino background events.

    For each sampled event:
    1. Draw neutrino energy from the beam flux spectrum
    2. Draw interaction altitude from air density profile
    3. Compute outgoing lepton energy and direction
    4. Compute Cherenkov photons at the satellite
    5. Apply photon threshold

    Parameters
    ----------
    E_mu_beam : float
        Muon beam energy [GeV]
    satellite_height_m : float
        Satellite/balloon altitude [m]
    target_depth_m : float
        Depth of production target [m]
    N_samples : int
        Number of MC events to sample
    R_det_m : float or None
        Detector radius [m]
    E_nu_min : float
        Minimum neutrino energy [GeV]
    E_nu_max : float or None
        Maximum neutrino energy [GeV]
    Pmuon : float
        Muon polarization
    min_photons : float
        Minimum photons for detection

    Returns
    -------
    photon_counts : ndarray
        Cherenkov photon counts per sampled event
    event_info : dict
        Diagnostic info (energies, altitudes, etc.)
    """
    if R_det_m is None:
        R_det_m = R_det
    if E_nu_max is None:
        E_nu_max = E_mu_beam

    # Build neutrino energy spectrum (probability distribution)
    N_E = 200
    E_nu_arr = np.logspace(np.log10(max(E_nu_min, 0.1)), np.log10(E_nu_max), N_E)

    # Use a mid-range baseline to get the spectrum shape
    baseline_mid = target_depth_m + satellite_height_m / 2
    flux_numu, flux_nue = beam_neutrino_flux_spectrum(
        E_nu_arr, E_mu_beam, baseline_mid, Pmuon=Pmuon
    )
    total_flux = flux_numu + flux_nue

    # Weight by cross section to get interaction spectrum
    interaction_spectrum = total_flux * sigma_CC(E_nu_arr) * E_nu_arr  # E weighting for log spacing
    if np.sum(interaction_spectrum) <= 0:
        return np.zeros(N_samples), {'detected': 0}
    interaction_spectrum = interaction_spectrum / np.sum(interaction_spectrum)

    # Build altitude distribution (weighted by air density * cross section)
    N_z = 200
    z_arr = np.linspace(100, satellite_height_m * 0.99, N_z)  # avoid exact satellite height
    density_profile = np.array([air_nucleon_density(z) for z in z_arr])
    if np.sum(density_profile) <= 0:
        return np.zeros(N_samples), {'detected': 0}
    density_weights = density_profile / np.sum(density_profile)

    # Sample events
    E_nu_samples = np.random.choice(E_nu_arr, size=N_samples, p=interaction_spectrum)
    z_samples = np.random.choice(z_arr, size=N_samples, p=density_weights)

    # Outgoing lepton properties
    # For nu_mu CC: mu carries (1-y)*E_nu, where y (inelasticity) averages ~0.25
    y_samples = np.random.uniform(0, 1, N_samples)  # flat in y for simplicity
    # Weight by (1-y) for the muon spectrum (leading-order DIS)
    # For a more accurate treatment: d(sigma)/dy ~ 1 + (1-y)^2 for nu, ~ (1-y)^2 for nubar
    # Use average: E_lepton ~ 0.75 * E_nu
    E_lepton = (1 - y_samples * 0.5) * E_nu_samples  # rough
    E_lepton = np.maximum(E_lepton, 0.1)

    # Lepton direction: nearly collinear with neutrino at high energy
    # Angular spread ~ m_lepton / E_lepton + sqrt(Q^2) / E_nu
    # For simplicity, assume collinear (upward along beam axis)
    # Small angular smearing: theta ~ 1/sqrt(E_nu) [rad]
    theta_scatter = 1.0 / np.sqrt(E_lepton + 1)  # characteristic scattering angle
    phi_scatter = np.random.uniform(0, 2 * np.pi, N_samples)

    # Lepton direction (nearly vertical, upward)
    lepton_dirs = np.column_stack([
        theta_scatter * np.cos(phi_scatter),
        theta_scatter * np.sin(phi_scatter),
        np.ones(N_samples)  # z component ~ 1
    ])
    # Normalize
    norms = np.linalg.norm(lepton_dirs, axis=1, keepdims=True)
    lepton_dirs = lepton_dirs / norms

    # Satellite position
    sat_pos = np.array([0, 0, satellite_height_m])

    # Compute Cherenkov photons for each event
    photon_counts = np.zeros(N_samples)

    for i in range(N_samples):
        altitude = z_samples[i]
        E_l = E_lepton[i]
        p_hat = lepton_dirs[i]

        # Position of interaction
        interaction_pos = np.array([0, 0, altitude])
        r_rel = interaction_pos - sat_pos

        # Determine particle type: assume nu_mu CC -> muon (dominant)
        # (nu_e CC -> electron would give EM shower)
        if E_nu_samples[i] > 1.0:
            # Most interactions are nu_mu CC at high energy
            track_length = muon_range_in_air(
                E_l, altitude, direction_cosine=max(p_hat[2], 0.1)
            )
        else:
            # Low energy: treat as electron-like shower
            track_length = electron_shower_length_in_air(E_l, altitude)

        if track_length <= 0:
            continue

        N_track = min(500, max(100, int(track_length / 100)))
        try:
            N_ph, _, _ = cherenkov_photons_detected_vectorized(
                r_rel, p_hat, track_length, R_det_m, N_psi=200, N_track=N_track
            )
        except Exception:
            N_ph = 0

        # Apply atmospheric transmission
        zenith_angle = np.arccos(max(p_hat[2], 0.1))
        transmission = cherenkov_transmission(altitude, zenith_angle)
        N_ph *= transmission

        photon_counts[i] = N_ph

    detected = photon_counts >= min_photons
    event_info = {
        'E_nu_samples': E_nu_samples,
        'z_samples': z_samples,
        'E_lepton': E_lepton,
        'photon_counts': photon_counts,
        'detected_mask': detected,
        'N_detected': np.sum(detected),
        'detection_efficiency': np.sum(detected) / N_samples if N_samples > 0 else 0,
    }

    return photon_counts, event_info


def estimate_background_events(E_mu_beam, satellite_height_m, target_depth_m,
                               R_det_m=None, N_mc=5000, min_photons=10,
                               Pmuon=-1.0):
    """
    Estimate total background events combining analytic rate with MC efficiency.

    1. Compute total neutrino interaction rate (analytic)
    2. Estimate Cherenkov detection efficiency via MC sampling
    3. Multiply: N_background = rate * efficiency * N_muon_decays

    Parameters
    ----------
    E_mu_beam : float
        Muon beam energy [GeV]
    satellite_height_m : float
        Satellite/balloon altitude [m]
    target_depth_m : float
        Target depth [m]
    R_det_m : float or None
        Detector radius [m]
    N_mc : int
        Number of MC samples for efficiency estimation
    min_photons : float
        Photon threshold for detection
    Pmuon : float
        Muon polarization

    Returns
    -------
    N_background_total : float
        Expected total background events
    info : dict
        Detailed breakdown
    """
    if R_det_m is None:
        R_det_m = R_det

    # Step 1: Analytic interaction rate
    N_bg_analytic, bg_info = compute_background_rate(
        E_mu_beam, satellite_height_m, target_depth_m,
        R_det_m=R_det_m, Pmuon=Pmuon
    )

    # Step 2: MC-based detection efficiency
    photon_counts, mc_info = sample_background_events(
        E_mu_beam, satellite_height_m, target_depth_m,
        N_samples=N_mc, R_det_m=R_det_m, min_photons=min_photons,
        Pmuon=Pmuon
    )

    detection_eff = mc_info['detection_efficiency']

    # Step 3: Total expected background
    N_background_total = N_bg_analytic * detection_eff

    info = {
        'N_interactions_per_muon_decay': bg_info['total_interactions'],
        'N_interactions_total': N_bg_analytic,
        'detection_efficiency': detection_eff,
        'N_background_total': N_background_total,
        'analytic_info': bg_info,
        'mc_info': mc_info,
    }

    return N_background_total, info
