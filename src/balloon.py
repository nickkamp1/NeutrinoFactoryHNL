# HNL Flux Geometry Model - Upward Beam Configuration
import numpy as np

from src.cherenkov import *
from src.constants import *
from src.xs_and_decays import *

# Realistic muon track length calculation
# Muons lose energy via ionization in air; track length depends on energy and air density

def air_density(altitude_m):
    """
    Approximate air density as function of altitude using exponential atmosphere.

    Parameters
    ----------
    altitude_m : float
        Altitude above sea level [m]

    Returns
    -------
    rho : float
        Air density [kg/m^3]
    """
    rho_0 = 1.225  # sea level density [kg/m^3]
    H = 8500  # scale height [m]
    return rho_0 * np.exp(-altitude_m / H)


def muon_range_in_air(E_mu_GeV, start_altitude_m, direction_cosine=1.0):
    """
    Calculate muon range in air accounting for varying density with altitude.

    Uses dE/dx ≈ 2 MeV/(g/cm^2) for minimum ionizing muons.
    Integrates through atmosphere with exponential density profile.
    Also accounts for muon decays

    Parameters
    ----------
    E_mu_GeV : float
        Initial muon energy [GeV]
    start_altitude_m : float
        Starting altitude [m]
    direction_cosine : float
        cos(theta) where theta is angle from vertical (1 = straight up)

    Returns
    -------
    range_m : float
        Muon range [m]
    """

    # muon lifetime
    tau_mu = 2.2e3 # nanoseconds
    L_mu = E_mu_GeV/ m_mu * speed_of_light * tau_mu


    # Energy loss rate: dE/dx = 2 MeV/(g/cm^2) = 2e-3 GeV/(g/cm^2)
    dEdX_mass = 2e-3  # GeV / (g/cm^2)

    # For an exponential atmosphere, we can integrate analytically
    # dE/dz = dE/dx * rho(z) where rho(z) = rho_0 * exp(-z/H)
    #
    # For a muon going straight up: integral of rho dz from z0 to inf = rho_0 * H * exp(-z0/H)
    # This gives the "column depth" in g/cm^2

    rho_0 = 1.225e-3  # sea level density [g/cm^3]
    H = 8500 * 100  # scale height [cm]

    # Column depth above start altitude [g/cm^2]
    column_depth = rho_0 * H * np.exp(-start_altitude_m * 100 / H)  # g/cm^2

    # Correct for non-vertical trajectory
    column_depth = column_depth / direction_cosine

    # Energy lost traversing this column
    E_lost = dEdX_mass * column_depth

    # If muon has enough energy to escape atmosphere
    if E_mu_GeV > E_lost:
        # Range is entire atmosphere (for our purposes, cap at balloon altitude)
        # Actual geometric distance
        range_m = (L_det - start_altitude_m) / direction_cosine
    else:
        # Muon stops in atmosphere
        # Approximate: range ≈ E / (dE/dx * avg_density)
        avg_density = air_density(start_altitude_m + 5000)  # rough average
        avg_density_gcm3 = avg_density * 1e-3  # convert kg/m^3 to g/cm^3
        range_cm = E_mu_GeV / (dEdX_mass * avg_density_gcm3)
        range_m = range_cm / 100

    return min(range_m, L_mu)  # cap at muon decay length


def muon_energy_from_hnl_decay(m_N, E_N):
    """
    Estimate muon energy from HNL decay N -> mu + X.

    For 2-body decay in HNL rest frame, muon gets ~half the mass.
    Then boost to lab frame.

    Parameters
    ----------
    m_N : float
        HNL mass [GeV]
    E_N : float
        HNL energy in lab frame [GeV]

    Returns
    -------
    E_mu_mean : float
        Mean muon energy [GeV]
    E_mu_std : float
        Spread in muon energy [GeV]
    """
    gamma_N = E_N / m_N
    beta_N = np.sqrt(1 - 1/gamma_N**2)

    # In rest frame, for N -> mu + X (where X could be W* -> hadrons or leptons)
    # Muon energy in rest frame: E_mu_rest ~ m_N/2 (rough, depends on X mass)
    E_mu_rest = m_N / 2
    p_mu_rest = np.sqrt(max(0, E_mu_rest**2 - m_mu**2))

    # Boost to lab frame
    # E_lab = gamma * (E_rest + beta * p_rest * cos(theta_rest))
    # For isotropic decay, average cos(theta) = 0, so E_lab_mean = gamma * E_rest
    E_mu_mean = gamma_N * E_mu_rest

    # Spread from boost: E ranges from gamma*(E-beta*p) to gamma*(E+beta*p)
    E_mu_min = gamma_N * (E_mu_rest - beta_N * p_mu_rest)
    E_mu_max = gamma_N * (E_mu_rest + beta_N * p_mu_rest)
    E_mu_std = (E_mu_max - E_mu_min) / 4  # rough estimate

    return E_mu_mean, E_mu_std

class HNLFluxGeometry:
    """
    Model for HNL production from a muon beam emerging from the Earth.

    Setup:
    - Muon beam travels UPWARD through Earth toward the satellite
    - HNLs are produced when muons scatter in Earth (below surface)
    - HNLs travel upward and decay in atmosphere
    - Satellite/balloon detects Cherenkov light from decay products

    Coordinate system:
    - Origin at Earth surface, directly below the satellite
    - z-axis points up (away from Earth)
    - x-axis in the horizontal plane, along beam offset direction
    - Beam emerges from underground, traveling upward

    Parameters
    ----------
    E_mu : float
        Muon beam energy [GeV]
    beam_offset_angle : float
        Angle of beam from vertical [rad]. 0 = straight up toward satellite.
        Small positive values = beam tilted slightly away from satellite.
    target_depth : float
        Depth below surface where target/production region is centered [m]
    satellite_height : float
        Height of satellite/balloon above Earth surface [m]
    L_target : float
        Effective target length along beam [m]
    """

    def __init__(self, E_mu=1500, beam_offset_angle=0.0, target_depth=100,
                 satellite_height=33000, L_target=500):
        self.E_mu = E_mu
        self.beam_offset_angle = beam_offset_angle  # angle from vertical (upward)
        self.target_depth = target_depth
        self.satellite_height = satellite_height
        self.L_target = target_depth*2

        # Muon beam direction (unit vector) - going OUT OF Earth (upward)
        # beam_offset_angle = 0: straight up toward satellite
        # beam_offset_angle > 0: tilted in +x direction
        self.beam_dir = np.array([
            np.sin(beam_offset_angle),   # x component (horizontal offset)
            0,                            # y component
            np.cos(beam_offset_angle)     # z component (positive = upward)
        ])

        # Satellite position (directly above origin)
        self.satellite_pos = np.array([0, 0, satellite_height])

    def sample_production_points(self, N_samples):
        """
        Sample HNL production points along the muon path in Earth.

        Production occurs underground along the beam path.
        """
        # Distance along beam from the target center
        s = np.random.uniform(-self.L_target/2, self.L_target/2, N_samples)

        # Target center is at depth target_depth, on the beam axis
        # The beam passes through (0, 0, -target_depth) if beam_offset_angle = 0
        # For non-zero offset, we trace back from surface along beam direction

        # Point where beam crosses z = 0 (surface)
        # If beam comes from depth d at angle theta, it crosses surface at:
        #   x_surface = d * tan(theta)
        # We define the target center at the specified depth along the beam

        target_center = np.array([
            -self.target_depth * np.tan(self.beam_offset_angle),  # x offset
            0,                                                      # y = 0
            -self.target_depth                                      # z = -depth
        ])

        production_points = target_center + np.outer(s, self.beam_dir)
        return production_points

    def sample_hnl_directions(self, m_N, N_samples):
        """
        Sample HNL emission directions from production.

        HNLs are produced with angular spread around the beam direction.
        The distribution is forward-peaked with characteristic angle ~ m_N/E_mu.
        """
        # Characteristic production angle
        theta_char = m_N / self.E_mu

        # Sample polar angles relative to beam direction
        # Use exponential distribution for forward-peaked behavior
        theta_rel = np.random.exponential(theta_char, N_samples)

        # Azimuthal angle is uniform
        phi_rel = np.random.uniform(0, 2*np.pi, N_samples)

        # Convert to direction vectors in beam-aligned frame
        dx_beam = np.sin(theta_rel) * np.cos(phi_rel)
        dy_beam = np.sin(theta_rel) * np.sin(phi_rel)
        dz_beam = np.cos(theta_rel)

        # Build orthonormal basis with beam_dir as z-axis
        z_hat = self.beam_dir

        # Find perpendicular vectors
        if abs(z_hat[2]) < 0.9:
            x_hat = np.cross(z_hat, np.array([0, 0, 1]))
        else:
            x_hat = np.cross(z_hat, np.array([1, 0, 0]))
        x_hat = x_hat / np.linalg.norm(x_hat)
        y_hat = np.cross(z_hat, x_hat)

        # Transform to lab frame
        directions = np.zeros((N_samples, 3))
        directions[:, 0] = dx_beam * x_hat[0] + dy_beam * y_hat[0] + dz_beam * z_hat[0]
        directions[:, 1] = dx_beam * x_hat[1] + dy_beam * y_hat[1] + dz_beam * z_hat[1]
        directions[:, 2] = dx_beam * x_hat[2] + dy_beam * y_hat[2] + dz_beam * z_hat[2]

        # Normalize
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        directions = directions / norms

        return directions

    def sample_decay_points(self, production_points, hnl_directions, decay_length):
        """
        Sample HNL decay points given production points, directions, and decay length.
        """
        N = len(production_points)
        travel_distances = np.random.exponential(decay_length, N)
        decay_points = production_points + travel_distances[:, np.newaxis] * hnl_directions
        return decay_points, travel_distances

    def get_satellite_position(self):
        """Return the satellite position."""
        return self.satellite_pos.copy()


def compute_signal_at_satellite(m_N, E_mu, U2, flux_geometry,
                                N_samples=1000, min_photons=10):
    """
    Compute expected signal at a satellite position from HNL decays.
    """
    sat_position = flux_geometry.get_satellite_position()

    HNL_xs = sigma(E_mu, m_N, U2)
    E_N = E_mu / 2  # Approximate HNL energy
    N_HNLs_per_muon = HNL_xs * flux_geometry.L_target   * n_earth_m3  # number of HNLs produced per muon

    # HNL decay length
    decay_length = HNL_decay_length(m_N, U2, E_N)

    # Sample production points
    prod_points = flux_geometry.sample_production_points(N_samples)

    # Sample HNL directions
    hnl_dirs = flux_geometry.sample_hnl_directions(m_N, N_samples)

    # Sample decay points
    decay_points, travel_dist = flux_geometry.sample_decay_points(
        prod_points, hnl_dirs, decay_length
    )

    # Filter: decays in atmosphere (z > 0 and z < satellite altitude)
    above_surface = decay_points[:, 2] > 0
    below_satellite = decay_points[:, 2] < sat_position[2]
    valid_decay = above_surface & below_satellite

    if not np.any(valid_decay):
        return 0.0, 0.0

    # For valid decays, compute Cherenkov photons
    photon_counts = []
    valid_indices = np.where(valid_decay)[0]

    for idx in valid_indices:
        decay_pos = decay_points[idx]
        hnl_dir = hnl_dirs[idx]

        # Position relative to satellite
        r_rel = decay_pos - sat_position

        # Muon direction from HNL decay (boosted, so roughly along HNL direction)
        gamma_N = E_N / m_N
        theta_decay_spread = np.random.exponential(1/gamma_N)
        phi_decay = np.random.uniform(0, 2*np.pi)

        # Muon direction ≈ HNL direction with small spread
        mu_dir = hnl_dir.copy()
        # Add small perpendicular perturbation
        perp = np.array([np.cos(phi_decay), np.sin(phi_decay), 0])
        perp = perp - np.dot(perp, hnl_dir) * hnl_dir  # make perpendicular
        if np.linalg.norm(perp) > 0:
            perp = perp / np.linalg.norm(perp)
            mu_dir = mu_dir + theta_decay_spread * perp
            mu_dir = mu_dir / np.linalg.norm(mu_dir)

        # Only count if muon is going upward (can produce detectable Cherenkov)
        if mu_dir[2] <= 0:
            photon_counts.append(0)
            continue

        r_0 = r_rel
        p_hat = mu_dir

        # Muon track length
        decay_altitude = max(0, decay_pos[2])
        E_mu_daughter, _ = muon_energy_from_hnl_decay(m_N, E_N)
        track_length = muon_range_in_air(E_mu_daughter, decay_altitude,
                                          direction_cosine=max(p_hat[2], 0.1))

        # Compute photons
        N_track = max(1000, min(300, int(track_length / 100)))
        try:
            N_ph,_,_ = cherenkov_photons_detected_vectorized(
                r_0, p_hat, track_length, R_det, N_psi=30, N_track=N_track
            )
        except:
            N_ph = 0

        photon_counts.append(N_ph)

    photon_counts = np.array(photon_counts)

    # Detection efficiency
    detected = photon_counts >= min_photons
    detection_efficiency = np.sum(detected) / N_samples

    # Mean photons for detected events
    mean_photons = np.mean(photon_counts[detected]) if np.any(detected) else 0.0

    number_of_events = N_HNLs_per_muon * detection_efficiency * N_muon_decays

    return detection_efficiency, mean_photons, number_of_events