# HNL Flux Geometry Model - Upward Beam Configuration
import numpy as np
import pandas as pd

from src.cherenkov import *
from src.constants import *
from src.xs_and_decays import *

# Realistic muon track length calculation
# Muons lose energy via ionization in air; track length depends on energy and air density

# Track length constants for different particle types in air
PION_INTERACTION_LENGTH_GCM2 = 120.0   # nuclear interaction length in air [g/cm²]
RADIATION_LENGTH_AIR_GCM2 = 36.62      # radiation length in air [g/cm²]
E_CRITICAL_AIR = 0.080                  # critical energy for EM showers in air [GeV]

HNL_MC = {}
for mass in [5,6,7,8,9,10,20,30,40,50,60,70,80,90,95,96]:
    filename = "data/HNL_kinematics/Momentum%2.1f.dat" % mass
    HNL_MC[mass] = pd.read_csv(filename, sep=r'\s+')

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


def pion_range_in_air(E_pi_GeV, start_altitude_m, direction_cosine=1.0):
    """
    Estimate charged pion track length in air.

    Pions are stopped by nuclear interactions. The track length is
    approximately one nuclear interaction length in air (~120 g/cm²).
    At sea level (~1.225 kg/m³), this corresponds to ~1 km.
    At higher altitudes, the lower density means longer geometric range.

    Also limited by pion decay: tau_pi = 26 ns, so L_decay = beta*gamma*c*tau.

    Parameters
    ----------
    E_pi_GeV : float
        Pion kinetic energy [GeV]
    start_altitude_m : float
        Starting altitude [m]
    direction_cosine : float
        cos(theta) where theta is angle from vertical

    Returns
    -------
    range_m : float
        Pion effective track length [m]
    """
    m_pi = 0.13957  # GeV
    gamma = E_pi_GeV / m_pi
    beta = np.sqrt(1 - 1 / gamma**2) if gamma > 1 else 0.0

    # Decay length
    tau_pi = 26e-9  # seconds
    L_decay = beta * gamma * speed_of_light * tau_pi

    # Nuclear interaction length
    rho = air_density(start_altitude_m)  # kg/m³
    rho_gcm3 = rho * 1e-3  # g/cm³
    if rho_gcm3 > 0:
        L_interaction = PION_INTERACTION_LENGTH_GCM2 / rho_gcm3 / 100  # meters
    else:
        L_interaction = 1e6  # effectively infinite

    return min(L_decay, L_interaction)


def electron_shower_length_in_air(E_e_GeV, start_altitude_m):
    """
    Estimate the effective Cherenkov track length for an electromagnetic shower in air.

    An electron/positron (or gamma from pi0) initiates an EM shower.
    The effective track length for Cherenkov emission is approximately:
        L_eff = X_0 * ln(E / E_c)
    where X_0 is the radiation length and E_c is the critical energy.

    This counts the total path length of all shower particles above Cherenkov threshold.

    Parameters
    ----------
    E_e_GeV : float
        Electron/gamma energy [GeV]
    start_altitude_m : float
        Starting altitude [m]

    Returns
    -------
    range_m : float
        Effective Cherenkov track length [m]
    """
    if E_e_GeV <= E_CRITICAL_AIR:
        # Below critical energy, no shower development
        return 0.0

    rho = air_density(start_altitude_m)  # kg/m³
    rho_gcm3 = rho * 1e-3  # g/cm³

    if rho_gcm3 <= 0:
        return 0.0

    # Radiation length in meters at this altitude
    X_0_m = RADIATION_LENGTH_AIR_GCM2 / rho_gcm3 / 100  # meters

    # Number of radiation lengths for shower development
    t_max = np.log(E_e_GeV / E_CRITICAL_AIR)

    # Effective total track length: sum of all charged particle paths
    # Approximation: N_charged ~ E/E_c at shower max, total track ~ X_0 * E/E_c
    # More precisely, the total Cherenkov track length is:
    #   L_total ~ X_0 * (E / E_c) for the integrated path of all charged particles
    # But for a single effective track approximation:
    L_eff = X_0_m * t_max

    return L_eff


def particle_track_length(particle_type, energy_GeV, altitude_m, direction_cosine=1.0):
    """
    Compute track length for any charged particle type in air.

    Parameters
    ----------
    particle_type : str
        One of 'muon', 'electron', 'pion_charged', 'pion_neutral'
    energy_GeV : float
        Particle energy [GeV]
    altitude_m : float
        Altitude above sea level [m]
    direction_cosine : float
        cos(theta) from vertical

    Returns
    -------
    track_length : float
        Track length in meters
    """
    if particle_type == 'muon':
        return muon_range_in_air(energy_GeV, altitude_m, direction_cosine)
    elif particle_type == 'pion_charged':
        return pion_range_in_air(energy_GeV, altitude_m, direction_cosine)
    elif particle_type == 'electron':
        return electron_shower_length_in_air(energy_GeV, altitude_m)
    elif particle_type == 'pion_neutral':
        # pi0 -> 2 gamma -> EM showers
        # Each gamma gets ~E/2, treat as single EM shower with full energy
        return electron_shower_length_in_air(energy_GeV, altitude_m)
    else:
        return 0.0


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

# This function approximates the theta distribution of HNLs
def P_theta(theta):
    theta_lowmid,theta_midhigh = 3e-4,0.01
    a0_low,a1_low = 12.1,3.3
    a1_mid = 0.5
    a0_mid = a0_low - (a1_mid - a1_low)*np.log(theta_lowmid)
    a1_high = -2.5
    a0_high = a0_mid - (a1_high - a1_mid)*np.log(theta_midhigh)
    a0 = np.where(theta<theta_lowmid,a0_low,a0_mid)
    a0 = np.where(theta>theta_midhigh,a0_high,a0)
    a1 = np.where(theta<theta_lowmid,a1_low,a1_mid)
    a1 = np.where(theta>theta_midhigh,a1_high,a1)
    y = np.exp(a0 + a1*np.log(theta))
    norm = 1./np.sum(y)
    return y*norm

def muon_energy_in_earth(E_mu_initial, depth_m, rho=2.65):
    """
    Compute muon energy after propagating through standard rock.

    Uses the continuous energy loss approximation:
        -dE/dx = a + b*E
    where a ≈ 2 MeV/(g/cm²) (ionization, minimum ionizing)
    and b ≈ 3.5e-6 (g/cm²)^-1 (radiative: bremsstrahlung + pair production).

    The analytic solution is:
        E(x) = (E_0 + a/b) * exp(-b*x) - a/b
    where x is in g/cm².

    Parameters
    ----------
    E_mu_initial : float or array
        Initial muon energy [GeV]
    depth_m : float or array
        Depth of propagation through rock [m]
    rho : float
        Rock density [g/cm³] (default: 2.65 for standard rock)

    Returns
    -------
    E_mu : float or array
        Muon energy after traversing depth [GeV]. Clipped to 0 if muon stops.
    """
    a = 2.0e-3   # GeV / (g/cm²) — ionization loss
    b = 3.5e-6   # (g/cm²)^-1 — radiative loss coefficient

    # Convert depth in meters to column depth in g/cm²
    x = depth_m * 100 * rho  # depth_m * 100 cm/m * rho g/cm³ = g/cm²

    E_mu = (E_mu_initial + a / b) * np.exp(-b * x) - a / b
    return np.maximum(E_mu, 0.0)


def muon_critical_energy(rho=2.65):
    """
    Return the critical energy where radiative losses equal ionization losses.
    E_crit = a/b ≈ 571 GeV in standard rock.
    """
    a = 2.0e-3
    b = 3.5e-6
    return a / b


def muon_max_range_in_earth(E_mu_initial, rho=2.65):
    """
    Maximum range of a muon in rock before it stops [m].

    From E(x) = 0: x_max = (1/b) * ln(1 + b*E_0/a)
    """
    a = 2.0e-3
    b = 3.5e-6
    x_max = np.log(1 + b * E_mu_initial / a) / b  # g/cm²
    return x_max / (100 * rho)  # convert to meters


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

        self.hnl_mc = {}
        self.available_masses = []
        for mass in [5,6,7,8,9,10,20,30,40,50,60,70,80,90,95,96]:
            if mass in HNL_MC:
                self.hnl_mc[mass] = HNL_MC[mass]
                self.available_masses.append(mass)
        self.available_masses = np.array(self.available_masses)

    def _nearest_mc_mass(self, m_N):
        """Find the nearest available MC mass to the requested mass."""
        idx = np.argmin(np.abs(self.available_masses - m_N))
        return self.available_masses[idx]

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

    def compute_weighted_production_rate(self, m_N, U2, N_depth_bins=100):
        """
        Compute HNL production rate accounting for muon energy loss in Earth.

        The muon enters the target region at the deepest point (target_depth + L_target/2
        below the surface) and travels upward. Its energy decreases with depth
        traversed (Bethe-Bloch + radiative), so the cross section varies along
        the path.

        Returns
        -------
        N_HNLs_per_muon : float
            Total HNLs produced per muon, integrating sigma(E_mu(x)) along path.
        depth_bins : ndarray
            Depth values for each bin center [m below surface]
        xs_weights : ndarray
            Normalized cross-section weight at each depth bin (for importance sampling).
        """
        # The muon enters the rock at the bottom of the target region
        # and travels upward toward the surface.
        # Distance along beam path from entry point to surface = target_depth + L_target/2
        # (for beam_offset_angle = 0)

        # Discretize the target into depth bins
        # depth_from_surface ranges from (target_depth + L_target/2) at entry
        # to max(0, target_depth - L_target/2) at exit
        depth_max = self.target_depth + self.L_target / 2  # deepest point
        depth_min = max(0, self.target_depth - self.L_target / 2)  # shallowest

        # Bin edges in terms of depth below surface
        depth_edges = np.linspace(depth_max, depth_min, N_depth_bins + 1)
        depth_centers = 0.5 * (depth_edges[:-1] + depth_edges[1:])
        dx = np.abs(depth_edges[1] - depth_edges[0])  # bin width [m]

        # Muon energy at each depth bin center
        # The muon has traversed (depth_max - depth_center) meters of rock
        # from its entry point
        traversed = depth_max - depth_centers  # meters of rock from entry
        E_mu_local = muon_energy_in_earth(self.E_mu, traversed)

        # Cross section at each depth
        xs_local = np.array([sigma(E, m_N, U2) for E in E_mu_local])  # m^2

        # Integrated production rate: N = n_earth * sum(sigma_i * dx_i)
        N_HNLs_per_muon = n_earth_m3 * np.sum(xs_local * dx)

        # Normalized weights for importance sampling
        xs_weights = xs_local * dx
        total = np.sum(xs_weights)
        if total > 0:
            xs_weights = xs_weights / total
        else:
            xs_weights = np.ones(N_depth_bins) / N_depth_bins

        return N_HNLs_per_muon, depth_centers, xs_weights

    def sample_production_points_weighted(self, m_N, U2, N_samples, N_depth_bins=100):
        """
        Sample HNL production points weighted by local cross section.

        Production positions are importance-sampled: positions where the muon
        has higher energy (and thus higher cross section) are sampled more often.

        Returns
        -------
        production_points : ndarray, shape (N_samples, 3)
        N_HNLs_per_muon : float
        """
        N_HNLs_per_muon, depth_centers, xs_weights = \
            self.compute_weighted_production_rate(m_N, U2, N_depth_bins)

        # Sample depth bins according to cross-section weights
        bin_indices = np.random.choice(len(depth_centers), size=N_samples, p=xs_weights)
        # Add uniform jitter within each bin
        dx = np.abs(depth_centers[1] - depth_centers[0]) if len(depth_centers) > 1 else 1.0
        sampled_depths = depth_centers[bin_indices] + np.random.uniform(-dx/2, dx/2, N_samples)

        # Convert depth below surface to 3D coordinates
        target_center_z = -sampled_depths  # z coordinate (negative = underground)
        target_center_x = -sampled_depths * np.tan(self.beam_offset_angle)
        target_center_y = np.zeros(N_samples)

        production_points = np.column_stack([target_center_x, target_center_y, target_center_z])
        return production_points, N_HNLs_per_muon

    def sample_hnl_directions(self, m_N, N_samples):
        """
        Sample HNL emission directions from production.

        HNLs are produced with angular spread around the beam direction.
        The distribution is forward-peaked with characteristic angle ~ m_N/E_mu.
        """

        ## OLD treatment from Claude ##

        # # Characteristic production angle
        # theta_char = 0.1#m_N / self.E_mu

        # # Sample polar angles relative to beam direction
        # # Use exponential distribution for forward-peaked behavior
        # theta_rel = np.random.exponential(theta_char, N_samples)

        ## NEW treatment sampling from Peng-Cheng's simulation ##
        theta_vector = np.logspace(-5,0,10*N_samples)
        p_theta = P_theta(theta_vector)
        theta_rel = np.random.choice(theta_vector,N_samples,p=p_theta)


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

    def sample_kinematics(self, m_N, N_samples):
        """
        Sample HNL and muon kinematics from pre-computed MC data.

        Samples N_samples events with replacement from the MC DataFrame
        for the nearest available HNL mass.
        """
        mc_mass = self._nearest_mc_mass(m_N)
        df = self.hnl_mc[mc_mass]
        sampled = df.sample(n=N_samples, replace=True)

        hnl_Ptot = np.sqrt(sampled.PNx.values**2 + sampled.PNy.values**2 + sampled.PNz.values**2)
        hnl_dir = np.column_stack((sampled.PNx.values/hnl_Ptot,
                                   sampled.PNy.values/hnl_Ptot,
                                   sampled.PNz.values/hnl_Ptot))
        mu_Ptot = np.sqrt(sampled.Pmux.values**2 + sampled.Pmuy.values**2 + sampled.Pmuz.values**2)
        mu_dir = np.column_stack((sampled.Pmux.values/mu_Ptot,
                                  sampled.Pmuy.values/mu_Ptot,
                                  sampled.Pmuz.values/mu_Ptot))

        return sampled.PNe.values, hnl_dir, sampled.Pmue.values, mu_dir

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


def summarize_signal(photon_counts, N_HNLs_per_muon, N_samples, min_photons=10,
                     cherenkov_weight=1.0):
    """
    Apply a photon threshold to raw simulation output and compute signal summary.

    Parameters
    ----------
    photon_counts : array
        Photon counts for valid-decay events (from compute_signal_at_satellite).
    N_HNLs_per_muon : float
        HNL production rate per muon (from compute_signal_at_satellite).
    N_samples : int
        Total number of MC samples used in the simulation.
    min_photons : float
        Minimum photon count for detection.
    cherenkov_weight : float
        Reweight factor from subsampled Cherenkov evaluation (default 1.0).

    Returns
    -------
    detection_efficiency : float
    mean_photons : float
    number_of_events : float
    """
    detected = photon_counts >= min_photons
    detection_efficiency = np.sum(detected) * cherenkov_weight / N_samples
    mean_photons = np.mean(photon_counts[detected]) if np.any(detected) else 0.0
    number_of_events = N_HNLs_per_muon * detection_efficiency * N_muon_decays
    return detection_efficiency, mean_photons, number_of_events


def sensitivity_criterion(S, B, method='gaussian', CL=0.95):
    """
    Evaluate whether a signal S is detectable above background B.

    Parameters
    ----------
    S : float or array
        Expected signal events
    B : float or array
        Expected background events
    method : str
        'gaussian' : S > n_sigma * sqrt(B) where n_sigma corresponds to CL
        'poisson'  : S / sqrt(S + B) > n_sigma (profile likelihood approximation)
        'simple'   : S > n_sigma * sqrt(B) (same as gaussian, kept for clarity)
    CL : float
        Confidence level (default 0.95 -> ~2 sigma)

    Returns
    -------
    is_sensitive : bool or array
        Whether the point passes the sensitivity criterion
    significance : float or array
        Test statistic value (number of sigmas)
    """
    from scipy.stats import norm

    n_sigma = norm.ppf(CL)  # 1.645 for 95%, 1.96 for 97.5%

    S = np.asarray(S, dtype=float)
    B = np.asarray(B, dtype=float)

    if method in ('gaussian', 'simple'):
        # S > n_sigma * sqrt(B)
        with np.errstate(divide='ignore', invalid='ignore'):
            significance = np.where(B > 0, S / np.sqrt(B), np.where(S > 0, np.inf, 0.0))
        is_sensitive = significance > n_sigma

    elif method == 'poisson':
        # S / sqrt(S + B) > n_sigma
        with np.errstate(divide='ignore', invalid='ignore'):
            denom = np.sqrt(S + B)
            significance = np.where(denom > 0, S / denom, 0.0)
        is_sensitive = significance > n_sigma

    else:
        raise ValueError(f"Unknown method: {method}")

    return is_sensitive, significance


def find_sensitivity_limit(photon_counts_grid, N_HNLs_per_muon_grid,
                           N_samples, U2_arr, N_background,
                           min_photons=10, method='gaussian', CL=0.95,
                           cherenkov_weights=None):
    """
    Find the sensitivity limit U2 for a given HNL mass from a pre-computed scan.

    For each U2 value, computes the expected signal events and checks
    if it exceeds the background-limited threshold.

    Parameters
    ----------
    photon_counts_grid : array, shape (N_U2, N_samples)
        Raw photon counts from compute_signal_at_satellite for each U2.
    N_HNLs_per_muon_grid : array, shape (N_U2,)
        Production rate for each U2.
    N_samples : int
        Number of MC samples used.
    U2_arr : array
        Array of U2 values scanned.
    N_background : float
        Expected background events for this geometry.
    min_photons : float
        Photon threshold for detection.
    method : str
        Sensitivity method ('gaussian', 'poisson').
    CL : float
        Confidence level.
    cherenkov_weights : array or None
        Reweight factors per U2 point (default None = all 1.0).

    Returns
    -------
    U2_limit : float or None
        The smallest U2 that passes the sensitivity criterion, or None.
    """
    for i, U2 in enumerate(U2_arr):
        w = cherenkov_weights[i] if cherenkov_weights is not None else 1.0
        _, _, S = summarize_signal(
            photon_counts_grid[i], N_HNLs_per_muon_grid[i], N_samples,
            min_photons, cherenkov_weight=w
        )
        is_sensitive, _ = sensitivity_criterion(S, N_background, method=method, CL=CL)
        if is_sensitive:
            return U2

    return None


def compute_signal_at_satellite(m_N, E_mu, U2, flux_geometry,
                                N_samples=1000, use_energy_loss=True,
                                use_all_channels=True, Umu2=None, Ue2=0.0,
                                max_cherenkov_events=None):
    """
    Compute expected signal at a satellite position from HNL decays.

    Returns raw photon counts and production rate so that different
    photon thresholds can be applied after the fact via summarize_signal().

    Parameters
    ----------
    m_N : float
        HNL mass [GeV]
    E_mu : float
        Muon beam energy [GeV]
    U2 : float
        Mixing parameter squared (total)
    flux_geometry : HNLFluxGeometry
        Geometry configuration
    N_samples : int
        Number of MC samples
    use_energy_loss : bool
        If True, account for muon energy loss in Earth and weight production
        by local cross section. If False, use the old uniform approximation.
    use_all_channels : bool
        If True, sample all HNL decay channels and sum Cherenkov from all
        charged daughters. If False, use only the muon from MC data (old behavior).
    Umu2 : float or None
        Muon mixing parameter. If None, assumes Umu2 = U2 (pure muon mixing).
    Ue2 : float
        Electron mixing parameter (default 0).
    max_cherenkov_events : int or None
        If set, cap the number of valid-decay events that go through the
        expensive Cherenkov calculation. The returned cherenkov_weight
        compensates for the subsampling. Use this to keep job runtimes
        uniform across parameter space. None means no cap.

    Returns
    -------
    photon_counts : ndarray
        Photon count per MC sample (length N_samples).
    N_HNLs_per_muon : float
        Number of HNLs produced per muon.
    cherenkov_weight : float
        Reweight factor for subsampled Cherenkov evaluation. Equals 1.0
        when all valid events are evaluated, > 1.0 when capped.
    """
    if Umu2 is None:
        Umu2 = U2

    sat_position = flux_geometry.get_satellite_position()

    if use_energy_loss:
        # Use energy-loss-weighted production
        prod_points, N_HNLs_per_muon = \
            flux_geometry.sample_production_points_weighted(m_N, U2, N_samples)
    else:
        # Old uniform approach
        HNL_xs = sigma(E_mu, m_N, U2)
        N_HNLs_per_muon = HNL_xs * flux_geometry.L_target * n_earth_m3
        prod_points = flux_geometry.sample_production_points(N_samples)

    hnl_energy, hnl_dirs, mu_energy, mu_dirs = flux_geometry.sample_kinematics(m_N, N_samples)

    # HNL decay length
    decay_length = HNL_decay_length(m_N, U2, hnl_energy)

    # Sample decay points
    decay_points, travel_dist = flux_geometry.sample_decay_points(
        prod_points, hnl_dirs, decay_length
    )

    # Filter: decays in atmosphere (z > 0 and z < satellite altitude)
    above_surface = decay_points[:, 2] > 0
    below_satellite = decay_points[:, 2] < sat_position[2]
    valid_decay = above_surface & below_satellite

    if not np.any(valid_decay):
        return np.zeros(N_samples), N_HNLs_per_muon, 1.0

    # For valid decays, compute Cherenkov photons
    photon_counts = np.zeros(N_samples)
    valid_indices = np.where(valid_decay)[0]
    N_valid = len(valid_indices)

    # Subsample valid events if there are too many for the Cherenkov loop
    if max_cherenkov_events is not None and N_valid > max_cherenkov_events:
        eval_indices = np.random.choice(valid_indices, max_cherenkov_events, replace=False)
        cherenkov_weight = N_valid / max_cherenkov_events
    else:
        eval_indices = valid_indices
        cherenkov_weight = 1.0

    for idx in eval_indices:
        decay_pos = decay_points[idx]
        decay_altitude = max(0, decay_pos[2])

        if use_all_channels:
            # Sample full HNL decay with all channels
            daughters = sample_hnl_decay(
                m_N, hnl_energy[idx], hnl_dirs[idx], Umu2=Umu2, Ue2=Ue2
            )

            if not daughters:
                continue

            N_ph_total = 0.0
            for ptype, E_daughter, p_hat_daughter in daughters:
                # Skip downward-going particles
                if p_hat_daughter[2] <= 0:
                    continue

                r_rel = decay_pos - sat_position
                dir_cosine = max(p_hat_daughter[2], 0.1)

                track_length = particle_track_length(
                    ptype, E_daughter, decay_altitude, dir_cosine
                )

                if track_length <= 0:
                    continue

                N_track = min(1000, max(100, int(track_length / 100)))
                try:
                    N_ph, _, _ = cherenkov_photons_detected_vectorized(
                        r_rel, p_hat_daughter, track_length, R_det,
                        N_psi=300, N_track=N_track
                    )
                except Exception:
                    N_ph = 0

                # Apply atmospheric transmission
                zenith_angle = np.arccos(dir_cosine)
                transmission = cherenkov_transmission(decay_altitude, zenith_angle)
                N_ph *= transmission

                N_ph_total += N_ph

            photon_counts[idx] = N_ph_total

        else:
            # Old behavior: single muon from MC data
            mu_dir = mu_dirs[idx]

            if mu_dir[2] <= 0:
                continue

            r_rel = decay_pos - sat_position
            p_hat = mu_dir

            E_mu_daughter = mu_energy[idx]
            track_length = muon_range_in_air(
                E_mu_daughter, decay_altitude,
                direction_cosine=max(p_hat[2], 0.1)
            )

            N_track = min(1000, max(300, int(track_length / 100)))
            try:
                N_ph, _, _ = cherenkov_photons_detected_vectorized(
                    r_rel, p_hat, track_length, R_det, N_psi=300, N_track=N_track
                )
            except Exception:
                N_ph = 0

            # Apply atmospheric transmission
            zenith_angle = np.arccos(max(p_hat[2], 0.1))
            transmission = cherenkov_transmission(decay_altitude, zenith_angle)
            N_ph *= transmission

            photon_counts[idx] = N_ph

    return photon_counts, N_HNLs_per_muon, cherenkov_weight