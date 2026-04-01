# HNL Flux Geometry Model - Upward Beam Configuration
import numpy as np
import pandas as pd

from src.cherenkov import *
from src.constants import *
from src.xs_and_decays import *
from src.flux import numu_flux

# Earth radius [m]
R_EARTH = 6.371e6

def sample_numu_from_muon_decay(E_mu, P_mu=0.0):
    """
    Sample nu_mu energy and direction from muon decay analytically.

    Samples in the muon rest frame where the distribution factorizes,
    then boosts to the lab frame.

    In the rest frame:
        dN/(dx d(cosθ_cm)) ∝ x² [(3 - 2x) + P_mu·cosθ_cm·(1 - 2x)]
    where x = 2E_nu_cm / m_mu ∈ [0, 1].

    Parameters
    ----------
    E_mu : np array(float)
        Muon energy in the lab frame [GeV]
    P_mu : float
        Muon polarization along the beam axis (-1 to +1). Default 0.

    Returns
    -------
    E_nu_lab : ndarray, shape (len(E_mu),)
        Neutrino energies in the lab frame [GeV]
    cos_theta_lab : ndarray, shape (len(E_mu),)
        Cosine of angle w.r.t. beam axis in the lab frame
    """
    gamma = E_mu / m_mu
    beta = np.sqrt(1.0 - 1.0 / gamma**2)

    # --- Step 1: sample x_cm from f(x) = x²(3-2x) via rejection ---
    # Max of x²(3-2x) is 27/32 at x=3/4
    f_max = 27.0 / 32.0
    x_cm = np.empty(len(E_mu))
    n_filled = 0
    while n_filled < len(E_mu):
        n_need = int(1.8 * (len(E_mu) - n_filled)) + 128  # ~59% acceptance
        x_prop = np.random.uniform(0, 1, n_need)
        f_val = x_prop**2 * (3.0 - 2.0 * x_prop)
        accept = np.random.uniform(0, f_max, n_need) < f_val
        x_accepted = x_prop[accept]
        n_take = min(len(x_accepted), len(E_mu) - n_filled)
        x_cm[n_filled:n_filled + n_take] = x_accepted[:n_take]
        n_filled += n_take

    # --- Step 2: sample cosθ_cm from P(c|x) = (1 + α·c)/2 on [-1,1] ---
    # where α = P_mu·(1-2x)/(3-2x)
    alpha = P_mu * (1.0 - 2.0 * x_cm) / (3.0 - 2.0 * x_cm)
    u = np.random.uniform(0, 1, len(E_mu))

    # Invert CDF: F(c) = (c+1)/2 + α(c²-1)/4 = u
    # → α/4·c² + c/2 + (1/2 - α/4 - u) = 0
    small_alpha = np.abs(alpha) < 1e-10
    alpha_safe = np.where(small_alpha, 1.0, alpha)  # avoid division by zero
    cos_cm = np.where(
        small_alpha,
        2.0 * u - 1.0,
        (-1.0 + np.sqrt(np.maximum(1.0 + 2.0 * alpha_safe * (2.0 * u - 1.0 + alpha_safe), 0.0))) / alpha_safe
    )
    cos_cm = np.clip(cos_cm, -1.0, 1.0)

    # --- Step 3: boost to lab frame ---
    E_nu_cm = x_cm * m_mu / 2.0
    E_nu_lab = gamma * E_nu_cm * (1.0 + beta * cos_cm)
    cos_theta_lab = (cos_cm + beta) / (1.0 + beta * cos_cm)

    return E_nu_lab, cos_theta_lab

def mcs_angle(L,E_mu_init):
    X0 = 26.5 # g/cm^2 for rock
    x = L * 2.65  # g/cm^2
    return 0.0136 * np.sqrt(x/X0) * (1 + 0.038 * np.log(x/X0)) / E_mu_init

def dump_length_earth(dump_depth, dump_angle):
    """
    Compute beam dump path length through rock using curved-Earth geometry.

    Uses the law of cosines on the triangle (Earth center, dump origin, surface exit).

    Parameters
    ----------
    dump_depth : float
        Depth of beam dump origin below the surface [m]
    dump_angle : float
        Angle of beam dump axis from vertical [rad] (0 = upgoing, pi/2 = horizontal)

    Returns
    -------
    L : float
        Path length through rock from dump origin to surface exit [m]
    """
    R = R_EARTH
    a = 1.0
    b = -2.0 * (R - dump_depth) * np.cos(np.pi - dump_angle)
    c = (R - dump_depth)**2 - R**2
    return (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)


def decay_length_earth(dump_depth, dump_angle, balloon_height=5e3):
    """
    Compute atmospheric decay region length using curved-Earth geometry.

    The decay region extends from the surface exit point to the point where
    the beam axis reaches balloon_height altitude, accounting for Earth curvature.

    Parameters
    ----------
    dump_depth : float
        Depth of beam dump origin below the surface [m]
    dump_angle : float
        Angle of beam dump axis from vertical [rad]
    balloon_height : float
        Balloon/detector altitude [m]

    Returns
    -------
    L : float
        Decay region length along beam axis [m]
    """
    R = R_EARTH
    L_dump = dump_length_earth(dump_depth, dump_angle)
    a = 1.0
    b = 2.0 * (L_dump - np.cos(np.pi - dump_angle) * (R - dump_depth))
    c = ((R - dump_depth)**2 + L_dump**2
         - 2.0 * L_dump * (R - dump_depth) * np.cos(np.pi - dump_angle)
         - (R + balloon_height)**2)
    return (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)


def d_max_curved_earth(cos_z, balloon_height):
    """
    Maximum distance a particle can travel before reaching balloon_height
    on a curved Earth.

    For a particle emitted at the surface going in direction with vertical
    component cos_z, the altitude above the (curved) surface after distance d is:
        h(d) = d * cos_z + d^2 * sin^2_z / (2 * R_earth)

    Solves h(d) = balloon_height for d.

    Parameters
    ----------
    cos_z : float or ndarray
        Vertical component of particle direction (cos of zenith angle)
    balloon_height : float
        Target altitude [m]

    Returns
    -------
    d_max : float or ndarray
        Distance to reach balloon_height [m]
    """
    cos_z = np.asarray(cos_z, dtype=float)
    sin2_z = 1.0 - cos_z**2
    a = sin2_z / (2.0 * R_EARTH)
    discriminant = cos_z**2 + 4.0 * a * balloon_height
    # For nearly vertical (a -> 0): d = balloon_height / cos_z
    result = np.where(
        a < 1e-15,
        balloon_height / np.maximum(cos_z, 1e-10),
        (-cos_z + np.sqrt(np.maximum(discriminant, 0.0)))
        / (2.0 * np.maximum(a, 1e-30))
    )
    return result


# Realistic muon track length calculation
# Muons lose energy via ionization in air; track length depends on energy and air density

# Track length constants for different particle types in air
PION_INTERACTION_LENGTH_GCM2 = 120.0   # nuclear interaction length in air [g/cm²]
RADIATION_LENGTH_AIR_GCM2 = 36.62      # radiation length in air [g/cm²]
E_CRITICAL_AIR = 0.080                  # critical energy for EM showers in air [GeV]
RHO_AIR_SEA_LEVEL = 1.225e-3            # sea level air density [g/cm³]

# Altitude-independent total Cherenkov yield constant (track-length integral)
# C_CH = (X_0 / ρ_0) × dN/dl_sea  [photons per (E/E_c)]
# The density cancels: longer radiation length at altitude × lower dN/dl = const.
C_CH = (RADIATION_LENGTH_AIR_GCM2 / (RHO_AIR_SEA_LEVEL * 100)) * get_cherenkov_yield_per_meter()

HNL_MC = {}
for mass in [5,6,7,8,9,10,12,14,16,20,25,30,40,50,60,70,80,90,95,96]:
    filename = "data/HNL_kinematics/Momentum%2.1f.dat" % mass
    HNL_MC[mass] = pd.read_csv(filename, sep=r'\s+')

BKG_MC = pd.read_csv("data/HNL_kinematics/Momentum.dat", sep=r'\s+')

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
    tau_mu = 2.2e-6 # seconds
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
    Total Cherenkov track length for an electromagnetic shower in air.

    The calorimetric track length integral gives the total path length of all
    charged particles above Cherenkov threshold:
        L_total = (E / E_c) × X_0 / ρ(z)
    This is the standard "track length integral" used in IACT analysis
    (Nerling et al. 2006, IceCube Astropart.Phys. 44 2013).

    Parameters
    ----------
    E_e_GeV : float
        Electron/gamma energy [GeV]
    start_altitude_m : float
        Starting altitude [m]

    Returns
    -------
    range_m : float
        Total Cherenkov track length [m]
    """
    if E_e_GeV <= E_CRITICAL_AIR:
        return 0.0

    rho = air_density(start_altitude_m)  # kg/m³
    rho_gcm3 = rho * 1e-3  # g/cm³

    if rho_gcm3 <= 0:
        return 0.0

    # Radiation length in meters at this altitude
    X_0_m = RADIATION_LENGTH_AIR_GCM2 / rho_gcm3 / 100  # meters

    # Total Cherenkov track length: sum of all charged particle paths
    # At shower max there are ~E/E_c particles, each traveling ~X_0
    L_total = X_0_m * (E_e_GeV / E_CRITICAL_AIR)

    return L_total


def hadronic_shower_cherenkov(E_had, had_dir, r_rel, altitude_m,
                              sigma_had=10 * np.pi / 180):
    """
    Cherenkov photons from a hadronic shower using the track-length-integral
    approach (Nerling et al. 2006).

    Total Cherenkov yield is altitude-independent:
        N_ph_total = f_EM(E) × (E / E_c) × C_CH
    where f_EM = 1 - E^{-0.14} (Gaisser) and C_CH ≈ 19,200 photons/(E/E_c).

    The hadronic angular spread (σ ~ 10°) >> Cherenkov angle (θ_C ~ 1.4°),
    so the ring structure is smeared into a Gaussian centered on the shower
    axis. The geometric acceptance is:
        Φ = (A_det / 2πσ²d²) × exp(-α² / 2σ²)
    where α = angle between shower axis and direction to detector.

    Parameters
    ----------
    E_had : float
        Total hadronic energy [GeV]
    had_dir : array (3,)
        Shower axis direction (unit vector)
    r_rel : array (3,)
        Shower position relative to detector [m] (= shower_pos - det_pos)
    altitude_m : float
        Shower altitude [m] (for atmospheric transmission)
    sigma_had : float
        Angular spread of shower particles [rad] (default 10°)

    Returns
    -------
    N_det : float
        Detected Cherenkov photons (includes atmospheric transmission)
    """
    if E_had <= 1.0:
        return 0.0

    # Total Cherenkov photons (altitude-independent)
    f_EM = 1.0 - E_had**(-0.14)
    N_ph_total = f_EM * (E_had / E_CRITICAL_AIR) * C_CH

    # Geometric acceptance: Gaussian angular distribution
    d = np.linalg.norm(r_rel)
    if d <= 0:
        return 0.0
    r_hat = r_rel / d
    cos_alpha = np.clip(-np.dot(had_dir, r_hat), -1, 1)
    alpha = np.arccos(cos_alpha)

    A_det = np.pi * R_det**2
    geom = (A_det / (2 * np.pi * sigma_had**2 * d**2)) * np.exp(
        -alpha**2 / (2 * sigma_had**2))

    # Atmospheric transmission along line of sight to detector
    cos_zenith = max(abs(r_hat[2]), 0.1)
    zenith = np.arccos(cos_zenith)
    transmission = cherenkov_transmission(altitude_m, zenith)

    return N_ph_total * geom * transmission


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
        return electron_shower_length_in_air(energy_GeV, altitude_m)
    else:
        return 0.0


def position_resolution(event_pos, balloon_pos, delta_theta=0.2*np.pi/180,
                        baseline=20e3):
    """
    Stereo position resolution for a two-balloon system.

    Transverse resolution (perpendicular to line of sight) is set by the
    pixel angular resolution.  Longitudinal resolution (along line of sight)
    requires stereo parallax from two cameras separated by baseline B.

    Parameters
    ----------
    event_pos : array, shape (3,) or (N, 3)
        Event position(s) [m]
    balloon_pos : array, shape (3,)
        Position of the nearer balloon [m]
    delta_theta : float
        Angular resolution per pixel [rad] (default 0.2 deg)
    baseline : float
        Stereo baseline between two balloons [m] (default 20 km)

    Returns
    -------
    sigma_perp : float or array
        Transverse position resolution [m]
    sigma_par : float or array
        Longitudinal (along line of sight) position resolution [m]
    """
    diff = np.asarray(event_pos) - np.asarray(balloon_pos)
    D = np.linalg.norm(diff, axis=-1)
    sigma_perp = D * delta_theta
    sigma_par = D**2 * delta_theta / baseline
    return sigma_perp, sigma_par

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
    - Muon beam travels through Earth from dump origin to the surface
    - HNLs are produced when muons scatter in Earth (below surface)
    - HNLs travel upward and decay in atmosphere
    - Balloon/satellite detects Cherenkov light from decay products

    The geometry is fully determined by dump_depth and dump_angle, which
    self-consistently fix the beam dump path length, beam direction, and
    atmospheric decay region length using curved-Earth geometry.

    Coordinate system:
    - Origin at the surface exit point of the beam
    - z-axis points up (local vertical, away from Earth center)
    - x-axis in the horizontal plane, along beam offset direction
    - Beam emerges from underground, traveling upward/outward

    Parameters
    ----------
    E_mu : float
        Muon beam energy [GeV]
    dump_depth : float
        Depth of beam dump origin below the surface [m]
    dump_angle : float
        Angle of beam dump axis from vertical [rad].
        0 = straight up, pi/2 = horizontal.
    """

    def __init__(self, E_mu=1500, dump_depth=200, dump_angle=np.pi/2):
        self.E_mu = E_mu
        self.dump_depth = dump_depth
        self.dump_angle = dump_angle

        # Compute beam dump path length through rock (curved-Earth geometry)
        self.L_target = dump_length_earth(dump_depth, dump_angle)

        # Beam direction at surface exit in the beam frame, i.e. along z axis
        self.beam_offset_angle = dump_angle
        self.beam_dir = np.array([
            0,   # x component (horizontal)
            0,   # y component
            1    # z component (positive = upward)
        ])

        # Compute decay region length
        self.decay_region_length = decay_length_earth(
            dump_depth, dump_angle, balloon_height=5000
        )
        print("Beam dump path length in rock: %.1f m" % self.L_target)
        print("Atmospheric decay region length for a balloon height 5 km: %.1f m" % (self.decay_region_length))

        # cos surface exit angle, 1 = vertical, 0 = horizontal. For a flat Earth, this would be cos(dump_angle).
        self.cos_surface_exit_angle = self.dump_depth/self.L_target + self.L_target/(2*R_EARTH) - self.dump_depth**2 / (2*R_EARTH*self.L_target)

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

    def compute_weighted_production_rate(self, m_N, U2, N_depth_bins=100, mode="scattering"):
        """
        Compute HNL production rate accounting for muon energy loss in Earth.

        The muon enters at the dump origin and traverses L_target meters of
        rock to the surface. Energy loss along the path reduces the cross
        section.

        Supports production via both scattering and decay

        Returns
        -------
        N_HNLs_per_muon : float
            Total HNLs produced per muon, integrating sigma(E_mu(s)) along path.
        s_centers : ndarray
            Distance along beam from dump origin [m] for each bin center.
        position_weight : ndarray
            Normalized cross-section or decay weight at each bin (for importance sampling).
        E_mu_local : ndarray
            Muon energy at each bin center along the path [GeV]
        """
        # Discretize the beam path: s=0 at dump origin, s=L_target at surface
        s_edges = np.linspace(0, self.L_target, N_depth_bins + 1)
        s_centers = 0.5 * (s_edges[:-1] + s_edges[1:])
        ds = s_edges[1] - s_edges[0]

        # Muon energy at each point: it has traversed s meters of rock
        E_mu_local = muon_energy_in_earth(self.E_mu, s_centers-s_centers[0])

        if mode == "scattering":
            # Cross section at each point
            scattering_coefficient_local = np.array([n_earth_m3 * sigma(E, m_N, U2) for E in E_mu_local])  # m^-1, TODO: needs to be updated to correct xs
            position_weights = scattering_coefficient_local * ds
        elif mode == "decay":
            # Decay coefficient at each point: Gamma / (beta*gamma*c) = m/E * Gamma / hbar_in_GeV_m
            decay_coefficient_local = np.array([m_mu/E * Gamma_muon / hbar_in_GeV_m if E>0 else 0 for E in E_mu_local]) # m^-1
            position_weights = decay_coefficient_local * ds

        # Integrated production rate: N = sum(decay or scattering length * ds)
        N_HNLs_per_muon = np.sum(position_weights * ds)

        total = np.sum(position_weights)
        if total > 0:
            position_weights = position_weights / total
        else:
            position_weights = np.ones(N_depth_bins) / N_depth_bins

        return N_HNLs_per_muon, s_centers, position_weights, E_mu_local

    def sample_production_points_weighted(self, m_N, U2, N_samples, N_depth_bins=100, mode="scattering"):
        """
        Sample HNL production points weighted by local cross section or decay length.

        Production positions are importance-sampled along the beam path:
        positions where the muon has higher energy (and thus higher cross
        section) are sampled more often.

        Returns
        -------
        production_points : ndarray, shape (N_samples, 3)
        E_mu_local : ndarray, shape (N_samples,)
            Muon energy at each production point [GeV]
        N_HNLs_per_muon : float
        """
        N_HNLs_per_muon, s_centers, position_weights, E_mu_local = \
            self.compute_weighted_production_rate(m_N, U2, N_depth_bins, mode)

        # Sample beam-path distance bins according to position weights
        bin_indices = np.random.choice(len(s_centers), size=N_samples, p=position_weights)
        ds = s_centers[1] - s_centers[0] if len(s_centers) > 1 else 1.0
        sampled_s = s_centers[bin_indices] + np.random.uniform(-ds/2, ds/2, N_samples)

        # Convert to 3D: surface exit at origin, beam comes from -beam_dir
        s_from_exit = self.L_target - sampled_s
        production_points = np.outer(-s_from_exit, self.beam_dir)
        return production_points, E_mu_local, N_HNLs_per_muon, s_centers

    def sample_kinematics(self, E_muon, m_N=None, mode="scattering"):
        """
        Sample HNL and muon kinematics from pre-computed MC data.

        Samples len(E_muon) events with replacement from the MC DataFrame
        for the nearest available HNL mass.
        """

        if mode == "scattering":
            if m_N is None:
                df = BKG_MC
                sampled = df.sample(n=len(E_muon), replace=True)
                nu_Ptot = np.sqrt(sampled.Pnux.values**2 + sampled.Pnuy.values**2 + sampled.Pnuz.values**2)
                nu_dir = np.column_stack((sampled.Pnux.values/nu_Ptot,
                                        sampled.Pnuy.values/nu_Ptot,
                                        sampled.Pnuz.values/nu_Ptot))
                return sampled.Pnue.values * E_muon/5000, nu_dir, None, None # approximation, tables are made for 5 TeV muons

            else:
                mc_mass = self._nearest_mc_mass(m_N)
                df = self.hnl_mc[mc_mass]
                sampled = df.sample(n=len(E_muon), replace=True)

                hnl_Ptot = np.sqrt(sampled.PNx.values**2 + sampled.PNy.values**2 + sampled.PNz.values**2)
                hnl_dir = np.column_stack((sampled.PNx.values/hnl_Ptot,
                                        sampled.PNy.values/hnl_Ptot,
                                        sampled.PNz.values/hnl_Ptot))
                mu_Ptot = np.sqrt(sampled.Pmux.values**2 + sampled.Pmuy.values**2 + sampled.Pmuz.values**2)
                mu_dir = np.column_stack((sampled.Pmux.values/mu_Ptot,
                                        sampled.Pmuy.values/mu_Ptot,
                                        sampled.Pmuz.values/mu_Ptot))

                return sampled.PNe.values * E_muon/5000, hnl_dir, sampled.Pmue.values * E_muon/5000, mu_dir # approximation, tables are made for 5 TeV muons
        elif mode == "decay":
            if m_N is None:
                Enu_lab, costheta_lab = sample_numu_from_muon_decay(E_muon)
                azimuth = np.random.uniform(0, 2*np.pi, len(E_muon))
                neutrino_directions = np.column_stack((np.sqrt(1 - costheta_lab**2) * np.cos(azimuth),
                                                       np.sqrt(1 - costheta_lab**2) * np.sin(azimuth),
                                                       costheta_lab))
                return Enu_lab, neutrino_directions, None, None
            else:
                raise ValueError("decay class does not support HNL kinematics")

    def sample_decay_points(self, production_points, hnl_directions, decay_length):
        """
        Sample HNL decay points given production points, directions, and decay length.
        """
        N = len(production_points)
        travel_distances = np.random.exponential(decay_length, N)
        decay_points = production_points + travel_distances[:, np.newaxis] * hnl_directions
        return decay_points, travel_distances


def summarize_signal(photon_counts, N_HNLs_per_muon, N_samples, min_photons=10,
                     cherenkov_weight=1.0, decay_weights=None):
    """
    Apply a photon threshold to raw simulation output and compute signal summary.

    When decay_weights are provided (from the truncated-exponential sampling),
    the expected event count is:
        N = N_HNLs_per_muon * N_muon_decays * <w_decay * 1(N_ph >= thr)>_MC
    This parallels the P_CC weighting in summarize_background().

    Parameters
    ----------
    photon_counts : array
        Photon counts from compute_signal_at_satellite.
        Shape (N_samples,) for single detector or (N_det, N_samples).
    N_HNLs_per_muon : float
        HNL production rate per muon (from compute_signal_at_satellite).
    N_samples : int
        Total number of MC samples used in the simulation.
    min_photons : float
        Minimum photon count for detection.
    cherenkov_weight : float
        Reweight factor from subsampled Cherenkov evaluation (default 1.0).
    decay_weights : array, shape (N_samples,), or None
        Per-event decay probability weight. If None, uses the old
        unweighted counting (backward compatible).

    Returns
    -------
    detection_efficiency : float or array
    mean_photons : float or array
    number_of_events : float or array
        If photon_counts is 2D, returns arrays of length N_det.
    """
    photon_counts = np.asarray(photon_counts)

    # Handle multi-detector case: recurse over detectors
    if photon_counts.ndim == 2:
        results = [summarize_signal(photon_counts[i], N_HNLs_per_muon,
                                    N_samples, min_photons, cherenkov_weight,
                                    decay_weights)
                   for i in range(photon_counts.shape[0])]
        return (np.array([r[0] for r in results]),
                np.array([r[1] for r in results]),
                np.array([r[2] for r in results]))

    # Single-detector case
    detected = photon_counts >= min_photons

    if decay_weights is not None:
        # Weighted by decay probability (parallels P_CC in background)
        weighted_sum = np.sum(decay_weights[detected]) * cherenkov_weight
        detection_efficiency = weighted_sum / N_samples
        if np.any(detected):
            mean_photons = np.average(photon_counts[detected],
                                      weights=decay_weights[detected])
        else:
            mean_photons = 0.0
    else:
        # Old unweighted counting (backward compatible)
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
                                detector_positions,
                                N_samples=1000, use_energy_loss=True,
                                include_hadronic_shower=True,
                                max_cherenkov_events=None,
                                uniform_gen=False):
    """
    Compute expected signal at one or more detector positions from HNL decays.

    The signal model is N → μ + hadronic shower (dominant CC channel for
    m_N > 1 GeV).  The muon kinematics come from pre-computed MC data;
    the hadronic energy is E_had = E_HNL - E_muon, with direction from
    momentum conservation.

    Each HNL is simulated out to d_max (distance to the highest detector)
    and assigned a decay_weight = 1 - exp(-d_max / L_decay), the probability
    to decay within the observable volume.

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
        by local cross section.
    include_hadronic_shower : bool
        If True, include Cherenkov from the hadronic shower in addition to
        the muon.  The shower is modeled with EM-fraction scaling and
        multiple sub-tracks for angular spread.
    max_cherenkov_events : int or None
        Cap on expensive Cherenkov evaluations.  The returned cherenkov_weight
        compensates for the subsampling.
    detector_positions : list of array-like
        List of 3D detector positions [m].

    Returns
    -------
    photon_counts : ndarray
        Photon count per MC sample. Shape (N_samples,) for single detector,
        or (N_det, N_samples) for multiple detectors.
    decay_weights : ndarray, shape (N_samples,)
        Per-event decay probability weight = 1 - exp(-d_max / L_decay).
    N_HNLs_per_muon : float
        Number of HNLs produced per muon.
    cherenkov_weight : float
        Reweight factor for subsampled Cherenkov evaluation.
    decay_points : ndarray, shape (N_samples, 3)
        Sampled decay positions.
    """
    # Set up detector positions
    detector_positions = [np.asarray(p, dtype=float) for p in detector_positions]
    N_det = len(detector_positions)
    max_det_height = max(p[2] for p in detector_positions)

    if use_energy_loss:
        prod_points, N_HNLs_per_muon = \
            flux_geometry.sample_production_points_weighted(m_N, U2, N_samples)
    else:
        HNL_xs = sigma(E_mu, m_N, U2)
        N_HNLs_per_muon = HNL_xs * flux_geometry.L_target * n_earth_m3
        prod_points = flux_geometry.sample_production_points(N_samples)

    hnl_energy, hnl_dirs, mu_energy, mu_dirs = flux_geometry.sample_kinematics(N_samples, m_N=m_N)

    # HNL decay length (per event)
    decay_length = HNL_decay_length(m_N, U2, hnl_energy)

    # --- Decay probability weighting (curved-Earth d_max) ---
    cos_z = hnl_dirs[:, 2]
    upward = cos_z > 0
    d_max = np.where(upward, d_max_curved_earth(cos_z, max_det_height), 0.0)
    d_max = np.maximum(d_max, 0.0)

    with np.errstate(divide='ignore', invalid='ignore'):
        decay_weights = np.where(
            d_max > 0,
            1.0 - np.exp(-d_max / decay_length),
            0.0
        )

    if uniform_gen:
        decay_dist = np.random.uniform(0, d_max, N_samples)
        with np.errstate(divide='ignore', invalid='ignore'):
            decay_pos_weights = np.where(
                decay_weights > 0,
                np.exp(-decay_dist / decay_length) * d_max
                / (decay_length * (1 - np.exp(-d_max / decay_length))),
                0.0
            )
    else:
        u = np.random.uniform(0, 1, N_samples)
        with np.errstate(divide='ignore', invalid='ignore'):
            exp_term = np.exp(-d_max / decay_length)
            decay_dist = np.where(
                decay_weights > 0,
                -decay_length * np.log(1.0 - u * (1.0 - exp_term)),
                0.0
            )
            decay_pos_weights = np.ones_like(decay_dist)

    # Compute decay points
    decay_points = prod_points + decay_dist[:, np.newaxis] * hnl_dirs

    # Filter: decays above surface
    above_surface = decay_points[:, 2] > 0
    valid_decay = above_surface & (decay_weights > 0)

    if not np.any(valid_decay):
        photon_counts = np.zeros((N_det, N_samples))
        return photon_counts, decay_weights, N_HNLs_per_muon, 1.0, decay_points

    photon_counts = np.zeros((N_det, N_samples))
    valid_indices = np.where(valid_decay)[0]
    N_valid = len(valid_indices)

    if max_cherenkov_events is not None and N_valid > max_cherenkov_events:
        eval_indices = np.random.choice(valid_indices, max_cherenkov_events, replace=False)
        cherenkov_weight = N_valid / max_cherenkov_events
    else:
        eval_indices = valid_indices
        cherenkov_weight = 1.0

    for idx in eval_indices:
        decay_pos = decay_points[idx]
        decay_altitude = max(0, decay_pos[2])
        mu_dir = mu_dirs[idx]

        if mu_dir[2] <= 0:
            continue

        # Filter detectors above the decay point
        valid_dets = [i for i, dp in enumerate(detector_positions)
                      if decay_pos[2] < dp[2]]
        if not valid_dets:
            continue

        dir_cosine = mu_dir[2]
        zenith_angle = np.arccos(dir_cosine)
        transmission = cherenkov_transmission(decay_altitude, zenith_angle)

        # --- Muon Cherenkov (computed once for all detectors) ---
        track_length = muon_range_in_air(
            mu_energy[idx], decay_altitude, direction_cosine=dir_cosine
        )
        N_track = min(1000, max(300, int(track_length / 100)))
        valid_det_pos = [detector_positions[i] for i in valid_dets]
        try:
            N_ph_mu_all = cherenkov_photons_multi_detector(
                decay_pos, mu_dir, track_length, R_det,
                valid_det_pos, N_psi=300, N_track=N_track
            )
            N_ph_mu_all *= transmission
        except Exception:
            N_ph_mu_all = np.zeros(len(valid_dets))

        # --- Hadronic shower Cherenkov ---
        for j, i_det in enumerate(valid_dets):
            N_ph_had = 0.0
            if include_hadronic_shower:
                E_had = hnl_energy[idx] - mu_energy[idx]
                if E_had > 1.0:
                    p_had = (hnl_energy[idx] * hnl_dirs[idx]
                             - mu_energy[idx] * mu_dirs[idx])
                    p_had_norm = np.linalg.norm(p_had)
                    if p_had_norm > 0:
                        had_dir = p_had / p_had_norm
                        r_rel = decay_pos - detector_positions[i_det]
                        N_ph_had = hadronic_shower_cherenkov(
                            E_had, had_dir, r_rel, decay_altitude
                        )

            photon_counts[i_det, idx] = N_ph_mu_all[j] + N_ph_had

    return (photon_counts, decay_weights * decay_pos_weights,
            N_HNLs_per_muon, cherenkov_weight, decay_points)