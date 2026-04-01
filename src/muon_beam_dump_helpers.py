
import numpy as np
from src.cherenkov import *
from src.constants import *
from src.xs_and_decays import *

# Material constants for standard rock
X0_ROCK = 26.5      # radiation length [g/cm^2]
RHO_ROCK = 2.65     # density [g/cm^3]
MCS_CONSTANT = 13.6e-3  # Highland constant [GeV]

# Earth radius [m]
R_EARTH = 6.371e6

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


"""
Multiple Coulomb scattering for muons in rock.

Highland formula (PDG 2024, Sec. 34.3):
    theta_rms = 13.6 MeV / (beta*c*p) * sqrt(X/X0)

For varying momentum, integrate (13.6/p(x))^2 / X0 along the path.
"""


def muon_mcs_table(E_mu_initial, L_dump_m, rho=RHO_ROCK, n_steps=500):
    """
    Precompute cumulative MCS angle vs depth for a muon traversing rock.

    Returns an interpolation table: given a depth s [m] along the dump,
    returns the RMS projected scattering angle [rad] accumulated from
    s=0 (dump entrance) to s.

    Uses the Highland formula integrated with varying p(s) from
    continuous energy loss.

    Parameters
    ----------
    E_mu_initial : float
        Initial muon energy [GeV]
    L_dump_m : float
        Total dump length [m]
    rho : float
        Rock density [g/cm^3]
    n_steps : int
        Number of integration steps

    Returns
    -------
    s_arr : ndarray, shape (n_steps,)
        Depth along dump [m]
    theta_rms_arr : ndarray, shape (n_steps,)
        Cumulative RMS projected scattering angle [rad] at each depth
    """
    s_arr = np.linspace(0, L_dump_m, n_steps)
    ds = s_arr[1] - s_arr[0]
    dX = ds * 100 * rho  # column depth per step [g/cm^2]

    E_local = muon_energy_in_earth(E_mu_initial, s_arr, rho=rho)

    # Integrand: (13.6 MeV / p)^2 / X0, accumulated over steps of dX
    # Set to zero where muon has stopped
    valid = E_local > 0.5  # GeV
    integrand = np.where(valid, (MCS_CONSTANT / E_local)**2 / X0_ROCK, 0.0)

    # Cumulative integral: theta^2(s) = sum of integrand * dX
    theta2_cumulative = np.cumsum(integrand) * dX
    theta_rms_arr = np.sqrt(theta2_cumulative)

    return s_arr, theta_rms_arr


def get_mcs_at_depths(s_table, theta_table, depths_m):
    """
    Interpolate MCS angle at given depths from a precomputed table.

    Parameters
    ----------
    s_table : ndarray
        Depth array from muon_mcs_table [m]
    theta_table : ndarray
        RMS angle array from muon_mcs_table [rad]
    depths_m : ndarray
        Depths at which to evaluate [m]

    Returns
    -------
    theta_rms : ndarray
        RMS scattering angle at each depth [rad]
    """
    return np.interp(depths_m, s_table, theta_table)


def apply_mcs_smearing(directions, theta_rms):
    """
    Apply MCS angular smearing to particle direction vectors.

    For each particle, draws a random deflection angle from a 2D Gaussian
    with width theta_rms (projected), then rotates the direction vector.

    Parameters
    ----------
    directions : ndarray, shape (N, 3)
        Unit direction vectors (e.g. along beam axis before smearing)
    theta_rms : ndarray, shape (N,)
        RMS projected scattering angle [rad] for each particle

    Returns
    -------
    smeared_directions : ndarray, shape (N, 3)
        Smeared unit direction vectors
    """
    N = len(directions)
    theta_rms = np.atleast_1d(np.asarray(theta_rms, dtype=float))
    # MCS is Gaussian in two projected planes: theta_x and theta_y
    theta_total = np.abs(np.random.normal(0, theta_rms, N))
    phi_scat = np.random.uniform(0, 2*np.pi, N)

    # Build perpendicular basis for each direction
    d = directions
    ref = np.zeros_like(d)
    mostly_z = np.abs(d[:, 2]) > 0.9
    ref[mostly_z] = [1.0, 0.0, 0.0]
    ref[~mostly_z] = [0.0, 0.0, 1.0]

    # Gram-Schmidt to get e1 perp to d
    e1 = ref - np.sum(ref * d, axis=1, keepdims=True) * d
    e1 = e1 / np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(d, e1)

    # Smeared direction: rotate d by theta_total around axis in e1-e2 plane
    cos_t = np.cos(theta_total)
    sin_t = np.sin(theta_total)
    cos_p = np.cos(phi_scat)
    sin_p = np.sin(phi_scat)

    smeared = (cos_t[:, np.newaxis] * d
               + sin_t[:, np.newaxis] * (cos_p[:, np.newaxis] * e1
                                         + sin_p[:, np.newaxis] * e2))
    smeared = smeared / np.linalg.norm(smeared, axis=1, keepdims=True)

    return smeared


def rotate_frame(directions, new_z):
    """
    Rotate direction vectors from the z-axis frame into a new frame.

    Given directions defined relative to [0,0,1] (e.g. from MC or from
    muon decay sampling), rotate each one so that [0,0,1] maps to new_z.
    This is the correct way to apply MCS: the muon direction is deflected
    to new_z by MCS, and all products (HNL, neutrino, outgoing muon) that
    were sampled relative to [0,0,1] must be rotated by the same rotation.

    Parameters
    ----------
    directions : ndarray, shape (N, 3)
        Unit direction vectors in the z-axis frame
    new_z : ndarray, shape (N, 3)
        The new z-axis for each event (e.g. smeared muon directions)

    Returns
    -------
    rotated : ndarray, shape (N, 3)
        Direction vectors rotated into the new frame
    """
    N = len(directions)
    d = directions
    nz = new_z

    # For each event, build rotation matrix R that maps [0,0,1] -> new_z
    # Using Rodrigues' formula: R = I + [k]_x sin(theta) + [k]_x^2 (1-cos(theta))
    # where k = z_hat x new_z / |z_hat x new_z|, theta = arccos(new_z · z_hat)

    cos_theta = nz[:, 2]  # new_z dot [0,0,1]
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    # Cross product [0,0,1] x new_z = [-new_z_y, new_z_x, 0]
    kx = -nz[:, 1]
    ky = nz[:, 0]
    # kz = 0
    k_norm = np.sqrt(kx**2 + ky**2)

    # Handle the degenerate case: new_z ≈ [0,0,±1]
    small = k_norm < 1e-10

    # For non-degenerate cases, normalize k
    k_norm_safe = np.where(small, 1.0, k_norm)
    kx_n = kx / k_norm_safe
    ky_n = ky / k_norm_safe

    sin_theta = k_norm  # |z x nz| = sin(angle between them)

    # Apply Rodrigues: v_rot = v cos(t) + (k x v) sin(t) + k (k·v)(1-cos(t))
    # k = [kx_n, ky_n, 0]
    # k x d = [ky_n*d_z, -kx_n*d_z, kx_n*d_y - ky_n*d_x]
    # k · d = kx_n*d_x + ky_n*d_y
    cross_x = ky_n * d[:, 2]
    cross_y = -kx_n * d[:, 2]
    cross_z = kx_n * d[:, 1] - ky_n * d[:, 0]

    k_dot_d = kx_n * d[:, 0] + ky_n * d[:, 1]

    ct = cos_theta
    st = sin_theta

    rot_x = d[:, 0] * ct + cross_x * st + kx_n * k_dot_d * (1 - ct)
    rot_y = d[:, 1] * ct + cross_y * st + ky_n * k_dot_d * (1 - ct)
    rot_z = d[:, 2] * ct + cross_z * st + 0.0  # kz=0, so kz*k_dot_d = 0

    rotated = np.column_stack([rot_x, rot_y, rot_z])

    # For degenerate cases: new_z ≈ [0,0,1] → identity, new_z ≈ [0,0,-1] → flip
    rotated[small] = np.where(
        cos_theta[small, np.newaxis] > 0,
        d[small],           # new_z ≈ +z, no rotation needed
        -d[small]           # new_z ≈ -z, flip (shouldn't happen in practice)
    )

    # Re-normalize
    rotated = rotated / np.linalg.norm(rotated, axis=1, keepdims=True)

    return rotated
