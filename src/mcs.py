"""
Multiple Coulomb scattering for muons in rock.

Highland formula (PDG 2024, Sec. 34.3):
    theta_rms = 13.6 MeV / (beta*c*p) * sqrt(X/X0)

For varying momentum, integrate (13.6/p(x))^2 / X0 along the path.
"""
import numpy as np
from src.balloon import muon_energy_in_earth

# Material constants for standard rock
X0_ROCK = 26.5      # radiation length [g/cm^2]
RHO_ROCK = 2.65     # density [g/cm^3]
MCS_CONSTANT = 13.6e-3  # Highland constant [GeV]


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
    # MCS is Gaussian in two projected planes: theta_x and theta_y
    theta_x = np.random.normal(0, theta_rms)
    theta_y = np.random.normal(0, theta_rms)
    theta_total = np.sqrt(theta_x**2 + theta_y**2)
    phi_scat = np.arctan2(theta_y, theta_x)

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
