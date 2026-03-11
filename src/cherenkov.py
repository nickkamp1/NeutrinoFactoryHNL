import numpy as np
from scipy.interpolate import RegularGridInterpolator

# Default physical constants
N_AIR = 1.0003  # index of refraction of air
ALPHA = 1/137.035999084  # fine-structure constant
LAMBDA_MIN = 300e-9  # minimum wavelength in meters
LAMBDA_MAX = 1000e-9  # maximum wavelength in meters


def cherenkov_transmission(altitude_m, zenith_angle_rad, lambda_min=LAMBDA_MIN, lambda_max=LAMBDA_MAX):
    """
    Compute the atmospheric transmission factor for Cherenkov photons
    traveling from emission altitude upward through the atmosphere.

    Accounts for Rayleigh scattering (molecular) and Mie scattering (aerosol).
    Returns a multiplicative factor in [0, 1] to apply to the raw photon count.

    Parameters
    ----------
    altitude_m : float or array
        Altitude of photon emission [m]
    zenith_angle_rad : float or array
        Zenith angle of photon propagation direction [rad].
        0 = straight up, pi/2 = horizontal.
    lambda_min : float
        Minimum wavelength [m] (default 300 nm)
    lambda_max : float
        Maximum wavelength [m] (default 1000 nm)

    Returns
    -------
    transmission : float or array
        Fraction of photons surviving atmospheric scattering (0 to 1)
    """
    altitude_m = np.asarray(altitude_m, dtype=float)
    zenith_angle_rad = np.asarray(zenith_angle_rad, dtype=float)

    # Atmosphere parameters
    H_R = 8500.0    # Rayleigh scale height [m]
    H_M = 1200.0    # Mie (aerosol) scale height [m]

    # Vertical optical depths from sea level to infinity (at reference wavelength 400 nm)
    # Rayleigh: tau_R ~ 0.36 at 400 nm at sea level
    # Mie: tau_M ~ 0.12 at 400 nm at sea level (clear atmosphere)
    tau_R_0_ref = 0.36   # Rayleigh optical depth at 400 nm, sea level to infinity
    tau_M_0_ref = 0.12   # Mie optical depth at 400 nm, sea level to infinity
    lambda_ref = 400e-9  # reference wavelength [m]

    # Band-averaged effective wavelength for Cherenkov spectrum (dN/dlambda ~ 1/lambda^2)
    # <1/lambda^4> for Rayleigh, <1/lambda^1.2> for Mie, weighted by 1/lambda^2 spectrum
    # Effective wavelength for Rayleigh: integral of lambda^-6 / integral of lambda^-2
    # Compute band-averaged scaling factors
    def _band_avg_power(n_power, lmin, lmax):
        """Compute <lambda^n> weighted by Cherenkov spectrum (1/lambda^2)."""
        # weight ~ 1/lambda^2, so <lambda^n> = integral(lambda^(n-2)) / integral(lambda^(-2))
        if n_power - 2 + 1 == 0:  # n_power = 1
            num = np.log(lmax / lmin)
        else:
            num = (lmax**(n_power - 1) - lmin**(n_power - 1)) / (n_power - 1)
        denom = (1.0 / lmin - 1.0 / lmax)
        return num / denom

    # Rayleigh: tau ~ lambda^(-4)
    rayleigh_factor = _band_avg_power(-4, lambda_min, lambda_max) * lambda_ref**4
    # Mie: tau ~ lambda^(-1.2) (Angstrom exponent ~1.2 for typical aerosol)
    mie_factor = _band_avg_power(-1.2, lambda_min, lambda_max) * lambda_ref**1.2

    # Optical depth from altitude z to infinity (vertical)
    # tau_R(z) = tau_R_0 * exp(-z / H_R)
    # tau_M(z) = tau_M_0 * exp(-z / H_M)
    tau_R_vertical = tau_R_0_ref * rayleigh_factor * np.exp(-altitude_m / H_R)
    tau_M_vertical = tau_M_0_ref * mie_factor * np.exp(-altitude_m / H_M)

    # Slant path correction: for upward-going photons at zenith angle theta,
    # the column depth is multiplied by 1/cos(theta) (plane-parallel approximation,
    # valid for zenith_angle < ~75 deg)
    cos_zen = np.cos(zenith_angle_rad)
    cos_zen = np.maximum(cos_zen, 0.1)  # prevent divergence near horizon

    tau_total = (tau_R_vertical + tau_M_vertical) / cos_zen

    transmission = np.exp(-tau_total)
    return transmission


def get_cherenkov_angle(n=N_AIR):
    """Return the Cherenkov angle for a relativistic particle in medium with index n."""
    return np.arccos(1.0 / n)


def get_cherenkov_yield_per_meter(n=N_AIR, lambda_min=LAMBDA_MIN, lambda_max=LAMBDA_MAX):
    """
    Return the number of Cherenkov photons emitted per meter of track.

    Uses the Frank-Tamm formula: dN/dx = 2*pi*alpha*sin^2(theta_C) * (1/lambda_min - 1/lambda_max)
    """
    theta_C = get_cherenkov_angle(n)
    return 2 * np.pi * ALPHA * np.sin(theta_C)**2 * (1/lambda_min - 1/lambda_max)


def orthonormal_basis(p_hat):
    """
    Construct an orthonormal basis (p_hat, e1, e2) given a unit vector p_hat.

    Parameters
    ----------
    p_hat : array-like, shape (3,)
        Unit vector defining the primary direction

    Returns
    -------
    e1, e2 : ndarray, shape (3,)
        Two unit vectors orthogonal to p_hat and each other
    """
    p_hat = np.asarray(p_hat, dtype=float)

    # Find a vector not parallel to p_hat
    if abs(p_hat[0]) < 0.9:
        v = np.array([1., 0., 0.])
    else:
        v = np.array([0., 1., 0.])

    # Gram-Schmidt
    e1 = v - np.dot(v, p_hat) * p_hat
    e1 = e1 / np.linalg.norm(e1)

    e2 = np.cross(p_hat, e1)

    return e1, e2


def cherenkov_direction(p_hat, psi, theta_C):
    """
    Compute the direction of a Cherenkov photon.

    Parameters
    ----------
    p_hat : array-like, shape (3,)
        Unit vector of particle momentum
    psi : float or array
        Azimuthal angle around p_hat (0 to 2*pi)
    theta_C : float
        Cherenkov angle

    Returns
    -------
    k_hat : ndarray, shape (3,) or (3, N)
        Unit vector(s) of Cherenkov photon direction
    """
    e1, e2 = orthonormal_basis(p_hat)

    psi = np.atleast_1d(psi)

    # k_hat = cos(theta_C) * p_hat + sin(theta_C) * (cos(psi) * e1 + sin(psi) * e2)
    k_hat = (np.cos(theta_C) * p_hat[:, np.newaxis] +
             np.sin(theta_C) * (np.cos(psi) * e1[:, np.newaxis] +
                                np.sin(psi) * e2[:, np.newaxis]))

    return k_hat.squeeze()


def ray_disk_intersection(ray_origin, ray_dir, R_det):
    """
    Find intersection of ray with disk at z=0 with radius R_det.

    Parameters
    ----------
    ray_origin : ndarray, shape (3,) or (3, N)
        Starting point(s) of ray
    ray_dir : ndarray, shape (3,) or (3, N)
        Direction(s) of ray (unit vectors)
    R_det : float
        Radius of detector disk

    Returns
    -------
    hits : bool or ndarray of bool
        Whether the ray hits the detector
    """
    ray_origin = np.atleast_2d(ray_origin.T).T  # shape (3, N)
    ray_dir = np.atleast_2d(ray_dir.T).T

    # Time to reach z=0 plane
    # ray_origin[2] + t * ray_dir[2] = 0
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        t = -ray_origin[2] / ray_dir[2]

    # Intersection point
    x_int = ray_origin[0] + t * ray_dir[0]
    y_int = ray_origin[1] + t * ray_dir[1]

    # Check if within detector radius and photon traveling toward detector (t > 0)
    r_squared = x_int**2 + y_int**2
    hits = (t > 0) & (r_squared <= R_det**2)

    return hits.squeeze()


def cherenkov_photons_detected(r_0, p_hat, track_length, R_det,
                                n=N_AIR, N_psi=100, N_track=100):
    """
    Compute the number of Cherenkov photons detected by a circular disk detector.

    The detector is a flat disk of radius R_det centered at the origin in the z=0 plane,
    with normal vector pointing in the -z direction (toward incoming particles).

    Parameters
    ----------
    r_0 : array-like, shape (3,)
        Starting position of the charged particle [m]
        Convention: z < 0 means below the detector
    p_hat : array-like, shape (3,)
        Unit vector of particle direction
    track_length : float
        Length of particle track to consider [m]
    R_det : float
        Radius of circular detector [m]
    n : float, optional
        Index of refraction (default: 1.0003 for air)
    N_psi : int, optional
        Number of azimuthal samples for Cherenkov cone (default: 100)
    N_track : int, optional
        Number of samples along track (default: 100)

    Returns
    -------
    N_photons : float
        Expected number of Cherenkov photons hitting the detector
    """
    r_0 = np.asarray(r_0, dtype=float)
    p_hat = np.asarray(p_hat, dtype=float)
    p_hat = p_hat / np.linalg.norm(p_hat)  # ensure unit vector

    theta_C = get_cherenkov_angle(n)
    dN_dx = get_cherenkov_yield_per_meter(n)

    # Azimuthal angles around particle direction
    psi_arr = np.linspace(0, 2*np.pi, N_psi, endpoint=False)
    dpsi = 2*np.pi / N_psi

    # Sample points along track
    s_arr = np.linspace(0, track_length, N_track)
    ds = track_length / (N_track - 1) if N_track > 1 else track_length

    # Pre-compute Cherenkov directions (shape: 3, N_psi)
    k_hat_arr = cherenkov_direction(p_hat, psi_arr, theta_C)

    N_photons = 0.0

    for s in s_arr:
        # Emission point
        r_emit = r_0 + s * p_hat

        # Check which Cherenkov directions hit the detector
        # Broadcast r_emit to match k_hat_arr shape
        r_emit_broadcast = r_emit[:, np.newaxis] * np.ones((1, N_psi))

        hits = ray_disk_intersection(r_emit_broadcast, k_hat_arr, R_det)

        # Fraction of Cherenkov cone that hits detector
        f_hit = np.sum(hits) / N_psi

        # Add photons from this track segment
        N_photons += dN_dx * ds * f_hit

    return N_photons


def cherenkov_photons_detected_vectorized(r_0, p_hat, track_length, R_det,
                                          n=N_AIR, N_psi=100, N_track=100):
    """
    Vectorized version of cherenkov_photons_detected for better performance.

    Same parameters and return value as cherenkov_photons_detected.
    """
    r_0 = np.asarray(r_0, dtype=float)
    p_hat = np.asarray(p_hat, dtype=float)
    p_hat = p_hat / np.linalg.norm(p_hat)

    theta_C = get_cherenkov_angle(n)
    dN_dx = get_cherenkov_yield_per_meter(n)

    # Azimuthal angles
    psi_arr = np.linspace(0, 2*np.pi, N_psi, endpoint=False)

    # Track positions
    s_arr = np.linspace(0, track_length, N_track)
    ds = track_length / (N_track - 1) if N_track > 1 else track_length

    # Cherenkov directions (3, N_psi)
    k_hat_arr = cherenkov_direction(p_hat, psi_arr, theta_C)

    # All emission points (3, N_track)
    r_emit_all = r_0[:, np.newaxis] + s_arr[np.newaxis, :] * p_hat[:, np.newaxis]

    # Broadcast to (3, N_track, N_psi)
    r_emit_broadcast = r_emit_all[:, :, np.newaxis]
    k_hat_broadcast = k_hat_arr[:, np.newaxis, :]

    # Time to reach z=0
    with np.errstate(divide='ignore', invalid='ignore'):
        t = -r_emit_broadcast[2] / k_hat_broadcast[2]

    # Intersection points
    x_int = r_emit_broadcast[0] + t * k_hat_broadcast[0]
    y_int = r_emit_broadcast[1] + t * k_hat_broadcast[1]

    # Check hits
    r_squared = x_int**2 + y_int**2
    hits = (t > 0) & (r_squared <= R_det**2)

    # Fraction hitting at each track position
    f_hit = np.sum(hits, axis=1) / N_psi  # shape (N_track,)

    # Total photons
    N_photons = dN_dx * ds * np.sum(f_hit)

    N_photons_per_step = dN_dx * ds * f_hit
    nonzero_mask = N_photons_per_step>0

    return N_photons, s_arr[nonzero_mask], N_photons_per_step[nonzero_mask]


def cherenkov_photons_multi_detector(r_0, p_hat, track_length, R_det,
                                     detector_positions,
                                     n=N_AIR, N_psi=100, N_track=100):
    """
    Compute Cherenkov photons hitting multiple detectors from a single track.

    The Cherenkov emission (directions and emission points) is computed once
    and reused for all detectors, avoiding redundant ray generation.

    Each detector is a flat disk of radius R_det with normal along -z
    (facing downward), centered at the given 3D position.

    Parameters
    ----------
    r_0 : array-like, shape (3,)
        Starting position of the charged particle [m] (absolute coordinates)
    p_hat : array-like, shape (3,)
        Unit vector of particle direction
    track_length : float
        Length of particle track [m]
    R_det : float
        Radius of circular detector [m]
    detector_positions : array-like, shape (N_det, 3)
        Absolute 3D positions of each detector [m]
    n : float, optional
        Index of refraction (default: 1.0003 for air)
    N_psi : int, optional
        Number of azimuthal samples for Cherenkov cone (default: 100)
    N_track : int, optional
        Number of samples along track (default: 100)

    Returns
    -------
    N_photons : ndarray, shape (N_det,)
        Expected Cherenkov photons hitting each detector
    """
    r_0 = np.asarray(r_0, dtype=float)
    p_hat = np.asarray(p_hat, dtype=float)
    p_hat = p_hat / np.linalg.norm(p_hat)
    detector_positions = np.asarray(detector_positions, dtype=float)
    if detector_positions.ndim == 1:
        detector_positions = detector_positions[np.newaxis, :]
    N_det = detector_positions.shape[0]

    theta_C = get_cherenkov_angle(n)
    dN_dx = get_cherenkov_yield_per_meter(n)

    # Azimuthal angles
    psi_arr = np.linspace(0, 2*np.pi, N_psi, endpoint=False)

    # Track positions
    s_arr = np.linspace(0, track_length, N_track)
    ds = track_length / (N_track - 1) if N_track > 1 else track_length

    # Cherenkov directions (3, N_psi)
    k_hat_arr = cherenkov_direction(p_hat, psi_arr, theta_C)

    # All emission points in absolute coordinates (3, N_track)
    r_emit_all = r_0[:, np.newaxis] + s_arr[np.newaxis, :] * p_hat[:, np.newaxis]

    # For each detector, shift to detector-relative coordinates and test hits
    # r_emit_all: (3, N_track), k_hat_arr: (3, N_psi)
    N_photons = np.zeros(N_det)

    for i_det in range(N_det):
        det = detector_positions[i_det]

        # Emission points relative to this detector: (3, N_track)
        r_rel = r_emit_all - det[:, np.newaxis]

        # Broadcast to (3, N_track, N_psi)
        r_rel_broadcast = r_rel[:, :, np.newaxis]
        k_hat_broadcast = k_hat_arr[:, np.newaxis, :]

        # Time for each ray to reach detector z-plane (z_rel = 0)
        with np.errstate(divide='ignore', invalid='ignore'):
            t = -r_rel_broadcast[2] / k_hat_broadcast[2]

        # Intersection points in detector plane
        x_int = r_rel_broadcast[0] + t * k_hat_broadcast[0]
        y_int = r_rel_broadcast[1] + t * k_hat_broadcast[1]

        # Check hits
        r_squared = x_int**2 + y_int**2
        hits = (t > 0) & (r_squared <= R_det**2)

        # Fraction hitting at each track position
        f_hit = np.sum(hits, axis=1) / N_psi  # shape (N_track,)

        N_photons[i_det] = dN_dx * ds * np.sum(f_hit)

    return N_photons


class CherenkovLookupTable:
    """
    Pre-computed lookup table for fast Cherenkov photon calculation.

    Uses cylindrical coordinates relative to the detector:
    - z: distance below detector (z < 0 in the coordinate system, but stored as positive)
    - r: radial distance from detector axis
    - theta: polar angle of particle direction (0 = toward detector)
    - phi: azimuthal angle of particle direction

    Parameters
    ----------
    R_det : float
        Detector radius [m]
    z_range : tuple
        (z_min, z_max) range of distances below detector [m]
    r_range : tuple
        (r_min, r_max) radial range [m]
    theta_range : tuple
        (theta_min, theta_max) polar angle range [rad]
    N_z, N_r, N_theta, N_phi : int
        Number of grid points in each dimension
    track_length : float
        Track length for Cherenkov calculation [m]
    """

    def __init__(self, R_det, z_range, r_range, theta_range,
                 N_z=50, N_r=50, N_theta=50, N_phi=36,
                 track_length=1000.0, n=N_AIR):

        self.R_det = R_det
        self.track_length = track_length
        self.n = n

        # Grid arrays
        self.z_arr = np.linspace(z_range[0], z_range[1], N_z)
        self.r_arr = np.linspace(r_range[0], r_range[1], N_r)
        self.theta_arr = np.linspace(theta_range[0], theta_range[1], N_theta)
        self.phi_arr = np.linspace(0, 2*np.pi, N_phi, endpoint=False)

        # Store grid parameters
        self.z_range = z_range
        self.r_range = r_range
        self.theta_range = theta_range

        # Initialize lookup table
        self.table = None
        self.interpolator = None

    def build(self, verbose=True):
        """Build the lookup table by computing photons at each grid point."""
        N_z = len(self.z_arr)
        N_r = len(self.r_arr)
        N_theta = len(self.theta_arr)
        N_phi = len(self.phi_arr)

        self.table = np.zeros((N_z, N_r, N_theta, N_phi))

        total = N_z * N_r * N_theta * N_phi
        count = 0

        for iz, z in enumerate(self.z_arr):
            if verbose:
                print(f"Building lookup table: z = {z:.1f} m ({iz+1}/{N_z})")

            for ir, r in enumerate(self.r_arr):
                for itheta, theta in enumerate(self.theta_arr):
                    for iphi, phi in enumerate(self.phi_arr):
                        # Convert to Cartesian
                        # Position: particle at (r, 0, -z) in detector frame
                        r_0 = np.array([r, 0, -z])

                        # Direction: theta from +z axis, phi in x-y plane
                        p_hat = np.array([
                            np.sin(theta) * np.cos(phi),
                            np.sin(theta) * np.sin(phi),
                            np.cos(theta)
                        ])

                        N_photons,_,_ = cherenkov_photons_detected_vectorized(
                            r_0, p_hat, self.track_length, self.R_det, n=self.n
                        )

                        self.table[iz, ir, itheta, iphi] = N_photons
                        count += 1

        # Build interpolator
        self.interpolator = RegularGridInterpolator(
            (self.z_arr, self.r_arr, self.theta_arr, self.phi_arr),
            self.table,
            method='linear',
            bounds_error=False,
            fill_value=0.0
        )

        if verbose:
            print("Lookup table complete.")

    def __call__(self, z, r, theta, phi):
        """
        Interpolate the number of photons at given coordinates.

        Parameters
        ----------
        z : float or array
            Distance below detector [m] (positive value)
        r : float or array
            Radial distance from detector axis [m]
        theta : float or array
            Polar angle of particle direction [rad]
        phi : float or array
            Azimuthal angle of particle direction [rad]

        Returns
        -------
        N_photons : float or array
            Interpolated number of Cherenkov photons
        """
        if self.interpolator is None:
            raise ValueError("Lookup table not built. Call build() first.")

        # Handle phi periodicity
        phi = phi % (2*np.pi)

        points = np.column_stack([
            np.atleast_1d(z),
            np.atleast_1d(r),
            np.atleast_1d(theta),
            np.atleast_1d(phi)
        ])

        result = self.interpolator(points)
        return result.squeeze()

    def from_cartesian(self, r_0, p_hat):
        """
        Compute photons from Cartesian coordinates.

        Parameters
        ----------
        r_0 : array-like, shape (3,) or (N, 3)
            Starting position(s) [m]
        p_hat : array-like, shape (3,) or (N, 3)
            Direction unit vector(s)

        Returns
        -------
        N_photons : float or array
        """
        r_0 = np.atleast_2d(r_0)
        p_hat = np.atleast_2d(p_hat)

        # Convert to cylindrical/spherical
        z = -r_0[:, 2]  # distance below detector (positive)
        r = np.sqrt(r_0[:, 0]**2 + r_0[:, 1]**2)

        # Direction angles
        theta = np.arccos(p_hat[:, 2])  # angle from +z
        phi = np.arctan2(p_hat[:, 1], p_hat[:, 0])
        phi = phi % (2*np.pi)

        return self(z, r, theta, phi)

    def save(self, filename):
        """Save lookup table to file."""
        np.savez(filename,
                 table=self.table,
                 z_arr=self.z_arr,
                 r_arr=self.r_arr,
                 theta_arr=self.theta_arr,
                 phi_arr=self.phi_arr,
                 R_det=self.R_det,
                 track_length=self.track_length,
                 n=self.n)

    @classmethod
    def load(cls, filename):
        """Load lookup table from file."""
        data = np.load(filename)

        z_range = (data['z_arr'][0], data['z_arr'][-1])
        r_range = (data['r_arr'][0], data['r_arr'][-1])
        theta_range = (data['theta_arr'][0], data['theta_arr'][-1])

        lut = cls(
            R_det=float(data['R_det']),
            z_range=z_range,
            r_range=r_range,
            theta_range=theta_range,
            N_z=len(data['z_arr']),
            N_r=len(data['r_arr']),
            N_theta=len(data['theta_arr']),
            N_phi=len(data['phi_arr']),
            track_length=float(data['track_length']),
            n=float(data['n'])
        )

        lut.z_arr = data['z_arr']
        lut.r_arr = data['r_arr']
        lut.theta_arr = data['theta_arr']
        lut.phi_arr = data['phi_arr']
        lut.table = data['table']

        lut.interpolator = RegularGridInterpolator(
            (lut.z_arr, lut.r_arr, lut.theta_arr, lut.phi_arr),
            lut.table,
            method='linear',
            bounds_error=False,
            fill_value=0.0
        )

        return lut
