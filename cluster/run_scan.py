"""
Run balloon sensitivity scan for a single HNL mass and batch of U2 points.

Usage:
    python run_scan.py <mass_index> <u2_batch_index>

mass_index:    0-13, indexing into the list of available MC masses.
u2_batch_index: indexes into U2_RANGE in chunks of U2_BATCH_SIZE.

Saves raw photon counts per (mass, U2 batch) to disk, including:
- Signal photon counts, decay weights, and decay positions for each detector
- Background photon counts, interaction weights, and interaction positions
  (computed once per geometry)
"""
import sys
import os
import numpy as np
import time

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

from src.constants import *
from src.xs_and_decays import *
from src.balloon import compute_signal_at_satellite, HNLFluxGeometry
from src.background import compute_background_at_satellite

# --- Grid parameters ---
MASSES = np.array([5, 6, 7, 8, 9, 10, 12, 14, 16, 20, 25, 30, 40, 50, 60, 70, 80, 90])
U2_RANGE = np.logspace(-14, -7, 100)
U2_BATCH_SIZE = 1
N_SAMPLES = 1000000
N_SAMPLES_BKG = 1000000
E_MU = 5000  # GeV

# Geometry
DUMP_DEPTH = 100        # m (depth of beam dump origin below surface)
DUMP_ANGLE = 1.53  # rad (pi/2 = horizontal beam)

# Detector positions (multiple balloons)
DETECTOR_POSITIONS = [
    np.array([500, 0, 20000.0]),   # low balloon at 20 km: displaced 500 m in x
     np.array([-500, 0, 20000.0]),  # low balloon at 20 km: displaced -500 m in x
     np.array([0, 0, 20000.0]),   # low balloon at 20 km: centered
     np.array([500, 0, 50000.0]),   # low balloon at 50 km: displaced 500 m in x
     np.array([-500, 0, 50000.0]),  # low balloon at 50 km: displaced -500 m in x
     np.array([0, 0, 50000.0]),   # low balloon at 50 km: centered
     np.array([500, 0, 100000.0]),   # low balloon at 100 km: displaced 500 m in x
     np.array([-500, 0, 100000.0]),  # low balloon at 100 km: displaced -500 m in x
     np.array([0, 0, 100000.0]),   # low balloon at 100 km: centered
    #  np.array([500, 0, 150000.0]),   # high balloon at 150 km: displaced 500 m in x
    #  np.array([-500, 0, 150000.0]),  # high balloon at 150 km: displaced -500 m in x
    #  np.array([0, 0, 150000.0]),   # high balloon at 150 km: centered
]

N_MASSES = len(MASSES)
N_U2_BATCHES = int(np.ceil(len(U2_RANGE) / U2_BATCH_SIZE))
N_DET = len(DETECTOR_POSITIONS)

# --- Parse args ---
mass_idx = int(sys.argv[1])
u2_batch_idx = int(sys.argv[2])
run_bkg = bool(int(sys.argv[3])) if len(sys.argv) > 3 else False

m_N = MASSES[mass_idx]
u2_start = u2_batch_idx * U2_BATCH_SIZE
u2_end = min(u2_start + U2_BATCH_SIZE, len(U2_RANGE))
U2_batch = U2_RANGE[u2_start:u2_end]

print(f"m_N = {m_N} GeV (mass index {mass_idx})")
print(f"U2 batch {u2_batch_idx}: indices {u2_start}-{u2_end-1} "
      f"({U2_batch[0]:.2e} to {U2_batch[-1]:.2e})")
print(f"N_samples: {N_SAMPLES}, N_det: {N_DET}")
sys.stdout.flush()

# --- Setup geometry (loads MC data) ---
t0 = time.time()
flux_geometry = HNLFluxGeometry(
    E_mu=E_MU,
    dump_depth=DUMP_DEPTH,
    dump_angle=DUMP_ANGLE
)
print(f"Geometry setup: {time.time()-t0:.1f}s")
sys.stdout.flush()

# --- Output directory ---
outdir = os.path.join(project_root, "data", "scan_results_balloon")
os.makedirs(outdir, exist_ok=True)

if run_bkg:
    # --- Background (computed once for this geometry) ---
    print("Computing background...")
    t_bkg = time.time()
    bkg_photons, bkg_weights, N_nu_per_muon, bkg_ch_weight, bkg_positions = \
        compute_background_at_satellite(
            flux_geometry, N_samples=N_SAMPLES_BKG,
            detector_positions=DETECTOR_POSITIONS
        )
    print(f"Background done: {time.time()-t_bkg:.1f}s")
    sys.stdout.flush()

    # --- Save results ---
    outfile = os.path.join(outdir, f"scan_background.npz")
    np.savez(outfile,
            # Grid info
            N_samples_bkg=N_SAMPLES_BKG,
            detector_positions=np.array(DETECTOR_POSITIONS),
            # Background: (N_det, N_samples_bkg)
            bkg_photons=bkg_photons,
            bkg_weights=bkg_weights,
            bkg_positions=bkg_positions,
            N_nu_per_muon=N_nu_per_muon,
            bkg_cherenkov_weight=bkg_ch_weight)

else:

    # --- Signal scan over U2 batch ---
    # Shape: (N_U2, N_det, N_samples)
    photon_counts_batch = np.zeros((len(U2_batch), N_DET, N_SAMPLES))
    decay_weights_batch = np.zeros((len(U2_batch), N_SAMPLES))
    decay_positions_batch = np.zeros((len(U2_batch), N_SAMPLES, 3))
    N_HNLs_per_muon_batch = np.zeros(len(U2_batch))
    cherenkov_weights_batch = np.ones(len(U2_batch))

    for i, U2 in enumerate(U2_batch):
        t1 = time.time()
        try:
            ph_counts, decay_wts, n_hnl, ch_weight, decay_pts = \
                compute_signal_at_satellite(
                    m_N, E_MU, U2, flux_geometry, N_samples=N_SAMPLES,
                    use_energy_loss=True,
                    detector_positions=DETECTOR_POSITIONS
                )
            photon_counts_batch[i] = ph_counts
            decay_weights_batch[i] = decay_wts
            decay_positions_batch[i] = decay_pts
            N_HNLs_per_muon_batch[i] = n_hnl
            cherenkov_weights_batch[i] = ch_weight
        except Exception as e:
            print(f"  WARNING: U2={U2:.2e} failed: {e}")

        dt = time.time() - t1
        print(f"  U2={U2:.2e} ({i+1}/{len(U2_batch)}): {dt:.1f}s, weight={cherenkov_weights_batch[i]:.2f}")
        sys.stdout.flush()

    # --- Save results ---
    outfile = os.path.join(outdir, f"scan_mN_{m_N:.0f}_u2batch_{u2_batch_idx:03d}.npz")
    np.savez(outfile,
            # Grid info
            m_N=m_N,
            u2_start=u2_start,
            u2_end=u2_end,
            U2_batch=U2_batch,
            U2_range_full=U2_RANGE,
            N_samples=N_SAMPLES,
            detector_positions=np.array(DETECTOR_POSITIONS),
            # Signal: (N_U2, N_det, N_samples) photon counts
            photon_counts=photon_counts_batch,
            decay_weights=decay_weights_batch,
            decay_positions=decay_positions_batch,
            N_HNLs_per_muon=N_HNLs_per_muon_batch,
            cherenkov_weights=cherenkov_weights_batch)

print(f"Done! Saved to {outfile}")
print(f"Total time: {time.time()-t0:.1f}s")
