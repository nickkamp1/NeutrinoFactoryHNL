"""
Run balloon sensitivity scan for a single HNL mass and batch of U2 points.

Usage:
    python run_scan.py <mass_index> <u2_batch_index>

mass_index:    0-13, indexing into the list of available MC masses.
u2_batch_index: indexes into U2_RANGE in chunks of U2_BATCH_SIZE.

Saves raw photon counts per (mass, U2 batch) to disk.
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

# --- Grid parameters ---
MASSES = np.array([5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90])
U2_RANGE = np.logspace(-14, -7, 100)
U2_BATCH_SIZE = 5
N_SAMPLES = 1000000
E_MU = 5000  # GeV

# Geometry
TARGET_DEPTH = 200      # m
SATELLITE_HEIGHT = 10000  # m (10 km)

N_MASSES = len(MASSES)
N_U2_BATCHES = int(np.ceil(len(U2_RANGE) / U2_BATCH_SIZE))

# --- Parse args ---
mass_idx = int(sys.argv[1])
u2_batch_idx = int(sys.argv[2])

m_N = MASSES[mass_idx]
u2_start = u2_batch_idx * U2_BATCH_SIZE
u2_end = min(u2_start + U2_BATCH_SIZE, len(U2_RANGE))
U2_batch = U2_RANGE[u2_start:u2_end]

print(f"m_N = {m_N} GeV (mass index {mass_idx})")
print(f"U2 batch {u2_batch_idx}: indices {u2_start}-{u2_end-1} "
      f"({U2_batch[0]:.2e} to {U2_batch[-1]:.2e})")
print(f"N_samples: {N_SAMPLES}")
sys.stdout.flush()

# --- Setup geometry (loads MC data) ---
t0 = time.time()
flux_geometry = HNLFluxGeometry(
    E_mu=E_MU,
    beam_offset_angle=0.0,
    target_depth=TARGET_DEPTH,
    satellite_height=SATELLITE_HEIGHT,
)
print(f"Geometry setup: {time.time()-t0:.1f}s")
sys.stdout.flush()

# --- Output directory ---
outdir = os.path.join(project_root, "data", "scan_results")
os.makedirs(outdir, exist_ok=True)

# --- Run batch ---
photon_counts_batch = np.zeros((len(U2_batch), N_SAMPLES))
N_HNLs_per_muon_batch = np.zeros(len(U2_batch))

for i, U2 in enumerate(U2_batch):
    t1 = time.time()
    try:
        ph_counts, n_hnl = compute_signal_at_satellite(
            m_N, E_MU, U2, flux_geometry, N_samples=N_SAMPLES
        )
        photon_counts_batch[i] = ph_counts
        N_HNLs_per_muon_batch[i] = n_hnl
    except Exception as e:
        print(f"  WARNING: U2={U2:.2e} failed: {e}")

    dt = time.time() - t1
    print(f"  U2={U2:.2e} ({i+1}/{len(U2_batch)}): {dt:.1f}s")
    sys.stdout.flush()

# --- Save results ---
outfile = os.path.join(outdir, f"scan_mN_{m_N:.0f}_u2batch_{u2_batch_idx:03d}.npz")
np.savez(outfile,
         m_N=m_N,
         u2_start=u2_start,
         u2_end=u2_end,
         U2_batch=U2_batch,
         U2_range_full=U2_RANGE,
         N_samples=N_SAMPLES,
         photon_counts=photon_counts_batch,
         N_HNLs_per_muon=N_HNLs_per_muon_batch)

print(f"Done! Saved to {outfile}")
print(f"Total time: {time.time()-t0:.1f}s")
