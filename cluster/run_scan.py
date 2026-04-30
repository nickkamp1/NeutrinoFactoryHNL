"""
Run balloon sensitivity scan for a single HNL mass and batch of U2 points,
or compute the background.

Usage:
    python run_scan.py <mass_index> <u2_batch_index>   # signal
    python run_scan.py --bkg                            # background only

mass_index:    0-17, indexing into MASSES.
u2_batch_index: indexes into U2_RANGE in chunks of U2_BATCH_SIZE.
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
from src.balloon import HNLFluxGeometry
from src.background import compute_background_at_satellite

# --- Grid parameters ---
MASSES = np.array([5, 6, 7, 8, 9, 10, 12, 14, 16, 20, 25, 30, 40, 50, 60, 70])#, 80, 90])
U2_RANGE = np.logspace(-14, -7, 100)
U2_BATCH_SIZE = 1
N_SAMPLES = 500000
N_SAMPLES_BKG = 500000
MODE = "scattering"
E_MU = 5000  # GeV

# Geometry
DUMP_DEPTH = 100        # m (depth of beam dump origin below surface)
DUMP_ANGLE = 1.53       # rad (nearly horizontal beam)

# Detector positions (multiple balloons)
DETECTOR_POSITIONS = [
    np.array([500, 0, 20000.0]),
    np.array([-500, 0, 20000.0]),
    np.array([0, 0, 20000.0]),
    np.array([500, 0, 50000.0]),
    np.array([-500, 0, 50000.0]),
    np.array([0, 0, 50000.0]),
    np.array([500, 0, 100000.0]),
    np.array([-500, 0, 100000.0]),
    np.array([0, 0, 100000.0]),
]

N_MASSES = len(MASSES)
N_U2_BATCHES = int(np.ceil(len(U2_RANGE) / U2_BATCH_SIZE))
N_DET = len(DETECTOR_POSITIONS)

# --- Parse args ---
run_bkg = "--bkg" in sys.argv

# --- Setup geometry (loads MC data) ---
t0 = time.time()
flux_geometry = HNLFluxGeometry(
    E_mu=E_MU,
    dump_depth=DUMP_DEPTH,
    dump_angle=DUMP_ANGLE,
)
print(f"Geometry setup: {time.time()-t0:.1f}s")
sys.stdout.flush()

# --- Output directory ---
outdir = os.path.join(project_root, "data", "scan_results_balloon_detailed")
os.makedirs(outdir, exist_ok=True)

if run_bkg:
    # --- Background ---
    print(f"Computing background (N_samples={N_SAMPLES_BKG}, N_det={N_DET})...")
    sys.stdout.flush()
    t_bkg = time.time()
    bkg_photons, bkg_weights, N_nu_per_muon, bkg_ch_weight, bkg_positions = \
        compute_background_at_satellite(
            flux_geometry, N_samples=N_SAMPLES_BKG,
            detector_positions=DETECTOR_POSITIONS,
            uniform_gen=True,
            mode=MODE
        )
    print(f"Background done: {time.time()-t_bkg:.1f}s")
    sys.stdout.flush()

    outfile = os.path.join(outdir, f"scan_background_{MODE}.npz")
    np.savez(outfile,
             N_samples_bkg=N_SAMPLES_BKG,
             detector_positions=np.array(DETECTOR_POSITIONS),
             bkg_photons=bkg_photons,
             bkg_weights=bkg_weights,
             bkg_positions=bkg_positions,
             N_nu_per_muon=N_nu_per_muon,
             bkg_cherenkov_weight=bkg_ch_weight)
    print(f"Saved to {outfile}")

else:
    # --- Signal ---
    mass_idx = int(sys.argv[1])
    u2_batch_idx = int(sys.argv[2])

    m_N = MASSES[mass_idx]
    u2_start = u2_batch_idx * U2_BATCH_SIZE
    u2_end = min(u2_start + U2_BATCH_SIZE, len(U2_RANGE))
    U2_batch = U2_RANGE[u2_start:u2_end]

    print(f"m_N = {m_N} GeV (mass index {mass_idx})")
    print(f"U2 batch {u2_batch_idx}: indices {u2_start}-{u2_end-1} "
          f"({U2_batch[0]:.2e} to {U2_batch[-1]:.2e})")
    print(f"N_samples: {N_SAMPLES}, N_det: {N_DET}")
    sys.stdout.flush()

    # Signal scan over U2 batch
    photon_counts_batch = np.zeros((len(U2_batch), N_DET, N_SAMPLES))
    muon_photon_counts_batch = np.zeros((len(U2_batch), N_DET, N_SAMPLES))
    hadronic_photon_counts_batch = np.zeros((len(U2_batch), N_DET, N_SAMPLES))
    production_positions_batch = np.zeros((len(U2_batch), N_SAMPLES, 3))
    decay_positions_batch = np.zeros((len(U2_batch), N_SAMPLES, 3))
    decay_probability_batch = np.zeros((len(U2_batch), N_SAMPLES))
    decay_pos_probability_batch = np.zeros((len(U2_batch), N_SAMPLES))
    interaction_probability_batch = np.zeros(len(U2_batch))
    cherenkov_weights_batch = np.ones(len(U2_batch))

    for i, U2 in enumerate(U2_batch):
        t1 = time.time()
        try:
            (photon_counts,
            muon_photon_counts,
            hadronic_photon_counts,
            prod_points,
            decay_points,
            decay_probability,
            decay_pos_probability,
            interaction_probability,
            cherenkov_weight) = \
                flux_geometry.compute_signal_at_satellite(
                    m_N, U2, N_samples=N_SAMPLES,
                    use_energy_loss=True,
                    detector_positions=DETECTOR_POSITIONS,
                    uniform_gen=True
                )
            photon_counts_batch[i] = photon_counts
            muon_photon_counts_batch[i] = muon_photon_counts
            hadronic_photon_counts_batch[i] = hadronic_photon_counts
            production_positions_batch[i] = prod_points
            decay_positions_batch[i] = decay_points
            decay_probability_batch[i] = decay_probability
            decay_pos_probability_batch[i] = decay_pos_probability
            interaction_probability_batch[i] = interaction_probability
            cherenkov_weights_batch[i] = cherenkov_weight
        except Exception as e:
            print(f"  WARNING: U2={U2:.2e} failed: {e}")

        dt = time.time() - t1
        print(f"  U2={U2:.2e} ({i+1}/{len(U2_batch)}): {dt:.1f}s, "
              f"weight={cherenkov_weights_batch[i]:.2f}")
        sys.stdout.flush()

    outfile = os.path.join(outdir,
                           f"scan_mN_{m_N:.0f}_u2batch_{u2_batch_idx:03d}.npz")
    np.savez(outfile,
             m_N=m_N,
             u2_start=u2_start,
             u2_end=u2_end,
             U2_batch=U2_batch,
             U2_range_full=U2_RANGE,
             N_samples=N_SAMPLES,
             detector_positions=np.array(DETECTOR_POSITIONS),
             photon_counts = photon_counts_batch,
             muon_photon_counts = muon_photon_counts_batch,
             hadronic_photon_counts = hadronic_photon_counts_batch,
             production_positions = production_positions_batch,
             decay_positions = decay_positions_batch,
             decay_probability = decay_probability_batch,
             decay_pos_probability = decay_pos_probability_batch,
             interaction_probability = interaction_probability_batch,
             cherenkov_weights = cherenkov_weights_batch
            )
    print(f"Done! Saved to {outfile}")

print(f"Total time: {time.time()-t0:.1f}s")
