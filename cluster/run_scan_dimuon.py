"""
Run the SIREN dimuon balloon/TRINITY sensitivity scan for a single HNL mass and
batch of U2 points.

Mirrors cluster/run_scan.py but uses SIRENDimuonGeometry (channel
N4 -> nu_mu mu- mu+, SIREN-sampled decay, two muon tracks, no hadronic shower).
The SIREN branching ratio BR(N->nu mu mu) is saved per point and must be folded
into the effective HNL rate downstream:  N_HNLs_eff = interaction_probability * BR.

Usage:
    python run_scan_dimuon.py <mass_index> <u2_batch_index>

mass_index:     index into MASSES.
u2_batch_index: index into U2_RANGE in chunks of U2_BATCH_SIZE.

Must run under the spack "lienv" environment (SIREN is installed there):
    source .../spack/share/spack/setup-env.sh && spack env activate lienv
"""
import sys
import os
import numpy as np
import time

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

from src.balloon_siren import SIRENDimuonGeometry

# --- Grid parameters (kept consistent with cluster/run_scan.py) ---
MASSES = np.array([5, 6, 7, 8, 9, 10, 12, 14, 16, 20, 25, 30, 40, 50, 60, 70])
U2_RANGE = np.logspace(-14, -7, 100)
U2_BATCH_SIZE = 5           # U2 points per array task (~19 min each => ~95 min/batch).
                            # MUST match N_U2_BATCHES + array range in submit_scan_dimuon.sh
                            # and U2_BATCH_SIZE in collect_results_dimuon.py.
N_SAMPLES = 200000          # SIREN decay sampling is cheap (~30 us/event); the
                            # Cherenkov calc dominates and is bounded by the cap.
MAX_CHERENKOV_EVENTS = 50000  # cap on expensive Cherenkov evals (~40 ms/event for
                              # two muons x 9 detectors => ~35 min/task); compensated
                              # by cherenkov_weight. Set None to evaluate all.
NATURE = "Majorana"
E_MU = 5000  # GeV

# Geometry
DUMP_DEPTH = 100        # m
DUMP_ANGLE = 1.53       # rad (nearly horizontal curved-Earth beam)

# Detector positions.  Transverse profile (distance from the beamline) is the
# discriminator, so keep off-axis stations for the transverse handle.  A
# TRINITY-style ground/mountain configuration would instead use low-altitude
# positions; edit here for that study.
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

# --- Setup geometry (loads MC production tables + builds SIREN decay) ---
t0 = time.time()
geom = SIRENDimuonGeometry(
    E_mu=E_MU,
    dump_depth=DUMP_DEPTH,
    dump_angle=DUMP_ANGLE,
    nature=NATURE,
    seed=12345,
)
print(f"Geometry setup: {time.time()-t0:.1f}s")
sys.stdout.flush()

outdir = os.path.join(project_root, "data", "scan_results_balloon_dimuon")
os.makedirs(outdir, exist_ok=True)

mass_idx = int(sys.argv[1])
u2_batch_idx = int(sys.argv[2])

m_N = MASSES[mass_idx]
u2_start = u2_batch_idx * U2_BATCH_SIZE
u2_end = min(u2_start + U2_BATCH_SIZE, len(U2_RANGE))
U2_batch = U2_RANGE[u2_start:u2_end]

BR_mumu = geom.dimuon_branching_ratio(m_N)
print(f"m_N = {m_N} GeV (mass index {mass_idx}), BR(N->nu mu mu) = {BR_mumu:.4e}")
print(f"U2 batch {u2_batch_idx}: indices {u2_start}-{u2_end-1} "
      f"({U2_batch[0]:.2e} to {U2_batch[-1]:.2e})")
print(f"N_samples: {N_SAMPLES}, N_det: {N_DET}, max_cherenkov_events: {MAX_CHERENKOV_EVENTS}")
sys.stdout.flush()

# Allocate batch arrays (hadronic dropped: leptonic channel)
photon_counts_batch = np.zeros((len(U2_batch), N_DET, N_SAMPLES))             # both muons summed
mu_photon_counts_batch = np.zeros((len(U2_batch), 2, N_DET, N_SAMPLES))       # per-muon (for both-tagging)
production_positions_batch = np.zeros((len(U2_batch), N_SAMPLES, 3))
decay_positions_batch = np.zeros((len(U2_batch), N_SAMPLES, 3))
decay_probability_batch = np.zeros((len(U2_batch), N_SAMPLES))
decay_pos_probability_batch = np.zeros((len(U2_batch), N_SAMPLES))
interaction_probability_batch = np.zeros(len(U2_batch))
cherenkov_weights_batch = np.ones(len(U2_batch))
hnl_energy_batch = np.zeros((len(U2_batch), N_SAMPLES))
decay_dist_batch = np.zeros((len(U2_batch), N_SAMPLES))
decay_length_batch = np.zeros((len(U2_batch), N_SAMPLES))
d_max_batch = np.zeros((len(U2_batch), N_SAMPLES))

for i, U2 in enumerate(U2_batch):
    t1 = time.time()
    try:
        (photon_counts,
         muon_photon_counts,        # sum of both muons (== photon_counts here)
         mu_photon_counts,          # (2, N_det, N_samples): per-muon
         hadronic_photon_counts,    # always zero
         prod_points,
         decay_points,
         decay_probability,
         decay_pos_probability,
         interaction_probability,
         cherenkov_weight,
         hnl_energy,
         decay_dist,
         decay_length,
         d_max,
         br_mumu) = \
            geom.compute_dimuon_signal_at_satellite(
                m_N, U2,
                detector_positions=DETECTOR_POSITIONS,
                N_samples=N_SAMPLES,
                use_energy_loss=True,
                max_cherenkov_events=MAX_CHERENKOV_EVENTS,
                uniform_gen=True,
            )
        photon_counts_batch[i] = photon_counts
        mu_photon_counts_batch[i] = mu_photon_counts
        production_positions_batch[i] = prod_points
        decay_positions_batch[i] = decay_points
        decay_probability_batch[i] = decay_probability
        decay_pos_probability_batch[i] = decay_pos_probability
        interaction_probability_batch[i] = interaction_probability
        cherenkov_weights_batch[i] = cherenkov_weight
        hnl_energy_batch[i] = hnl_energy
        decay_dist_batch[i] = decay_dist
        decay_length_batch[i] = decay_length
        d_max_batch[i] = d_max
    except Exception as e:
        print(f"  WARNING: U2={U2:.2e} failed: {e}")

    dt = time.time() - t1
    print(f"  U2={U2:.2e} ({i+1}/{len(U2_batch)}): {dt:.1f}s, "
          f"weight={cherenkov_weights_batch[i]:.2f}")
    sys.stdout.flush()

# Effective HNL rate folds in the dimuon branching ratio.
N_HNLs_per_muon_batch = interaction_probability_batch * BR_mumu

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
         BR_mumu=BR_mumu,
         photon_counts=photon_counts_batch,            # both muons summed
         mu_photon_counts=mu_photon_counts_batch,       # (batch, 2, N_det, N_samples): per-muon
         production_positions=production_positions_batch,
         decay_positions=decay_positions_batch,
         decay_probability=decay_probability_batch,
         decay_pos_probability=decay_pos_probability_batch,
         interaction_probability=interaction_probability_batch,
         N_HNLs_per_muon=N_HNLs_per_muon_batch,   # = interaction_probability * BR_mumu
         cherenkov_weights=cherenkov_weights_batch,
         hnl_energy=hnl_energy_batch,
         decay_dist=decay_dist_batch,
         decay_length=decay_length_batch,
         d_max=d_max_batch,
         uniform_gen=True,
         )
print(f"Done! Saved to {outfile}")
print(f"Total time: {time.time()-t0:.1f}s")
