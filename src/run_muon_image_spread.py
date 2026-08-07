"""
Generate the HNL dimuon imaging observables (including the mu-mu OPENING ANGLE)
for one HNL mass over a grid of U2, via src.muon_image_spread.scan.

Unlike the sensitivity scan, this records the per-event kinematic/imaging handles
used for the HNL-vs-charm discriminator study -- in particular ``opening_deg``
(the lab mu-mu opening angle), plus per-muon photon counts, ``same_detector`` and
``both_detected`` tags.  Detection depends on U2 (through the decay position), and
heavier HNLs need lower U2 to decay in the detectable band, so each mass is run
over a U2 GRID; the downstream comparison pools the detected events (weighted).

Usage:
    python run_muon_image_spread.py <mass_index>

mass_index: index into MASSES (0..15).  One array task per mass.

Writes data/muon_image_spread_<m>.npz (overwrites any existing file for that mass
with the consistent multi-U2 configuration).

Must run under the spack "lienv" environment (SIREN):
    source .../spack/share/spack/setup-env.sh && spack env activate lienv
"""
import sys
import os
import time

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

from src.muon_image_spread import scan
from src.cherenkov import N_AIR

# Detectors + Cherenkov n-mode MUST match the charm background / dimuon scan for a
# consistent HNL-vs-charm comparison: the 9-station array (scan()'s default is a
# single 100 km detector -- not what we want) and uniform sea-level n (balloon_siren
# and charm_background call cherenkov_photons_multi_detector in its default
# uniform-n mode; muon_image_spread otherwise uses altitude-dependent n and gives
# ~100x fewer photons, which no threshold can tag).
ALL_DETECTORS = [
    np.array([500, 0, 20000.0]), np.array([-500, 0, 20000.0]), np.array([0, 0, 20000.0]),
    np.array([500, 0, 50000.0]), np.array([-500, 0, 50000.0]), np.array([0, 0, 50000.0]),
    np.array([500, 0, 100000.0]), np.array([-500, 0, 100000.0]), np.array([0, 0, 100000.0]),
]
# Single-detector mode via env DET_INDEX (0-8); unset -> full 9-detector array.
# Output filename gets a det<i> tag so single-detector files sit alongside the
# aggregate ones (src/charm_vs_hnl.py toggles between them).
_DET_INDEX = os.environ.get("DET_INDEX", "")
if _DET_INDEX == "":
    DETECTOR_POSITIONS = ALL_DETECTORS
    FILE_TAG = ""
else:
    DETECTOR_POSITIONS = [ALL_DETECTORS[int(_DET_INDEX)]]
    FILE_TAG = f"det{int(_DET_INDEX)}_"
UNIFORM_N = N_AIR   # sea-level index, matching the analysis

# Finer mass grid for smooth sensitivity contours (the opening-angle discriminator
# is m_N-dependent and CANNOT be reweighted across mass, so masses are simulated;
# U2 is reweighted).  20 masses; submit with --array=0-19.
MASSES = np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 25, 28,
                   30, 35, 40])

# With uniform_gen=True the decay distance is sampled UNIFORMLY on (0, d_max),
# independent of U2, so a SINGLE reference U2 covers all decay positions and the
# events reweight to ANY target (m_N, U2) with low variance
# (compute_reweighted_signal_at_satellite).  No per-mass U2 grid needed -- the
# reference U2 value is irrelevant to the reweighting (uniform sampling), so we
# use one central value purely to build the geometry.
U2_REF = 1e-12

N_SAMPLES = 1000000   # beam MC events (uniform_gen -> high valid-decay fraction);
                      # large N to smooth the high-U2 (short decay length) upper
                      # edge of the sensitivity band, which is coverage-limited.
MAX_EVENTS = 50000    # cap on per-muon imaging evals (single detector is cheap)

mass_idx = int(sys.argv[1])
m_N = float(MASSES[mass_idx])

outdir = os.path.join(project_root, "data")
os.makedirs(outdir, exist_ok=True)
out = os.path.join(outdir, f"muon_image_spread_{FILE_TAG}{m_N:.0f}.npz")

t0 = time.time()
print(f"m_N = {m_N:.0f} GeV (mass index {mass_idx})")
print(f"reference U2={U2_REF:.1e} (uniform_gen -> reweightable to any U2)")
print(f"N_samples={N_SAMPLES}  max_events={MAX_EVENTS}")
sys.stdout.flush()

scan([m_N], [U2_REF], detector_positions=DETECTOR_POSITIONS, output=out,
     N_samples=N_SAMPLES, max_events=MAX_EVENTS, uniform_n=UNIFORM_N,
     sampling="mixture")  # uniform(long-lived)+log-uniform(short-lived) -> covers
                          # BOTH sensitivity edges when reweighted to any (m_N,U2)

print(f"Done! Saved to {out}")
print(f"Total time: {time.time()-t0:.1f}s")
