"""
Run the charm-production dimuon BACKGROUND simulation for the balloon/TRINITY
HNL search.

Unlike the signal scans there are NO parameters to scan: the charm background
(beam nu_mu -> CC/NC charm DIS in the atmosphere -> mu + Hadrons + D, D -> mu)
depends only on the fixed beam-dump + detector configuration.  Parallelism is
therefore just independent random SEEDS that accumulate MC statistics; combine
the per-seed output files downstream.

Usage:
    python run_charm_background.py <seed> [N_samples] [max_cherenkov_events]

Saves per-event arrays including the discriminator handles the HNL-vs-charm
separation will use -- the two-muon lab OPENING ANGLE, the HADRONIC-shower
energy, per-muon Cherenkov photon counts, and the interaction geometry.

Must run under the spack "lienv" environment (SIREN + charm splines):
    source .../spack/share/spack/setup-env.sh && spack env activate lienv
Point SIREN_CHARM_SPLINE_DIR at the charm splines if not the default.
"""
import sys
import os
import time

import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

from src.balloon import HNLFluxGeometry
from src.charm_background import (CharmDISModel,
                                 compute_charm_background_at_satellite,
                                 summarize_charm_background)
from siren.dataclasses import Particle

# --- Configuration (kept consistent with cluster/run_scan_dimuon.py) ---
N_SAMPLES = 200000            # beam-neutrino MC events
MAX_CHERENKOV_EVENTS = 50000  # cap on expensive SIREN+Cherenkov evals; the tiny
                              # charm interaction probability means only a small
                              # fraction of events ever need evaluating, but the
                              # cap bounds worst-case cost (compensated by
                              # cherenkov_weight).  Set None to evaluate all.
E_MU = 5000                   # GeV
DUMP_DEPTH = 100              # m
DUMP_ANGLE = 1.53             # rad (nearly horizontal curved-Earth beam)
NU_TYPE = Particle.ParticleType.NuMuBar
INCLUDE_NC = False             # include NC charm in the charm fraction
INCLUDE_HADRONIC = True       # record + image the hadronic shower (discriminator)
UNIFORM_GEN = True            # importance-sample interaction altitude
MIN_PHOTONS_REPORT = 50.0     # threshold for the log-line N_bg (20 PE / 0.4)

# Detector positions -- identical to run_scan_dimuon.py so the charm background
# is directly comparable to the dimuon signal, station by station.
ALL_DETECTORS = [
    np.array([500, 0, 20000.0]),
    np.array([-500, 0, 20000.0]),
    np.array([0, 0, 20000.0]),
    np.array([500, 0, 50000.0]),
    np.array([-500, 0, 50000.0]),
    np.array([0, 0, 50000.0]),
    np.array([500, 0, 100000.0]),
    np.array([-500, 0, 100000.0]),
    np.array([0, 0, 100000.0]),
    # 5 km and 1 km station added for the expanded campaign, APPENDED so that indices
    # 0-8 (20/50/100 km) keep their meaning.  Central 5 km detector = index 11.
    np.array([500, 0, 5000.0]),
    np.array([-500, 0, 5000.0]),
    np.array([0, 0, 5000.0]),
    np.array([500, 0, 1000.0]),
    np.array([-500, 0, 1000.0]),
    np.array([0, 0, 1000.0]),
]

# Single-detector mode: set env DET_INDEX to a detector index (0-8) to run with
# only that detector, writing to a detector-specific output dir.  Unset -> the
# full 9-detector array (the aggregate).  Used to build the single-detector
# toggle in src/charm_vs_hnl.py.
_DET_INDEX = os.environ.get("DET_INDEX", "")
if _DET_INDEX == "":
    DETECTOR_POSITIONS = ALL_DETECTORS
    _det_tag = ""
else:
    DETECTOR_POSITIONS = [ALL_DETECTORS[int(_DET_INDEX)]]
    _det_tag = f"_det{int(_DET_INDEX)}"
N_DET = len(DETECTOR_POSITIONS)

# Detector radius (m): default 2 m; the campaign also runs R=4 m.  Always tag R
# (R2/R4) to match run_hnl_signal.py and the on-disk scan_results_balloon_charm_* dirs.
R_DET = float(os.environ.get("R_DET", "2.0"))
_r_tag = f"_R{R_DET:g}"

# Neutrino source: "scattering" (mu N -> nu_mu X in the dump, the default) or
# "decay" (mu+ -> e+ nu_mu-bar nu_e).  The dimuon channel needs the muon-flavour
# neutrino, which for a mu+ beam is nu_mu-bar, so decay mode uses NuMuBar.
CHARM_MODE = os.environ.get("CHARM_MODE", "scattering")

if CHARM_MODE == "decay":
    _mode_tag = "_decay"
else:
    _mode_tag = "_scattering"

OUT_SUBDIR = f"scan_results_balloon_charm{_mode_tag}{_det_tag}{_r_tag}"

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
if len(sys.argv) > 2:
    N_SAMPLES = int(sys.argv[2])
if len(sys.argv) > 3:
    MAX_CHERENKOV_EVENTS = None if sys.argv[3] in ("none", "None") else int(sys.argv[3])
np.random.seed(seed)

t0 = time.time()
geom = HNLFluxGeometry(E_mu=E_MU, dump_depth=DUMP_DEPTH, dump_angle=DUMP_ANGLE)
model = CharmDISModel(nu_type=NU_TYPE, include_nc=INCLUDE_NC, seed=seed)
print(f"Setup (geometry + charm splines): {time.time()-t0:.1f}s")
print(f"seed={seed}  N_samples={N_SAMPLES}  N_det={N_DET}  R_det={R_DET}  "
      f"mode={CHARM_MODE}  nu_type={NU_TYPE}  max_cherenkov_events={MAX_CHERENKOV_EVENTS}")
sys.stdout.flush()

t1 = time.time()
out = compute_charm_background_at_satellite(
    geom, model=model, detector_positions=DETECTOR_POSITIONS,
    N_samples=N_SAMPLES, max_cherenkov_events=MAX_CHERENKOV_EVENTS,
    uniform_gen=UNIFORM_GEN, include_hadronic_shower=INCLUDE_HADRONIC,
    include_nc=INCLUDE_NC, R_det=R_DET, mode=CHARM_MODE)
print(f"\nSimulation: {time.time()-t1:.1f}s  "
      f"N_nu_per_muon={out['N_nu_per_muon']:.3e}  "
      f"cherenkov_weight={out['cherenkov_weight']:.2f}")

# Quick N_bg for the log (both-muon and single-muon tag) at one threshold.
# summarize_charm_background now takes the unified `out` dict and returns a scalar.
N_bg_both = summarize_charm_background(
    out, N_SAMPLES, min_photons=MIN_PHOTONS_REPORT, both_muon_tag=True)
N_bg_single = summarize_charm_background(
    out, N_SAMPLES, min_photons=MIN_PHOTONS_REPORT, both_muon_tag=False)
print(f"N_bg @ {MIN_PHOTONS_REPORT:.0f} photons  both-muon: {N_bg_both:.3g}"
      f"  single-muon: {N_bg_single:.3g}")

outdir = os.path.join(project_root, "data", OUT_SUBDIR)
os.makedirs(outdir, exist_ok=True)
outfile = os.path.join(outdir, f"charm_bkg_seed_{seed:03d}.npz")
# `out` carries the unified per-muon photon-hit schema (identical to the HNL
# signal), plus the hadronic-shower fields, charm kinematics, and normalization
# (interaction_weights, N_nu_per_muon, cherenkov_weight, interaction_altitudes).
np.savez(
    outfile,
    **out,
    seed=seed,
    N_samples=N_SAMPLES,
    detector_positions=np.array(DETECTOR_POSITIONS),
    E_mu=E_MU, dump_depth=DUMP_DEPTH, dump_angle=DUMP_ANGLE,
    include_nc=INCLUDE_NC, R_det=R_DET, mode=CHARM_MODE,
    beam_axis_pixel=np.array([0.0, 0.0]),
    camera_optical_axis=np.array([0.0, 0.0, 1.0]),
)
print(f"Done! Saved to {outfile}")
print(f"Total time: {time.time()-t0:.1f}s")
