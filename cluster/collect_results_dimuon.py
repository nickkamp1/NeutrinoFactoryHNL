"""
Collect SIREN dimuon scan results, apply a photoelectron threshold, and (since
the transverse profile is the discriminator) optionally apply a cut on the
event's transverse distance from the beamline.

Usage:
    python collect_results_dimuon.py [--min-pe 12] [--pde 0.40]
                                     [--max-transverse 500]
                                     [--output data/scan_results_balloon_dimuon/combined.npz]

The threshold is specified in photoelectrons (TRINITY-informed default ~12) and
converted to raw photons via the SiPM PDE.  BR(N->nu mu mu) is already folded
into the saved N_HNLs_per_muon.
"""
import os
import sys
import argparse
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

from src.constants import N_muon_decays

MASSES = np.array([5, 6, 7, 8, 9, 10, 12, 14, 16, 20, 25, 30, 40, 50, 60, 70])
U2_RANGE = np.logspace(-14, -7, 100)
U2_BATCH_SIZE = 5  # must match run_scan_dimuon.py / submit_scan_dimuon.sh
N_U2_BATCHES = int(np.ceil(len(U2_RANGE) / U2_BATCH_SIZE))


def beamline_dir(scan_dir):
    """Recover the beam direction from any available file's geometry, or assume
    the near-horizontal +x-dominated curved-Earth axis.  Production points lie
    along -beam_dir from the surface exit; the beamline passes through the
    origin along beam_dir."""
    # The dump exit is at the origin; the beam climbs nearly horizontally.
    # We compute transverse distance as the perpendicular distance of the decay
    # point from the line through the origin along the mean HNL direction.
    return None  # computed per-file from production geometry below


def transverse_distance(decay_points, prod_points):
    """Perpendicular distance of each decay point from the beamline.

    The beamline is the line through the dump exit (origin) along the beam
    direction.  We estimate the beam direction from the production points
    (which lie along -beam_dir), then project out the along-beam component."""
    # beam_dir ~ -mean(prod_points) normalised (prod points are at -s*beam_dir)
    v = -np.mean(prod_points.reshape(-1, 3), axis=0)
    n = np.linalg.norm(v)
    if n == 0:
        beam = np.array([0.0, 0.0, 1.0])
    else:
        beam = v / n
    dp = decay_points.reshape(-1, 3)
    along = dp @ beam
    perp = dp - np.outer(along, beam)
    return np.linalg.norm(perp, axis=1).reshape(decay_points.shape[:-1])


def load_raw_results(scan_dir="data/scan_results_balloon_dimuon"):
    import glob
    found = sorted(glob.glob(os.path.join(scan_dir, "scan_mN_*_u2batch_*.npz")))
    if not found:
        raise FileNotFoundError(f"No scan result files found in {scan_dir}")
    N_samples = int(np.load(found[0])["N_samples"])

    out = {}  # m_N -> dict of assembled arrays
    n_missing = 0
    missing_array_ids = []

    for i_m, m_N in enumerate(MASSES):
        ph = np.zeros((len(U2_RANGE), N_samples))         # single-tag metric: max_det(brighter muon)
        both = np.zeros((len(U2_RANGE), N_samples))       # both-tag metric: max_det(min(mu1,mu2))
        decay_pos = np.zeros((len(U2_RANGE), N_samples, 3))
        prod_pos = np.zeros((len(U2_RANGE), N_samples, 3))
        decay_prob = np.zeros((len(U2_RANGE), N_samples))
        nhnl = np.zeros(len(U2_RANGE))
        weights = np.ones(len(U2_RANGE))
        n_det = None

        for i_b, b in enumerate(range(N_U2_BATCHES)):
            array_idx = i_m * N_U2_BATCHES + i_b
            fpath = os.path.join(scan_dir, f"scan_mN_{m_N:.0f}_u2batch_{b:03d}.npz")
            if not os.path.exists(fpath):
                n_missing += 1
                missing_array_ids.append(array_idx)
                continue
            data = np.load(fpath)
            s, e = int(data["u2_start"]), int(data["u2_end"])
            mpc = data["mu_photon_counts"]        # (batch, 2, N_det, N_samples)
            n_det = mpc.shape[2]
            # Single-tag: brighter of the two muons on the most-illuminated
            # detector.  Each muon images to ~one pixel and the two muons usually
            # fall in DIFFERENT pixels, so the per-pixel quantity is the brighter
            # muon, not the summed disk count (data["photon_counts"]).
            brighter = mpc.max(axis=1)            # (batch, N_det, N_samples): brighter muon per det
            ph[s:e] = brighter.max(axis=1)        # best detector
            # Both-tag (background-free): require BOTH muons above threshold on the
            # SAME detector.  Reduce to a per-event metric = max_det( min(mu1,mu2) ),
            # so both-tag at threshold T  <=>  metric >= T.
            weaker = mpc.min(axis=1)              # (batch, N_det, N_samples): weaker muon per det
            both[s:e] = weaker.max(axis=1)        # best detector
            decay_pos[s:e] = data["decay_positions"]
            prod_pos[s:e] = data["production_positions"]
            decay_prob[s:e] = data["decay_probability"]
            nhnl[s:e] = data["N_HNLs_per_muon"]
            if "cherenkov_weights" in data:
                weights[s:e] = data["cherenkov_weights"]

        out[m_N] = dict(photon_counts=ph, both_metric=both, decay_positions=decay_pos,
                        production_positions=prod_pos, decay_probability=decay_prob,
                        N_HNLs_per_muon=nhnl, cherenkov_weights=weights, n_det=n_det)

    if n_missing:
        print(f"  {n_missing} batch files missing out of {len(MASSES) * N_U2_BATCHES}")
        print("  Re-run with: sbatch --array=" + ",".join(map(str, missing_array_ids))
              + " cluster/submit_scan_dimuon.sh")

    return out, U2_RANGE, N_samples


def apply_cuts(raw, U2_range, N_samples, min_photons, max_transverse=None):
    masses = sorted(raw.keys())
    n_m, n_U2 = len(masses), len(U2_range)
    events = np.zeros((n_U2, n_m))
    efficiency = np.zeros((n_U2, n_m))
    events_both = np.zeros((n_U2, n_m))        # both-muon-tagged (background-free) sample
    efficiency_both = np.zeros((n_U2, n_m))

    for im, m_N in enumerate(masses):
        d = raw[m_N]
        for iU2 in range(n_U2):
            ph = d["photon_counts"][iU2]
            both = d["both_metric"][iU2]
            w = d["cherenkov_weights"][iU2]
            wdecay = d["decay_probability"][iU2]
            tcut = np.ones_like(ph, dtype=bool)
            if max_transverse is not None:
                rt = transverse_distance(d["decay_positions"][iU2],
                                         d["production_positions"][iU2])
                tcut = rt <= max_transverse

            sel = (ph >= min_photons) & tcut
            eff = np.sum(wdecay[sel]) * w / N_samples
            events[iU2, im] = d["N_HNLs_per_muon"][iU2] * eff * N_muon_decays
            efficiency[iU2, im] = eff

            sel_both = (both >= min_photons) & tcut
            eff_b = np.sum(wdecay[sel_both]) * w / N_samples
            events_both[iU2, im] = d["N_HNLs_per_muon"][iU2] * eff_b * N_muon_decays
            efficiency_both[iU2, im] = eff_b

    return {"m_N": np.array(masses), "U2": U2_range,
            "events": events, "efficiency": efficiency,
            "events_both": events_both, "efficiency_both": efficiency_both}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--min-pe", type=float, default=20.0,
                   help="Single-pixel PE threshold (default 20, Trinity "
                        "Demonstrator operating point; use 12 for the "
                        "optimistic case).")
    p.add_argument("--pde", type=float, default=0.40)
    p.add_argument("--max-transverse", type=float, default=None,
                   help="Transverse-distance cut from the beamline [m] (the discriminator).")
    p.add_argument("--scan-dir", type=str, default="data/scan_results_balloon_dimuon")
    p.add_argument("--output", type=str,
                   default="data/scan_results_balloon_dimuon/combined.npz")
    args = p.parse_args()

    min_photons = args.min_pe / args.pde
    print(f"Loading dimuon scan results from {args.scan_dir} ...")
    raw, U2_range, N_samples = load_raw_results(args.scan_dir)
    print(f"  {len(raw)} masses, {len(U2_range)} U2 points, N_samples={N_samples}")
    print(f"Applying min_pe={args.min_pe} (={min_photons:.1f} photons at PDE={args.pde}), "
          f"max_transverse={args.max_transverse}")
    res = apply_cuts(raw, U2_range, N_samples, min_photons, args.max_transverse)
    print(f"  max events: single-tag={res['events'].max():.3g}  "
          f"both-tag (bkg-free)={res['events_both'].max():.3g}")
    np.savez(args.output, **res)
    print(f"Saved to {args.output}")
