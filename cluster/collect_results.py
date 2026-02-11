"""
Collect scan results from individual batch jobs and apply photon thresholds.

Usage:
    python collect_results.py [--min-photons 5] [--output results.npz]
"""
import os
import sys
import argparse
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

from src.constants import N_muon_decays

MASSES = np.array([5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90])
U2_RANGE = np.logspace(-14, -7, 100)
U2_BATCH_SIZE = 5
N_U2_BATCHES = int(np.ceil(len(U2_RANGE) / U2_BATCH_SIZE))


def load_raw_results(scan_dir="data/scan_results"):
    """Load all per-batch scan files and assemble into full arrays."""
    # Discover N_samples from first available file
    N_samples = None
    for m_N in MASSES:
        fpath = os.path.join(scan_dir, f"scan_mN_{m_N:.0f}_u2batch_000.npz")
        if os.path.exists(fpath):
            N_samples = int(np.load(fpath)["N_samples"])
            break
    if N_samples is None:
        raise FileNotFoundError("No scan result files found")

    photon_counts = {}
    N_HNLs_per_muon = {}
    n_missing = 0

    for m_N in MASSES:
        ph_all = np.zeros((len(U2_RANGE), N_samples))
        nhnl_all = np.zeros(len(U2_RANGE))
        mass_complete = True

        for b in range(N_U2_BATCHES):
            fpath = os.path.join(scan_dir, f"scan_mN_{m_N:.0f}_u2batch_{b:03d}.npz")
            if not os.path.exists(fpath):
                print(f"  WARNING: missing {fpath}")
                n_missing += 1
                mass_complete = False
                continue
            data = np.load(fpath)
            u2_start = int(data["u2_start"])
            u2_end = int(data["u2_end"])
            ph_all[u2_start:u2_end] = data["photon_counts"]
            nhnl_all[u2_start:u2_end] = data["N_HNLs_per_muon"]

        photon_counts[m_N] = ph_all
        N_HNLs_per_muon[m_N] = nhnl_all

    if n_missing > 0:
        print(f"  {n_missing} batch files missing out of {len(MASSES) * N_U2_BATCHES}")

    return photon_counts, N_HNLs_per_muon, U2_RANGE, N_samples


def apply_threshold(photon_counts, N_HNLs_per_muon, U2_range, N_samples,
                    min_photons=5):
    """Derive events/efficiency/mean_photons for a given threshold."""
    available_masses = sorted(photon_counts.keys())
    n_m = len(available_masses)
    n_U2 = len(U2_range)

    events = np.zeros((n_U2, n_m))
    efficiency = np.zeros((n_U2, n_m))
    mean_photons = np.zeros((n_U2, n_m))

    for im, m_N in enumerate(available_masses):
        for iU2 in range(n_U2):
            ph = photon_counts[m_N][iU2]
            detected = ph >= min_photons
            eff = np.sum(detected) / N_samples
            mph = np.mean(ph[detected]) if np.any(detected) else 0.0
            n_ev = N_HNLs_per_muon[m_N][iU2] * eff * N_muon_decays

            events[iU2, im] = n_ev
            efficiency[iU2, im] = eff
            mean_photons[iU2, im] = mph

    return {
        "m_N": np.array(available_masses),
        "U2": U2_range,
        "events": events,
        "efficiency": efficiency,
        "mean_photons": mean_photons,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-photons", type=float, default=5)
    parser.add_argument("--output", type=str, default="data/scan_results/combined.npz")
    args = parser.parse_args()

    print("Loading raw results...")
    ph_counts, n_hnl, U2_range, N_samples = load_raw_results()
    print(f"  Loaded {len(ph_counts)} masses, {len(U2_range)} U2 points, N_samples={N_samples}")

    print(f"Applying threshold min_photons={args.min_photons}...")
    results = apply_threshold(ph_counts, n_hnl, U2_range, N_samples,
                              min_photons=args.min_photons)

    np.savez(args.output, **results)
    print(f"Saved to {args.output}")
