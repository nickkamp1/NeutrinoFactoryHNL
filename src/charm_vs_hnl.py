"""Overlay the charm-background and HNL-signal dimuon discriminators.

The charm dimuon background (nu_mu -> CC charm DIS -> mu + Hadrons + D, D -> mu)
and the HNL signal (N4 -> nu mu mu) both give a muon pair, but they differ in
two handles this module plots:

  1. mu-mu OPENING ANGLE.  Detected charm pairs sit near ~1.3 deg, while the HNL
     pair opening angle scales with m_N/E_N (~0.25 deg at 5 GeV rising to ~1 deg
     by 50 GeV) -- so light HNLs are much more collimated than charm.
  2. HADRONIC SHOWER.  Charm DIS produces a hadronic shower; the leptonic HNL
     decay produces NONE.  Plotted as the number of DETECTED hadronic-shower
     Cherenkov photons -- but note only ~half the charm shower exceeds the ~30
     photon threshold, so a hadronic veto is only ~50% efficient at 12 PE.

Plots are in PHYSICAL EVENT RATE (expected events over the N_muon_decays
exposure).  Both samples use the same detection tag (default: both muons above
the photoelectron threshold, each on its own best detector) and physical-rate
weights.  The charm rate has no free parameter; each HNL mass is shown at its
best-detected (peak-sensitivity) U2 within the scanned grid.  Weights:

  * HNL signal (per data/muon_image_spread_<m>.npz), matching Balloon.ipynb:
        w = weight * BR_mumu * interaction_probability * N_muon_decays / n_events
  * charm background (per data/scan_results_balloon_charm/charm_bkg_seed_*.npz):
        w = interaction_weights * N_nu_per_muon * cherenkov_weight
            * N_muon_decays / N_samples / n_seeds   (n_seeds = files combined)

This module only reads the saved .npz outputs (numpy + matplotlib, no SIREN), so
it runs locally once the charm files are rsync'd down from the cluster.

Typical use
-----------
    from src.charm_vs_hnl import load_charm, load_hnl, plot_comparison
    charm = load_charm()                        # combines all seeds
    hnls = {m: load_hnl(m) for m in (5, 10)}
    plot_comparison(charm, hnls, output="figures/charm_vs_hnl.png")

or from the project root:
    python -m src.charm_vs_hnl --masses 5 10 --output figures/charm_vs_hnl.png
"""
import os
import glob
import argparse

import numpy as np

from src.constants import N_muon_decays

# Detection threshold.  The analysis operating point for this study is 12 PE
# (the optimistic Trinity threshold; the dimuon sensitivity contours were made at
# 12 PE / 0.4 = 30 photons), NOT the 20 PE demonstrator point.
SIPM_PDE = 0.40
MIN_PHOTOELECTRONS = 12.0
MIN_PHOTONS_DEFAULT = MIN_PHOTOELECTRONS / SIPM_PDE   # = 30 photons

CHARM_DIR_DEFAULT = "data/scan_results_balloon_charm"
HNL_DIR_DEFAULT = "data/scan_results_balloon_hnl"


# --------------------------------------------------------------------------- #
# Loading + tagging
# --------------------------------------------------------------------------- #
def load_charm(charm_dir=CHARM_DIR_DEFAULT, min_photons=MIN_PHOTONS_DEFAULT,
               tag="both_detected"):
    """Combine the per-seed charm files and return the tagged-event arrays.

    tag : {"both_detected", "same_detector", "single"}
        "both_detected" (default) -- each muon above threshold on its own best
            detector (they may be on different stations); the natural sample for
            measuring a mu-mu opening angle across the array, and higher-stats.
        "same_detector" -- both muons above threshold on the SAME detector
            (matches the analysis both-muon tag; opening angle = on-camera sep).
        "single" -- >=1 muon above threshold (loosest).

    Returns a dict with per-event arrays (opening_deg, had_E, mu1_E, mu2_E,
    nu_energy) and matching physical-rate ``weight`` for tagged events, plus
    scalars ``n_seeds`` and ``rate`` (total expected tagged events).
    """
    files = sorted(glob.glob(os.path.join(charm_dir, "charm_bkg_seed_*.npz")))
    if not files:
        raise FileNotFoundError(f"no charm files in {charm_dir}")
    n_seeds = len(files)

    keep = {k: [] for k in ("opening_deg", "oncam_sep_deg", "had_E",
                            "had_photons", "mu1_E", "mu2_E", "nu_energy", "weight")}
    for f in files:
        d = np.load(f)
        mpc = d["mu_photon_counts"]          # (2, N_det, N_samples)
        hpc = d["hadronic_photon_counts"]    # (N_det, N_samples)
        N_samples = int(d["N_samples"])
        both_thr = ((mpc[0].max(axis=0) >= min_photons)
                    & (mpc[1].max(axis=0) >= min_photons))
        if tag == "kinematic":
            tagged = np.ones(mpc.shape[-1], dtype=bool)   # no detection cut
        elif tag == "same_detector":
            # both muons above threshold on ONE camera (measurable on-cam sep).
            # Use the recorded same_detector flag (brightest coincide) so the
            # selection matches where oncam_sep_deg is defined.
            tagged = np.asarray(d["same_detector"], bool) & both_thr
        elif tag == "single":
            tagged = (mpc.max(axis=(0, 1)) >= min_photons)
        else:  # both_detected: each muon over threshold on its own best detector
            tagged = both_thr

        # detected hadronic-shower photons = brightest hadronic signal on any
        # detector (the best case for a hadronic veto).
        had_photons = np.asarray(hpc).max(axis=0)            # (N_samples,)

        # physical rate weight per event (see module docstring), split /n_seeds
        w = (d["interaction_weights"] * float(d["N_nu_per_muon"])
             * float(d["cherenkov_weight"]) * N_muon_decays
             / N_samples / n_seeds)

        sel = tagged & np.isfinite(d["opening_deg"])
        keep["opening_deg"].append(d["opening_deg"][sel])
        keep["oncam_sep_deg"].append(np.asarray(d["oncam_sep_deg"])[sel])
        keep["had_E"].append(d["had_E"][sel])
        keep["had_photons"].append(had_photons[sel])
        keep["mu1_E"].append(d["mu1_E"][sel])
        keep["mu2_E"].append(d["mu2_E"][sel])
        keep["nu_energy"].append(d["nu_energy"][sel])
        keep["weight"].append(w[sel])

    out = {k: np.concatenate(v) if v else np.array([]) for k, v in keep.items()}
    out["n_seeds"] = n_seeds
    out["rate"] = float(out["weight"].sum())
    return out


def load_hnl(mass, hnl_dir=HNL_DIR_DEFAULT, min_photons=MIN_PHOTONS_DEFAULT,
             tag="both_detected", u2min=None, u2max=None, select_u2="best",
             file_tag=""):
    """Load one HNL muon_image_spread file and return the tagged-event arrays.

    The per-event physical-rate weight is built per (m_N, U2) grid point
    (Balloon.ipynb cell 15 convention),

        w = weight * BR_mumu * interaction_probability * N_muon_decays / n_events

    select_u2 : {"best", "pool"}
        The HNL rate depends on the (unknown) mixing U2, so a sum over the whole
        grid is NOT a physical rate.  "best" (default) keeps only the single U2
        grid point with the largest tagged rate -- i.e. the most-detectable /
        peak-sensitivity mixing for that mass -- so the plotted rate is a real
        physical yield.  "pool" sums the grid (shape only; use with normalize).
    tag : {"both_detected", "same_detector", "kinematic"} -- see ``load_charm``.
    u2min/u2max : optional bounds restricting the grid before selection.

    Returns a dict with opening_deg, E_mu1, E_mu2, hnl_energy, U2, weight, plus
    scalars ``m_N``, ``rate``, and ``U2_sel`` (the chosen U2, or None if pooled).
    The HNL leptonic decay has NO hadronic shower.
    """
    path = os.path.join(hnl_dir, f"muon_image_spread_{file_tag}{mass:.0f}.npz")
    d = np.load(path, allow_pickle=True)

    # cast the tag arrays to bool: a multi-U2 file with any 0-event grid point
    # concatenates bool with an empty float array -> float64, breaking `&`.
    if tag == "kinematic":
        both = np.ones(len(d["opening_deg"]), dtype=bool)   # no detection cut
    else:
        both = (d["n_ph1"] >= min_photons) & (d["n_ph2"] >= min_photons)
        tag_arr = d["same_detector"] if tag == "same_detector" else d["both_detected"]
        both = both & np.asarray(tag_arr, dtype=bool)

    # per-event physical-rate weight, looked up per (m_N, U2) grid point
    ev_mN, ev_U2 = np.asarray(d["m_N"]), np.asarray(d["U2"])
    gk_mN = np.ravel(d["meta_m_N"]); gk_U2 = np.ravel(d["meta_U2"])
    g_ip = np.ravel(d["meta_interaction_probability"])
    g_br = np.ravel(d["meta_BR_mumu"]); g_ne = np.ravel(d["meta_n_events"])
    w_all = np.zeros(len(ev_mN))
    for gi in range(len(gk_mN)):
        m = np.isclose(ev_mN, gk_mN[gi], atol=0) & \
            np.isclose(ev_U2, gk_U2[gi], rtol=1e-6, atol=0)
        if g_ne[gi] > 0:
            w_all[m] = (d["weight"][m] * g_br[gi] * g_ip[gi]
                        * N_muon_decays / g_ne[gi])

    sel = both & np.isfinite(d["opening_deg"])
    if u2min is not None:
        sel = sel & (ev_U2 >= u2min)
    if u2max is not None:
        sel = sel & (ev_U2 <= u2max)

    U2_sel = None
    if select_u2 == "best":
        # pick the U2 grid point with the largest tagged expected rate
        best_rate, best_u2 = -1.0, None
        for u2 in np.unique(ev_U2[sel]) if sel.any() else []:
            r = w_all[sel & (ev_U2 == u2)].sum()
            if r > best_rate:
                best_rate, best_u2 = r, u2
        if best_u2 is not None:
            sel = sel & (ev_U2 == best_u2)
            U2_sel = float(best_u2)

    out = dict(
        opening_deg=d["opening_deg"][sel],
        oncam_sep_deg=np.asarray(d["oncam_sep_deg"])[sel],
        E_mu1=d["E_mu1"][sel], E_mu2=d["E_mu2"][sel],
        hnl_energy=d["hnl_energy"][sel],
        U2=ev_U2[sel], weight=w_all[sel], m_N=float(mass), U2_sel=U2_sel,
    )
    out["rate"] = float(out["weight"].sum())
    return out


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def _apply_style():
    import matplotlib.pyplot as plt
    style = os.path.join(os.getcwd(), "figures.mplstyle")
    if os.path.exists(style):
        try:
            plt.style.use(style)
        except Exception:
            pass


def plot_comparison(charm, hnls, min_photons=MIN_PHOTONS_DEFAULT,
                    normalize=False, tag="same_detector",
                    xvar="oncam_sep_deg", output=None, axes=None):
    """Two-panel overlay in PHYSICAL EVENT RATE (expected events over the nominal
    N_muon_decays exposure).

    Left  -- mu-mu opening angle: one curve per HNL mass, plus TWO charm curves:
             all tagged charm, and charm WITHOUT a detectable hadronic shower
             (had_photons < threshold) -- the charm that survives a hadronic veto.
    Right -- number of detected hadronic-shower photons for charm, with the
             detection threshold marked (HNL signal has none).

    charm : dict from ``load_charm`` (needs had_photons).
    hnls : dict {m_N: dict from ``load_hnl``} (each at its best-detected U2).
    normalize : bool
        False (default) -> physical expected events / bin.  True -> unit area.
    """
    import matplotlib.pyplot as plt
    _apply_style()

    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    else:
        fig = axes[0].figure
    ax_open, ax_had = axes

    op_bins = np.logspace(np.log10(0.02), np.log10(20.0), 40)   # deg
    ylabel = ("fraction of events" if normalize
              else "expected events / bin")
    PIXEL_DEG = 0.3
    is_oncam = (xvar == "oncam_sep_deg")
    xlabel = (r"on-camera $\mu\mu$ separation [deg]" if is_oncam
              else r"true $\mu\mu$ opening angle [deg]")

    def _hist(ax, x, w, bins, **kw):
        x = np.asarray(x)
        m = np.isfinite(x) & (x > 0)
        if m.sum() == 0:
            return
        ax.hist(x[m], bins=bins, weights=np.asarray(w)[m], histtype="step",
                lw=1.8, density=normalize, **kw)

    # --- opening / separation: HNL masses + two charm curves ---
    for m_N in sorted(hnls):
        h = hnls[m_N]
        u2lab = f", $U^2$={h['U2_sel']:.0e}" if h.get("U2_sel") else ""
        _hist(ax_open, h[xvar], h["weight"], op_bins,
              label=f"HNL $m_N$={m_N:.0f} GeV{u2lab} ({h['rate']:.2g})")
    _hist(ax_open, charm[xvar], charm["weight"], op_bins,
          label=f"charm, all ({charm['rate']:.2g})", color="k", ls="-")
    # charm that survives a hadronic veto: no detectable hadronic shower
    no_had = np.asarray(charm["had_photons"]) < min_photons
    rate_nohad = float(np.asarray(charm["weight"])[no_had].sum())
    _hist(ax_open, np.asarray(charm[xvar])[no_had],
          np.asarray(charm["weight"])[no_had], op_bins,
          label=f"charm, no had. shower ({rate_nohad:.2g})", color="k", ls="--")
    ax_open.set_xscale("log")
    if not normalize:
        ax_open.set_yscale("log")
    ax_open.set_xlabel(xlabel)
    ax_open.set_ylabel(ylabel)
    # For the measurable on-camera separation the pixel maps DIRECTLY: two spots
    # resolve only when separated by more than ~1 pixel (0.3 deg).
    if is_oncam:
        ax_open.axvspan(op_bins[0], PIXEL_DEG, color="gray", alpha=0.12)
        ax_open.axvline(PIXEL_DEG, color="gray", ls=":", lw=1)
        ax_open.text(PIXEL_DEG * 0.95, 0.02, "unresolved (<1 pixel, 0.3$\\degree$)",
                     rotation=90, fontsize=6, color="gray", ha="right", va="bottom",
                     transform=ax_open.get_xaxis_transform())
    ax_open.legend(fontsize=6.5, loc="upper left")
    ax_open.set_title(r"$\mu\mu$ separation on camera" if is_oncam
                      else r"true $\mu\mu$ opening angle", fontsize=10)

    # --- number of detected hadronic-shower photons: charm only ---
    ph_bins = np.logspace(0, np.log10(1e4), 40)        # photons
    _hist(ax_had, charm["had_photons"], charm["weight"], ph_bins,
          label="charm bkg", color="k")
    ax_had.axvline(min_photons, color="C3", ls=":", lw=1.5)
    ax_had.text(min_photons * 1.1, 0.92, f"threshold\n{min_photons:.0f} ph",
                fontsize=7, color="C3", transform=ax_had.get_xaxis_transform(),
                va="top")
    ax_had.set_xscale("log")
    if not normalize:
        ax_had.set_yscale("log")
    ax_had.set_xlabel("detected hadronic-shower photons")
    ax_had.set_ylabel(ylabel)
    ax_had.set_title(r"hadronic shower (HNL: none)", fontsize=10)
    ax_had.legend(fontsize=8)

    fig.suptitle(f"charm dimuon background vs HNL signal   "
                 f"(tag: {tag}, {min_photons * SIPM_PDE:.0f} PE, "
                 f"exposure $N_\\mu$={N_muon_decays:.0e})", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    if output:
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        fig.savefig(output, dpi=150, bbox_inches="tight")
        print(f"[charm_vs_hnl] saved {output}")
    return fig, axes


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--masses", type=float, nargs="+", default=[5, 10],
                   help="HNL masses to overlay (need data/muon_image_spread_<m>.npz)")
    p.add_argument("--charm-dir", default=None,
                   help="override charm dir (else derived from --detector)")
    p.add_argument("--hnl-dir", default=HNL_DIR_DEFAULT)
    p.add_argument("--detector", default="all",
                   help="'all' = full 9-detector aggregate (default), or a "
                        "detector index (e.g. 2/5/8 = central 20/50/100 km) for "
                        "the single-detector view")
    p.add_argument("--min-pe", type=float, default=MIN_PHOTOELECTRONS)
    p.add_argument("--pde", type=float, default=SIPM_PDE)
    p.add_argument("--normalize", action="store_true",
                   help="plot unit-area shapes instead of physical event rate (default)")
    p.add_argument("--tag", choices=["both_detected", "same_detector", "single",
                                     "kinematic"],
                   default="same_detector",
                   help="event selection (default: same_detector -- both muons "
                        "above threshold on ONE camera, where the on-camera "
                        "separation is measurable)")
    p.add_argument("--xvar", choices=["oncam_sep_deg", "opening_deg"],
                   default="oncam_sep_deg",
                   help="x-axis: measurable on-camera separation (default) or "
                        "true lab opening angle")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    min_photons = args.min_pe / args.pde
    # detector toggle: 'all' -> aggregate files; index i -> _det<i> files
    if args.detector == "all":
        charm_dir = args.charm_dir or CHARM_DIR_DEFAULT
        file_tag = ""
    else:
        di = int(args.detector)
        charm_dir = args.charm_dir or f"data/scan_results_balloon_charm_det{di}"
        file_tag = f"det{di}_"
    charm = load_charm(charm_dir, min_photons, tag=args.tag)
    hnl_tag = args.tag if args.tag in ("same_detector", "kinematic") else "both_detected"
    hnls = {}
    for m in args.masses:
        try:
            hnls[m] = load_hnl(m, args.hnl_dir, min_photons, tag=hnl_tag,
                               file_tag=file_tag)
        except FileNotFoundError:
            print(f"[charm_vs_hnl] skipping m_N={m:.0f}: no muon_image_spread file")

    def _med(x):
        x = np.asarray(x)
        return np.median(x) if x.size else float("nan")

    no_had = np.asarray(charm["had_photons"]) < min_photons
    rate_nohad = float(np.asarray(charm["weight"])[no_had].sum())
    print(f"[charm_vs_hnl] tag={args.tag}")
    print(f"[charm_vs_hnl] charm: {charm['n_seeds']} seeds, rate={charm['rate']:.3g}, "
          f"median opening={_med(charm['opening_deg']):.2f} deg, "
          f"median had-photons={_med(charm['had_photons']):.0f}")
    print(f"[charm_vs_hnl]   charm surviving hadronic veto (<{min_photons:.0f} ph) = "
          f"{rate_nohad:.3g}  ({100*rate_nohad/charm['rate']:.1f}% of charm)")
    for m, h in hnls.items():
        u2 = h.get("U2_sel")
        print(f"[charm_vs_hnl] HNL m_N={m:.0f}: rate={h['rate']:.3g} "
              f"(best U2={u2:.1e})  median opening={_med(h['opening_deg']):.2f} deg"
              if u2 else
              f"[charm_vs_hnl] HNL m_N={m:.0f}: rate={h['rate']:.3g}  "
              f"median opening={_med(h['opening_deg']):.2f} deg")

    if args.output:
        plot_comparison(charm, hnls, min_photons=min_photons,
                        normalize=args.normalize, tag=args.tag,
                        xvar=args.xvar, output=args.output)


if __name__ == "__main__":
    _main()
