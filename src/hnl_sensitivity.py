"""HNL sensitivity in (m_N, U2) from a 2D binned likelihood over the charm
discriminators: the on-camera mu-mu separation and the detected hadronic-shower
photon count.

Method
------
The HNL signal kinematics/imaging are U2-INDEPENDENT (they depend on m_N and the
decay POSITION, not the mixing), so a single reference sample per mass -- run
with uniform decay-distance sampling (``uniform_gen=True``) -- is reweighted to
any (m_N, U2) with ``HNLFluxGeometry.compute_reweighted_signal_at_satellite``.
Only the event weights change; the per-event (on-camera separation) carries over.

For each (m_N, U2):
  * reweight the reference HNL events -> per-event expected-signal weight;
  * bin the tagged signal in the 2D plane (on-camera separation x hadronic
    photons).  The HNL decay is LEPTONIC, so all signal sits in the
    "no hadronic shower" column (had_photons = 0);
  * the charm background (U2-INDEPENDENT) is binned once in the same plane;
  * the median expected significance is the binned Asimov value

        Z = sqrt( 2 * sum_bins [ (s+b) ln(1 + s/b) - s ] )   (b > 0)

    and the sensitivity contour is where Z crosses a chosen level (e.g. 2).

The 2D likelihood buys two things over a plain count: the hadronic axis removes
the ~half of charm with a detectable shower, and the separation axis then
distinguishes the collimated (often <1-pixel, "single blob") HNL pair from the
wider-separation charm survivors.

Runs locally: HNLFluxGeometry (the reweighter) is pure numpy -- no SIREN needed.

Typical use
-----------
    from src.hnl_sensitivity import SensitivityModel
    m = SensitivityModel(detector=8, masses=[5,10,20,30,50])
    grid = m.significance_grid()            # Z(m_N, U2)
    m.plot_contours(output="figures/sensitivity_det8.png")
"""
import os
import glob

import numpy as np

from src.constants import N_muon_decays

from scipy.stats import poisson,norm

# thresholds (12 PE / 0.4), matching charm_vs_hnl
SIPM_PDE = 0.40
MIN_PHOTOELECTRONS = 12.0
MIN_PHOTONS_DEFAULT = MIN_PHOTOELECTRONS / SIPM_PDE   # 30 photons
PIXEL_DEG = 0.15

# nominal beam-dump config (matches the sims)
E_MU, DUMP_DEPTH, DUMP_ANGLE = 5000.0, 100.0, 1.53


def _geom():
    """Pure-numpy HNLFluxGeometry for reweighting (no SIREN)."""
    from src.balloon import HNLFluxGeometry
    return HNLFluxGeometry(E_mu=E_MU, dump_depth=DUMP_DEPTH, dump_angle=DUMP_ANGLE)


# --------------------------------------------------------------------------- #
# Binning of the 2D discriminator plane
# --------------------------------------------------------------------------- #
def default_bins(min_photons=MIN_PHOTONS_DEFAULT):
    """(sep_edges [deg], had_edges [photons]).

    Separation: an 'unresolved' bin below one pixel (spots merge), then log bins.
    Hadronic photons: 'no detectable shower' [0, threshold), then above-threshold
    bins (which carry no signal but bound the charm shower).
    """
    sep_edges = np.concatenate([#[0.0, PIXEL_DEG],
                                np.logspace(np.log10(PIXEL_DEG), np.log10(6.0), 7)[1:]])
    had_edges = np.array([0.0, min_photons])#, 3 * min_photons, 10 * min_photons, np.inf])
    return sep_edges, had_edges


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_charm_2d(detector, R4=False, min_photons=MIN_PHOTONS_DEFAULT, data_dir="data"):
    """Charm background per-event (sep, had_photons, weight) for tagged events on
    one detector.  Returns arrays; weights are absolute expected events."""
    cdir = os.path.join(data_dir, f"scan_results_balloon_charm_det{detector}_{'R4' if R4 else ''}")
    files = sorted(glob.glob(os.path.join(cdir, "charm_bkg_seed_*.npz")))
    if not files:
        raise FileNotFoundError(f"no charm files in {cdir}")
    n_seeds = len(files)
    sep, had, w = [], [], []
    for f in files:
        d = np.load(f)
        mpc = d["mu_photon_counts"]          # (2, 1, N)
        hpc = d["hadronic_photon_counts"]    # (1, N)
        N_samples = int(d["N_samples"])
        # both muons above threshold on this (single) detector
        tagged = ((mpc[0, 0] >= min_photons) & (mpc[1, 0] >= min_photons)
                  & np.asarray(d["same_detector"], bool)
                  & np.isfinite(d["oncam_sep_deg"]))
        wev = (d["interaction_weights"] * float(d["N_nu_per_muon"])
               * float(d["cherenkov_weight"]) * N_muon_decays
               / N_samples / n_seeds)
        sep.append(np.asarray(d["oncam_sep_deg"])[tagged])
        had.append(np.asarray(hpc[0])[tagged])
        w.append(wev[tagged])
    return (np.concatenate(sep), np.concatenate(had), np.concatenate(w))


def load_hnl_ref(mass, detector, R4=False, data_dir="data"):
    """HNL reference events (uniform_gen) for one mass/detector: the reweight
    inputs + on-camera separation for tagged (both-muon, same-camera) events."""
    path = os.path.join(data_dir, f"scan_results_balloon_hnl/muon_image_spread_det{detector}_{'R4_' if R4 else ''}{mass:.0f}.npz")
    d = np.load(path, allow_pickle=True)
    return d


# --------------------------------------------------------------------------- #
# Reweighting a reference sample to (m_N, U2)
# --------------------------------------------------------------------------- #
def reweight_hnl(geom, ref, m_N, U2, min_photons=MIN_PHOTONS_DEFAULT):
    """Return (sep, weight) for the reference HNL events reweighted to (m_N, U2).

    General importance reweight using the stored proposal density
    ``sampling_pdf`` q(decay_dist): the decay-position weight for the target is
    decay_pos_probability = p_target(decay_dist) / q(decay_dist), where p_target
    is the truncated-exponential decay pdf at the target decay length.  This
    works for ANY proposal (uniform, log-uniform, mixture), unlike
    compute_reweighted_signal_at_satellite's uniform-only branch.

    Applies the same tag as charm (both muons above threshold on the shared
    camera).  Signal has no hadronic shower (had_photons = 0 by construction).
    """
    from src.xs_and_decays import HNL_decay_length
    E = np.asarray(ref["hnl_energy"], float)
    d = np.asarray(ref["decay_dist"], float)
    d_max = np.asarray(ref["d_max"], float)
    # proposal density q(decay_dist).  Newer files store it directly; older
    # (pre-mixture) uniform/trunc_exp files don't, so reconstruct it from
    # decay_pos_probability = p_ref/q  =>  q = p_ref/decay_pos_probability.
    if "sampling_pdf" in ref:
        q = np.asarray(ref["sampling_pdf"], float)
    else:
        Lref = np.asarray(ref["decay_length"], float)
        dpp_ref = np.asarray(ref["decay_pos_probability"], float)
        with np.errstate(divide="ignore", invalid="ignore"):
            p_ref = np.where((Lref > 0) & (d_max > 0),
                             np.exp(-d / Lref) / (Lref * (1.0 - np.exp(-d_max / Lref))),
                             0.0)
            q = np.where(dpp_ref > 0, p_ref / dpp_ref, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        L = HNL_decay_length(m_N, U2, E)                      # target decay length
        decay_prob = np.where(d_max > 0, 1.0 - np.exp(-d_max / L), 0.0)
        p_target = np.where(decay_prob > 0,
                            np.exp(-d / L) / (L * (1.0 - np.exp(-d_max / L))), 0.0)
        dpp = np.where((decay_prob > 0) & (q > 0), p_target / q, 0.0)
    ip = float(geom.compute_weighted_production_rate(m_N, U2)[0])
    br = float(np.ravel(ref["meta_BR_mumu"])[0])
    cw = float(np.ravel(ref["meta_cherenkov_weight"])[0])
    # Normalize by N_samples (beam MC events THROWN), NOT meta_n_events (the
    # imaged/valid subset): the cherenkov_weight cw already rescales the imaged
    # sample up to the full valid sample, and the Monte-Carlo estimator of the
    # expected event rate divides by the total number thrown.  Dividing by
    # meta_n_events instead over-counts by N_samples/n_events (~25x), which
    # matches Balloon.ipynb's sig_events_extrapolated to ~10% once corrected.
    # Older files predate meta_N_samples; they were all generated with the
    # run_muon_image_spread.py driver at N_samples=1e6, so fall back to that.
    if "meta_N_samples" in ref:
        ns = float(np.ravel(ref["meta_N_samples"])[0])
    else:
        ns = 1.0e6
    w = (decay_prob * dpp * ip * br * N_muon_decays * cw / ns
         if ns > 0 else np.zeros_like(d))
    both = ((np.asarray(ref["n_ph1"]) >= min_photons)
            & (np.asarray(ref["n_ph2"]) >= min_photons)
            & np.asarray(ref["same_detector"], bool)
            & np.isfinite(ref["oncam_sep_deg"]))
    return np.asarray(ref["oncam_sep_deg"])[both], np.asarray(w)[both]


# --------------------------------------------------------------------------- #
# Significance
# --------------------------------------------------------------------------- #
def save_all_contours(path, detectors=(2, 5, 8), masses=(5, 6, 7, 8, 9, 10, 12,
                      14, 16, 20, 25, 30), levels=(2.0, 5.0),
                      min_photons=MIN_PHOTONS_DEFAULT, data_dir="data", key="Z2d"):
    """Save contours for ALL detectors into ONE .npz, Balloon.ipynb-style: keys
    '<detector>_<photon_threshold>_<level>' (+ '..._count') -> (m_N, U2) points.
    Returns the merged dict."""
    merged = {}
    for det in detectors:
        m = SensitivityModel(detector=det, masses=masses, min_photons=min_photons,
                             data_dir=data_dir)
        g = m.significance_grid()
        M, U2, Z = m.significance_map(g, key=key)
        _, _, Zc = m.significance_map(g, key="Zcount")
        for level in levels:
            k = f"{det}_{min_photons:.1f}_{level:.1f}"
            merged[k] = m._contour_segments(M, U2, Z, level)
            merged[k + "_count"] = m._contour_segments(M, U2, Zc, level)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, **merged)
    print(f"[hnl_sensitivity] saved {len(merged)} contours to {path}")
    return merged


def asimov_Z(s, b):
    """Binned Asimov median significance, sum over bins with b > 0.
    Z = sqrt( 2 sum [ (s+b) ln(1 + s/b) - s ] )."""
    s = np.asarray(s, float); b = np.asarray(b, float)
    m = b > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        term = (s[m] + b[m]) * np.log1p(s[m] / b[m]) - s[m]
    return float(np.sqrt(2.0 * np.maximum(term, 0).sum()))


class SensitivityModel:
    """1D-likelihood HNL sensitivity for one detector."""

    def __init__(self, detector=8, masses=(5, 10, 20, 30, 50),
                 min_photons=MIN_PHOTONS_DEFAULT, data_dir="data",
                 U2_grid=None, R4=False):
        self.detector = detector
        self.masses = list(masses)
        self.min_photons = min_photons
        self.data_dir = data_dir
        self.U2_grid = (np.logspace(-14, -8, 61) if U2_grid is None
                        else np.asarray(U2_grid, float))
        self.sep_edges, self.had_edges = default_bins(min_photons)
        self.geom = _geom()
        # charm background (fixed), 2D histogram
        cs, ch, cw = load_charm_2d(detector, R4=R4, min_photons=min_photons, data_dir=data_dir)
        self.B, _ = np.histogram(cs, self.sep_edges, weights=(cw*(ch<min_photons)))
        self.B_total = float(cw.sum())
        self.refs = {m: load_hnl_ref(m, detector, R4=R4, data_dir=data_dir) for m in self.masses}

    def signal_hist(self, m_N, U2):
        """1D signal histogram."""
        sep, w = reweight_hnl(self.geom, self.refs[m_N], m_N, U2, self.min_photons)
        S, _ = np.histogram(sep, self.sep_edges, weights=w)
        return S, float(S.sum())

    def significance_grid(self):
        """Z(m_N, U2) for the 1D likelihood and for a plain count (no
        discriminator).  Returns dict of (n_mass, n_U2) arrays."""
        Z2d = np.zeros((len(self.masses), len(self.U2_grid)))
        Zcount = np.zeros_like(Z2d)
        Stot = np.zeros_like(Z2d)
        BackgroundFree = np.zeros_like(Z2d)
        for i, m in enumerate(self.masses):
            for j, U2 in enumerate(self.U2_grid):
                S, s_tot = self.signal_hist(m, U2)
                Z2d[i, j] = asimov_Z(S, self.B)
                # counting: total tagged S vs total tagged B (single bin)
                Zcount[i, j] = asimov_Z([s_tot], [self.B_total])
                Stot[i, j] = s_tot
                if s_tot < 30:
                    BackgroundFree[i, j] = norm.isf(poisson.cdf(0, s_tot))
                else:
                    BackgroundFree[i, j] = np.sqrt(s_tot)
        return dict(masses=np.array(self.masses, float), U2=self.U2_grid,
                    Z2d=Z2d, Zcount=Zcount, S_total=Stot, BackgroundFree=BackgroundFree)

    def contour(self, grid=None, Z_level=2.0, key="Z2d"):
        """U2 exclusion/discovery reach vs mass: the LOWEST U2 where Z crosses
        Z_level (the lower edge of the sensitive band), log-linearly interpolated
        between grid points for a smooth curve.  Returns (masses, U2_limit)."""
        grid = grid if grid is not None else self.significance_grid()
        U2 = grid["U2"]
        logU = np.log10(U2)
        out = np.full(len(self.masses), np.nan)
        for i in range(len(self.masses)):
            Z = grid[key][i]
            above = np.where(Z >= Z_level)[0]
            if not above.size:
                continue
            j = above[0]
            if j == 0:
                out[i] = U2[0]
            else:
                # interpolate the Z_level crossing in (Z, log U2) between j-1, j
                z0, z1 = Z[j - 1], Z[j]
                t = (Z_level - z0) / (z1 - z0) if z1 != z0 else 0.0
                out[i] = 10.0 ** (logU[j - 1] + t * (logU[j] - logU[j - 1]))
        return np.array(self.masses, float), out

    def significance_map(self, grid=None, key="Z2d", n_mass=120, n_u2=120):
        """Interpolate the significance onto a fine (m_N, U2) grid for the
        heatmap/contours.  Mass is only simulated at discrete points (the
        reference samples), so we interpolate log10(Z) over (log m_N, log U2);
        U2 is already continuous via reweighting.  Returns (M, U2, Z) meshes."""
        from scipy.interpolate import RegularGridInterpolator
        grid = grid if grid is not None else self.significance_grid()
        logm = np.log10(grid["masses"]); logu = np.log10(grid["U2"])
        Zfloor = np.maximum(grid[key], 1e-3)
        interp = RegularGridInterpolator((logm, logu), np.log10(Zfloor),
                                         bounds_error=False, fill_value=None)
        mfine = np.logspace(logm[0], logm[-1], n_mass)
        ufine = np.logspace(logu[0], logu[-1], n_u2)
        M, U2 = np.meshgrid(mfine, ufine, indexing="ij")
        pts = np.column_stack([np.log10(M).ravel(), np.log10(U2).ravel()])
        Z = 10.0 ** interp(pts).reshape(M.shape)
        return M, U2, Z

    def _contour_segments(self, M, U2, Z, level):
        """(m_N, U2) contour points at `level`, Balloon.ipynb format: multiple
        segments joined by [nan, nan] rows (empty -> (0,2) array)."""
        import matplotlib.pyplot as plt
        fig = plt.figure()
        cs = plt.contour(M, U2, Z, levels=[level])
        segs = cs.allsegs[0]
        plt.close(fig)
        if segs:
            parts = [np.vstack([seg, [np.nan, np.nan]]) for seg in segs]
            return np.concatenate(parts)[:-1]
        return np.empty((0, 2))

    def save_contours(self, path, levels=(2.0, 5.0), key="Z2d", grid=None):
        """Save (m_N, U2) contour curves to an .npz in the Balloon.ipynb format:
        keys '<detector>_<photon_threshold>_<level>' -> (N,2) arrays of contour
        points (nan-separated segments).  Also stored per count-only key with a
        '_count' suffix for comparison."""
        grid = grid if grid is not None else self.significance_grid()
        M, U2, Z = self.significance_map(grid, key=key)
        _, _, Zc = self.significance_map(grid, key="Zcount")
        data = {}
        for level in levels:
            k = f"{self.detector}_{self.min_photons:.1f}_{level:.1f}"
            data[k] = self._contour_segments(M, U2, Z, level)
            data[k + "_count"] = self._contour_segments(M, U2, Zc, level)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez(path, **data)
        print(f"[hnl_sensitivity] saved contours to {path}: {list(data.keys())}")
        return data

    def plot_heatmap(self, key="Z2d", levels=(2.0, 5.0), shade_level=2.0,
                     output=None, ax=None, grid=None):
        """2D significance heatmap over (m_N, U2) with contour lines and the
        Z>=shade_level (sensitive / excludable) region hatched."""
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        style = os.path.join(os.getcwd(), "figures.mplstyle")
        if os.path.exists(style):
            try:
                plt.style.use(style)
            except Exception:
                pass
        grid = grid if grid is not None else self.significance_grid()
        M, U2, Z = self.significance_map(grid, key=key)
        if ax is None:
            fig, ax = plt.subplots(figsize=(6.6, 4.8))
        else:
            fig = ax.figure
        mesh = ax.pcolormesh(M, U2, np.clip(Z, 0.1, None),
                             norm=LogNorm(vmin=0.5, vmax=max(10, Z.max())),
                             cmap="viridis", shading="auto")
        cb = fig.colorbar(mesh, ax=ax)
        cb.set_label("median significance $Z$ (2D likelihood)")
        cs = ax.contour(M, U2, Z, levels=list(levels), colors="white",
                        linestyles=["--", "-"][:len(levels)], linewidths=1.5)
        ax.clabel(cs, fmt=lambda v: f"{v:.0f}$\\sigma$", fontsize=8)
        # hatch the Z >= shade_level region (the reach)
        ax.contourf(M, U2, (Z >= shade_level).astype(float), levels=[0.5, 1.5],
                    colors="none", hatches=["///"], alpha=0)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"$m_N$ [GeV]"); ax.set_ylabel(r"$|U_\mu|^2$")
        ax.set_title(f"HNL sensitivity, detector {self.detector} "
                     f"({self.min_photons*SIPM_PDE:.0f} PE); hatched: $Z\\geq{shade_level:.0f}$",
                     fontsize=9)
        fig.tight_layout()
        if output:
            os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
            fig.savefig(output, dpi=150, bbox_inches="tight")
            print(f"[hnl_sensitivity] saved {output}")
        return fig, ax

    def plot_contours(self, Z_level=2.0, output=None, ax=None):
        import matplotlib.pyplot as plt
        style = os.path.join(os.getcwd(), "figures.mplstyle")
        if os.path.exists(style):
            try:
                plt.style.use(style)
            except Exception:
                pass
        grid = self.significance_grid()
        if ax is None:
            fig, ax = plt.subplots(figsize=(6.2, 4.6))
        else:
            fig = ax.figure
        m2d, u2d = self.contour(grid, Z_level, "Z2d")
        mc, uc = self.contour(grid, Z_level, "Zcount")
        ax.plot(m2d, u2d, "-o", ms=3, label="2D likelihood (sep + hadronic)")
        ax.plot(mc, uc, "--s", ms=3, color="gray",
                label="count only (no discriminator)")
        ax.set_yscale("log")
        ax.set_xlabel(r"$m_N$ [GeV]")
        ax.set_ylabel(r"$U^2$ reach (Z=%.0f)" % Z_level)
        ax.set_title(f"HNL sensitivity, detector {self.detector} "
                     f"({self.min_photons*SIPM_PDE:.0f} PE)", fontsize=10)
        ax.legend(fontsize=8)
        fig.tight_layout()
        if output:
            os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
            fig.savefig(output, dpi=150, bbox_inches="tight")
            print(f"[hnl_sensitivity] saved {output}")
        return fig, ax
