"""HNL sensitivity in (m_N, U2) from a 1D binned likelihood in one camera
discriminator, with an opening-angle pre-cut:

  * PRE-CUT  ``oncam_sep_deg`` -- the on-camera mu-mu separation.  HNL pairs are
    collimated; charm dimuons are wider.  Events are kept only if
        oncam_sep_deg < SEP_CUT_PIX * PIX_DEG   (5 pixels = 1.5 deg).
  * LIKELIHOOD ``pix_sep`` -- the beam-axis distance of the dimuon system,
        pix_sep = |mu1_pix + mu2_pix| * PIX_DEG   [deg],
    which exploits that the muon-decay-neutrino charm background is collimated
    on the beam axis (its dimuons cluster near pixel (0,0)), whereas the HNL
    signal is produced off-axis by muon scattering.  This variable carries
    essentially all of the discrimination, so a 1D pix_sep likelihood matches
    the full 2D one while being far more robust to sparse-background bins.

Method
------
The HNL signal kinematics/imaging are U2-INDEPENDENT (they depend on m_N and the
decay POSITION, not the mixing), so a single reference sample per mass -- run
with mixture decay-distance sampling -- is reweighted to any (m_N, U2) using the
stored proposal density ``sampling_pdf``.  Only the per-event weights change; the
per-event (oncam_sep_deg, pix_sep) carry over.

For each (m_N, U2):
  * reweight the reference HNL events -> per-event expected-signal weight;
  * apply the opening-angle cut and bin the tagged signal in 1D pix_sep;
  * the charm background (U2-INDEPENDENT), summed over the scattering and decay
    neutrino sources, is binned once the same way;
  * the median expected significance is the paper's background-limited value

        Z = sqrt( sum_bins s^2 / (s + b) )                    [s/sqrt(s+b)]

    and the sensitivity contour is where Z crosses a chosen level (e.g. 2).
    Because s^2/(s+b) <= s per bin, Z can never exceed its background-free
    limit sqrt(sum s) = sqrt(s_tot); empty-background bins contribute s (they
    are NOT dropped).

Runs locally: HNLFluxGeometry (the reweighter) is pure numpy -- no SIREN needed.

Typical use
-----------
    from src.hnl_sensitivity import SensitivityModel
    m = SensitivityModel(detector=8, R_det=2, masses=[5,10,20,30,50])
    grid = m.significance_grid()            # Z(m_N, U2)
    m.plot_contours(output="figures/sensitivity_det8.png")
"""
import os
import glob

import numpy as np

from src.constants import N_muon_decays

# thresholds (12 PE / 0.4), matching charm_vs_hnl
SIPM_PDE = 0.40
MIN_PHOTOELECTRONS = 12.0
MIN_PHOTONS_DEFAULT = MIN_PHOTOELECTRONS / SIPM_PDE   # 30 photons
# camera pixel size used by the sims (mu*_pix are stored in these pixel units);
# pix_sep converts them to degrees.
PIX_DEG = 0.3
# opening-angle pre-cut on the on-camera mu-mu separation, in pixels: keep the
# collimated HNL pairs, reject wider charm dimuons.  Applied (as oncam_sep_deg <
# SEP_CUT_PIX * PIX_DEG) identically to signal and charm before binning.
SEP_CUT_PIX = 5.0

# charm neutrino sources summed into the background
DEFAULT_CHARM_MODES = ("scattering", "decay")

# nominal beam-dump config (matches the sims)
E_MU, DUMP_DEPTH, DUMP_ANGLE = 5000.0, 100.0, 1.53


def _geom():
    """Pure-numpy HNLFluxGeometry for reweighting (no SIREN)."""
    from src.balloon import HNLFluxGeometry
    return HNLFluxGeometry(E_mu=E_MU, dump_depth=DUMP_DEPTH, dump_angle=DUMP_ANGLE)


def pix_beamline_angle(d):
    """Beam-axis distance of the dimuon system on the camera [deg]:
    |mu1_pix + mu2_pix| * PIX_DEG.  ``d`` is an npz mapping (or dict) with the
    unified per-event pixel arrays mu1_pix, mu2_pix of shape (N, 2)."""
    v = np.asarray(d["mu1_pix"], float) + np.asarray(d["mu2_pix"], float)
    return np.sqrt(np.sum(v ** 2, axis=1)) * PIX_DEG


# --------------------------------------------------------------------------- #
# Binning of the 2D discriminator plane
# --------------------------------------------------------------------------- #
def default_bins():
    """(sep_edges [deg], pixsep_edges [deg]).

    Both start at 0 (a near-axis / merged-spot bin) and end at +inf to catch the
    tail.  On-camera separation distinguishes collimated HNL pairs from wider
    charm; beam-axis distance distinguishes on-axis (decay) charm from off-axis
    HNL."""
    sep_edges = np.arange(PIX_DEG,6.0,PIX_DEG)
    pixsep_edges = list(np.arange(0,6.0,PIX_DEG)) + [np.inf]
    return sep_edges, pixsep_edges


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_charm_2d(detector, R_det=2, modes=DEFAULT_CHARM_MODES,
                  min_photons=MIN_PHOTONS_DEFAULT, had_veto=True, data_dir="data"):
    """Charm background per-event (sep, pix_sep, weight) for tagged events on one
    detector, SUMMED over the requested neutrino sources (modes).  Weights are
    absolute expected events (per 1e22 muon decays).

    ``had_veto`` (default True): additionally reject charm events with a
    detectable hadronic shower (had_n_ph >= min_photons).  The HNL signal is
    leptonic (no shower), so it passes the veto automatically -- this is the
    single strongest charm-rejection handle, applied here as a pre-cut."""
    sep, pxs, w = [], [], []
    for mode in modes:
        cdir = os.path.join(data_dir,
                            f"scan_results_balloon_charm_{mode}_det{detector}_R{R_det:g}")
        files = sorted(glob.glob(os.path.join(cdir, "charm_bkg_seed_*.npz")))
        if not files:
            raise FileNotFoundError(f"no charm files in {cdir}")
        n_seeds = len(files)
        for f in files:
            d = np.load(f)
            m0 = np.asarray(d["mu1_n_ph"], float)
            m1 = np.asarray(d["mu2_n_ph"], float)
            N_samples = int(d["N_samples"])
            # both muons above threshold on the (shared) camera, opening-angle cut
            tagged = ((m0 >= min_photons) & (m1 >= min_photons)
                      & np.asarray(d["same_detector"], bool)
                      & np.isfinite(d["oncam_sep_deg"])
                      & (np.asarray(d["oncam_sep_deg"], float) < SEP_CUT_PIX * PIX_DEG))
            if had_veto:                       # reject charm with a detectable shower
                tagged &= np.asarray(d["had_n_ph"], float) < min_photons
            wev = (np.asarray(d["interaction_weights"], float) * float(d["N_nu_per_muon"])
                   * float(d["cherenkov_weight"]) * N_muon_decays
                   / N_samples / n_seeds)
            sep.append(np.asarray(d["oncam_sep_deg"], float)[tagged])
            pxs.append(pix_beamline_angle(d)[tagged])
            w.append(wev[tagged])
    return np.concatenate(sep), np.concatenate(pxs), np.concatenate(w)


def load_hnl_ref(mass, detector, R_det=2, data_dir="data"):
    """HNL reference events for one mass/detector: the reweight inputs + the
    per-event discriminators (oncam_sep_deg, mu*_pix) for tagged events."""
    path = os.path.join(data_dir,
                        f"scan_results_balloon_hnl_det{detector}_R{R_det:g}",
                        f"hnl_signal_mN_{mass:.0f}.npz")
    return np.load(path, allow_pickle=True)


# --------------------------------------------------------------------------- #
# Reweighting a reference sample to (m_N, U2)
# --------------------------------------------------------------------------- #
def reweight_hnl(geom, ref, m_N, U2, min_photons=MIN_PHOTONS_DEFAULT):
    """Return (sep, pix_sep, weight) for the reference HNL events reweighted to
    (m_N, U2).

    General importance reweight using the stored proposal density
    ``sampling_pdf`` q(decay_dist): the decay-position weight for the target is
    decay_pos_probability = p_target(decay_dist) / q(decay_dist).  Applies the
    same tag as charm (both muons above threshold on the shared camera)."""
    from src.xs_and_decays import HNL_decay_length
    E = np.asarray(ref["hnl_energy"], float)
    d = np.asarray(ref["decay_dist"], float)
    d_max = np.asarray(ref["d_max"], float)
    q = np.asarray(ref["sampling_pdf"], float)
    with np.errstate(divide="ignore", invalid="ignore"):
        L = HNL_decay_length(m_N, U2, E)                      # target decay length
        decay_prob = np.where(d_max > 0, 1.0 - np.exp(-d_max / L), 0.0)
        p_target = np.where(decay_prob > 0,
                            np.exp(-d / L) / (L * (1.0 - np.exp(-d_max / L))), 0.0)
        dpp = np.where((decay_prob > 0) & (q > 0), p_target / q, 0.0)
    ip = float(geom.compute_weighted_production_rate(m_N, U2)[0])
    br = float(np.ravel(ref["meta_BR_mumu"])[0])
    cw = float(np.ravel(ref["meta_cherenkov_weight"])[0])
    # Normalize by N_samples (beam MC events THROWN), NOT meta_n_events: cw
    # already rescales the imaged sample up to the full valid sample, and the MC
    # estimator of the expected rate divides by the total number thrown.
    ns = float(np.ravel(ref["meta_N_samples"])[0])
    w = (decay_prob * dpp * ip * br * N_muon_decays * cw / ns
         if ns > 0 else np.zeros_like(d))
    both = ((np.asarray(ref["mu1_n_ph"], float) >= min_photons)
            & (np.asarray(ref["mu2_n_ph"], float) >= min_photons)
            & np.asarray(ref["same_detector"], bool)
            & np.isfinite(ref["oncam_sep_deg"])
            & (np.asarray(ref["oncam_sep_deg"], float) < SEP_CUT_PIX * PIX_DEG))
    return (np.asarray(ref["oncam_sep_deg"], float)[both],
            pix_beamline_angle(ref)[both], np.asarray(w)[both])


# --------------------------------------------------------------------------- #
# Significance
# --------------------------------------------------------------------------- #
def z_ssb(s, b):
    """Binned s/sqrt(s+b) significance: Z = sqrt( sum_bins s^2 / (s + b) ).

    The paper's background-limited sensitivity convention.  Per bin the term
    s^2/(s+b) is bounded above by s (attained as b -> 0), so:
      * the background-free limit is Z_bkgfree = sqrt(sum s) = sqrt(s_tot);
      * since s+b >= s, the with-background Z can NEVER exceed sqrt(s_tot)
        (no capping needed);
      * empty-background bins (b == 0, s > 0) contribute s and are NOT dropped.
    s, b may be any shape."""
    s = np.asarray(s, float); b = np.asarray(b, float)
    denom = s + b
    m = denom > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        z2 = np.where(m, s * s / np.where(m, denom, 1.0), 0.0)
    return float(np.sqrt(z2.sum()))


def asimov_Z(s, b):
    """Binned Asimov median DISCOVERY significance, sum over bins with b > 0.
    Z = sqrt( 2 sum [ (s+b) ln(1 + s/b) - s ] ).  s, b may be any shape.
    NOTE: kept for reference/back-compat; the sensitivity now uses ``z_ssb``.
    In near-background-free bins this Asimov form can exceed sqrt(s_tot), which
    is why the background-limited s/sqrt(s+b) is used instead."""
    s = np.asarray(s, float); b = np.asarray(b, float)
    m = b > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        term = (s[m] + b[m]) * np.log1p(s[m] / b[m]) - s[m]
    return float(np.sqrt(2.0 * np.maximum(term, 0).sum()))


def save_all_contours(path, detectors=(2, 5, 8), R_det=2,
                      masses=(5, 6, 7, 8, 9, 10, 12, 14, 16, 20, 25, 30),
                      levels=(2.0, 5.0), min_photons=MIN_PHOTONS_DEFAULT,
                      modes=DEFAULT_CHARM_MODES, had_veto=True, data_dir="data",
                      key="Z2d"):
    """Save contours for ALL detectors into ONE .npz, Balloon.ipynb-style: keys
    '<detector>_<photon_threshold>_<level>' (+ '..._count') -> (m_N, U2) points."""
    merged = {}
    for det in detectors:
        m = SensitivityModel(detector=det, R_det=R_det, masses=masses,
                             min_photons=min_photons, modes=modes,
                             had_veto=had_veto, data_dir=data_dir)
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


class SensitivityModel:
    """1D pix_sep-likelihood HNL sensitivity (with an oncam_sep_deg opening-angle
    pre-cut) for one detector, using the s/sqrt(s+b) background-limited TS."""

    def __init__(self, detector=8, R_det=2, masses=(5, 10, 20, 30, 50),
                 min_photons=MIN_PHOTONS_DEFAULT, modes=DEFAULT_CHARM_MODES,
                 had_veto=True, data_dir="data", U2_grid=None):
        self.detector = detector
        self.R_det = R_det
        self.masses = list(masses)
        self.min_photons = min_photons
        self.modes = tuple(modes)
        self.had_veto = had_veto
        self.data_dir = data_dir
        self.U2_grid = (np.logspace(-14, -8, 61) if U2_grid is None
                        else np.asarray(U2_grid, float))
        self.sep_edges, self.pixsep_edges = default_bins()
        self.geom = _geom()
        # charm background (U2-independent), 1D pix_sep histogram, summed over
        # modes.  The opening-angle cut is applied inside load_charm_2d.
        cs, cps, cw = load_charm_2d(detector, R_det=R_det, modes=self.modes,
                                    min_photons=min_photons, had_veto=had_veto,
                                    data_dir=data_dir)
        self.B, _ = np.histogram(cps, self.pixsep_edges, weights=cw)
        self.B_total = float(cw.sum())
        self.refs = {m: load_hnl_ref(m, detector, R_det=R_det, data_dir=data_dir)
                     for m in self.masses}

    def signal_hist(self, m_N, U2):
        """1D pix_sep signal histogram (after the opening-angle pre-cut)."""
        sep, ps, w = reweight_hnl(self.geom, self.refs[m_N], m_N, U2, self.min_photons)
        S, _ = np.histogram(ps, self.pixsep_edges, weights=w)
        return S, float(S.sum())

    def significance_grid(self):
        """Z(m_N, U2) for the 1D pix_sep likelihood (s/sqrt(s+b)) and for a plain
        count (no discriminator), plus the background-free ceiling sqrt(s_tot).
        The grid key ``Z2d`` holds the primary (1D pix_sep) significance -- the
        name is kept for downstream/Balloon.ipynb compatibility.  Returns dict of
        (n_mass, n_U2) arrays."""
        Z2d = np.zeros((len(self.masses), len(self.U2_grid)))
        Zcount = np.zeros_like(Z2d)
        Stot = np.zeros_like(Z2d)
        BackgroundFree = np.zeros_like(Z2d)
        for i, m in enumerate(self.masses):
            for j, U2 in enumerate(self.U2_grid):
                S, s_tot = self.signal_hist(m, U2)
                Z2d[i, j] = z_ssb(S, self.B)                    # 1D pix_sep s/sqrt(s+b)
                Zcount[i, j] = z_ssb([s_tot], [self.B_total])   # count-only s/sqrt(s+b)
                Stot[i, j] = s_tot
                BackgroundFree[i, j] = np.sqrt(s_tot)           # b -> 0 limit of z_ssb
        return dict(masses=np.array(self.masses, float), U2=self.U2_grid,
                    Z2d=Z2d, Zcount=Zcount, S_total=Stot, BackgroundFree=BackgroundFree)

    def contour(self, grid=None, Z_level=2.0, key="Z2d"):
        """U2 reach vs mass: the LOWEST U2 where Z crosses Z_level (lower edge of
        the sensitive band), log-linearly interpolated.  Returns (masses, U2_limit)."""
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
                z0, z1 = Z[j - 1], Z[j]
                t = (Z_level - z0) / (z1 - z0) if z1 != z0 else 0.0
                out[i] = 10.0 ** (logU[j - 1] + t * (logU[j] - logU[j - 1]))
        return np.array(self.masses, float), out

    def significance_map(self, grid=None, key="Z2d", n_mass=120, n_u2=120):
        """Interpolate log10(Z) onto a fine (m_N, U2) grid (mass is only simulated
        at discrete points; U2 is continuous via reweighting).  Returns (M,U2,Z)."""
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
        """(m_N, U2) contour points at `level`, Balloon.ipynb format: segments
        joined by [nan, nan] rows (empty -> (0,2) array)."""
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
        keys '<detector>_<photon_threshold>_<level>' -> (N,2) nan-separated segments.
        Count-only curves stored with a '_count' suffix."""
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
        Z>=shade_level region hatched."""
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
        cb.set_label(r"significance $Z=s/\sqrt{s+b}$ (1D pix_sep)")
        cs = ax.contour(M, U2, Z, levels=list(levels), colors="white",
                        linestyles=["--", "-"][:len(levels)], linewidths=1.5)
        ax.clabel(cs, fmt=lambda v: f"{v:.0f}$\\sigma$", fontsize=8)
        ax.contourf(M, U2, (Z >= shade_level).astype(float), levels=[0.5, 1.5],
                    colors="none", hatches=["///"], alpha=0)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"$m_N$ [GeV]"); ax.set_ylabel(r"$|U_\mu|^2$")
        ax.set_title(f"HNL sensitivity, detector {self.detector} R={self.R_det} m "
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
        ax.plot(m2d, u2d, "-o", ms=3, label="1D pix_sep likelihood")
        ax.plot(mc, uc, "--s", ms=3, color="gray",
                label="count only (no discriminator)")
        ax.set_yscale("log")
        ax.set_xlabel(r"$m_N$ [GeV]")
        ax.set_ylabel(r"$U^2$ reach (Z=%.0f)" % Z_level)
        ax.set_title(f"HNL sensitivity, detector {self.detector} R={self.R_det} m "
                     f"({self.min_photons*SIPM_PDE:.0f} PE)", fontsize=10)
        ax.legend(fontsize=8)
        fig.tight_layout()
        if output:
            os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
            fig.savefig(output, dpi=150, bbox_inches="tight")
            print(f"[hnl_sensitivity] saved {output}")
        return fig, ax
