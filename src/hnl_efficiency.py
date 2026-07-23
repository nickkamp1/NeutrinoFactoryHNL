"""Average HNL tagging efficiency vs distance and impact parameter.

This is a light post-processing / plotting layer over the pre-made balloon scan
``.npz`` files (produced by ``cluster/run_scan.py`` and ``run_scan_dimuon.py``).
It does NOT re-run any simulation -- it reads the saved per-event photon counts,
production/decay positions, and decay weights, applies the Trinity photoelectron
threshold, and turns them into the *expected number of tagged events* as a
function of two geometric variables, evaluated per (event, detector) pair:

  * DISTANCE           : straight-line distance from the HNL decay vertex to the
                         detector (the Cherenkov light-travel distance).
  * IMPACT PARAMETER   : perpendicular ("closest-approach") distance of the HNL
                         flight line -- the line from the production point along
                         the HNL direction -- to the detector.  Because the
                         daughter muons/hadrons are nearly collinear with the
                         parent HNL, this HNL impact parameter is transferred to
                         the daughters and sets how close the visible tracks (and
                         hence their Cherenkov pool) pass to the detector.

Tagging (numerator), matching the existing analysis (cluster/collect_results*):
  * dimuon channel  (data/scan_results_balloon_dimuon):
        BOTH muons above threshold on the SAME detector (background-free sample).
  * detailed channel (data/scan_results_balloon_detailed):
        total (muon + hadronic) photon count above threshold on the detector.
The threshold is a single-pixel photoelectron cut (default 20 PE at the Trinity
Demonstrator operating point) converted to raw photons via the SiPM PDE (0.40),
i.e. ~50 photons -- identical to run_dimuon_scan / collect_results_dimuon.

Absolute normalization (per (event, detector) pair), matching collect_results*:
    w = N_HNLs_per_muon(U2) * cherenkov_weight(U2)
        * decay_probability(event) * decay_pos_probability(event)
        / N_samples * N_muon_decays
Summing ``w`` over tagged pairs in a bin gives the expected number of tagged
events landing on a detector in that bin.  The dimuon files fold BR(N->nu mu mu)
into ``N_HNLs_per_muon``; the (older-format) detailed files store
``interaction_probability`` instead, which is used as ``N_HNLs_per_muon``.

Because we work per (event, detector) pair, an event that tags several detectors
is counted once per detector -- the correct semantics for "expected tagged
events vs distance/impact parameter to A detector", but note it therefore
double-counts relative to the best-detector event totals in the sensitivity
scan.  The ``detectors`` argument of ``compute`` controls this: "all" (default,
per-detector), "best" (each event's brightest tagging detector only), or a
detector ID / list of IDs to restrict to specific detector(s).

Typical use
-----------
    from src.hnl_efficiency import compute, plot_efficiency, average_efficiency
    res = compute(channel="dimuon")                 # nominal setup, all files
    print(average_efficiency(res))                  # scalar avg tag efficiency
    plot_efficiency(res, quantity="events", output="figures/eff_dimuon.png")

or from the project root:
    python -m src.hnl_efficiency --channel dimuon --output figures/eff.png
"""
import os
import glob
import argparse
from dataclasses import dataclass, field

import numpy as np

from src.constants import N_muon_decays

# --------------------------------------------------------------------------- #
# Detection threshold (mirrors balloon_siren, without importing SIREN).
# --------------------------------------------------------------------------- #
SIPM_PDE = 0.40             # SiPM photon detection efficiency (band-averaged)
MIN_PHOTOELECTRONS = 20.0   # single-pixel PE threshold (Trinity Demonstrator)


# --------------------------------------------------------------------------- #
# Beam-dump configuration (metadata / plot labels).  The shipped scan files were
# generated with the "nominal" values below; changing E_mu/depth/angle requires
# re-running the cluster scan, but the detector geometry is read from the files.
# --------------------------------------------------------------------------- #
@dataclass
class BeamDumpConfig:
    E_mu: float = 5000.0       # muon beam energy [GeV]
    dump_depth: float = 100.0  # beam-dump depth below surface [m]
    dump_angle: float = 1.53   # beam angle [rad] (nearly horizontal, curved Earth)
    nature: str = "Majorana"   # HNL chiral nature (dimuon channel)
    label: str = "nominal"

    def title(self):
        return (f"{self.label}: $E_\\mu$={self.E_mu:.0f} GeV, "
                f"depth={self.dump_depth:.0f} m, angle={self.dump_angle:.2f} rad")


NOMINAL = BeamDumpConfig()


# --------------------------------------------------------------------------- #
# Channel registry.  Each entry knows where its files live and how to turn a
# loaded file into a per-detector TAG METRIC: the photon count that must clear
# the threshold.  A detector tags at threshold T iff metric >= T, so a single
# metric supports an arbitrary set of thresholds in one pass.
# --------------------------------------------------------------------------- #
def _metric_dimuon(data, iu):
    """Both-muon tag metric = min(mu1, mu2) per detector -> (N_det, N_samples).
    Both muons clear T iff min(mu1, mu2) >= T."""
    mpc = np.asarray(data["mu_photon_counts"][iu])   # (2, N_det, N_samples)
    return mpc.min(axis=0)


def _metric_detailed(data, iu):
    """Total (muon + hadronic) light per detector -> (N_det, N_samples)."""
    return np.asarray(data["photon_counts"][iu])     # (N_det, N_samples)


CHANNELS = {
    "dimuon": dict(
        scan_dir="data/scan_results_balloon_dimuon",
        pattern="scan_mN_*_u2batch_*.npz",
        metric_fn=_metric_dimuon,
        desc=r"$N_4\to\nu\,\mu^-\mu^+$ (both-muon tag)",
    ),
    "detailed": dict(
        scan_dir="data/scan_results_balloon_detailed",
        pattern="scan_mN_*_u2batch_*.npz",
        metric_fn=_metric_detailed,
        desc=r"$N_4\to\mu+\mathrm{hadrons}$ (total-light tag)",
    ),
}


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def decay_detector_distance(decay_pos, det_pos):
    """Straight-line distance [m] from each decay vertex to the detector.

    decay_pos : (N, 3) array of decay positions; det_pos : (3,) detector position.
    """
    return np.linalg.norm(decay_pos - det_pos[None, :], axis=1)


def hnl_impact_parameter(prod_pos, decay_pos, det_pos):
    """Perpendicular distance [m] from the detector to each HNL flight line.

    The HNL line passes through the production point along the HNL direction
    (decay - production); the impact parameter is the closest approach of that
    line to the detector.  Since the daughters are ~collinear with the HNL, this
    is transferred to the daughter tracks.

    prod_pos, decay_pos : (N, 3); det_pos : (3,).
    """
    d = decay_pos - prod_pos                     # (N, 3) HNL flight vector
    n = np.linalg.norm(d, axis=1, keepdims=True)
    dhat = np.divide(d, n, out=np.zeros_like(d), where=n > 0)
    rel = det_pos[None, :] - prod_pos            # (N, 3) detector rel. to origin of line
    return np.linalg.norm(np.cross(rel, dhat), axis=1)


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
def _empty_acc(dist_edges, b_edges, n_thr):
    """Accumulator for one (m_N, U2).  Tag arrays carry a leading threshold axis;
    candidate arrays are threshold-independent."""
    nd, nb = len(dist_edges) - 1, len(b_edges) - 1
    return dict(
        tag_dist=np.zeros((n_thr, nd)), cand_dist=np.zeros(nd),
        tag_b=np.zeros((n_thr, nb)), cand_b=np.zeros(nb),
        tag_2d=np.zeros((n_thr, nd, nb)), cand_2d=np.zeros((nd, nb)),
        tag_tot=np.zeros(n_thr), cand_tot=0.0,
        # overflow (weight falling outside the histogram ranges), for diagnostics
        over_dist=0.0, over_b=0.0,
    )


@dataclass
class EfficiencyResult:
    channel: str
    config: BeamDumpConfig
    thresholds_pe: np.ndarray          # (n_thr,) PE thresholds
    pde: float
    dist_edges: np.ndarray
    b_edges: np.ndarray
    detector_positions: np.ndarray
    points: dict = field(default_factory=dict)   # (m_N, U2) -> accumulator dict
    detector_select: object = "all"              # "all" | "best" | list[int]

    def detector_label(self):
        """Human-readable description of the detector selection, for plot titles."""
        d = self.detector_select
        if d == "all":
            return "all detectors"
        if d == "best":
            return "best detector"
        parts = []
        for i in d:
            p = self.detector_positions[i]
            parts.append(f"det {i} ($z$={p[2] / 1e3:.0f} km, $x$={p[0]:.0f} m)")
        return ", ".join(parts)

    @property
    def min_photons(self):
        return self.thresholds_pe / self.pde       # (n_thr,) raw-photon thresholds

    @property
    def dist_centers(self):
        return 0.5 * (self.dist_edges[:-1] + self.dist_edges[1:])

    @property
    def b_centers(self):
        return 0.5 * (self.b_edges[:-1] + self.b_edges[1:])

    def thr_index(self, threshold_pe=None):
        """Index of the threshold nearest ``threshold_pe`` (default: first)."""
        if threshold_pe is None:
            return 0
        return int(np.argmin(np.abs(self.thresholds_pe - float(threshold_pe))))

    def keys(self, masses=None, u2_values=None):
        """Selected (m_N, U2) keys, optionally filtered by mass / mixing.

        NOTE: U2 matching uses atol=0 -- the default np.isclose atol (1e-8) is
        enormous next to U2 ~ 1e-11 and would match everything.
        """
        out = []
        for (m_N, U2) in sorted(self.points):
            if masses is not None and not np.any(
                    np.isclose(m_N, np.atleast_1d(masses), rtol=1e-3, atol=0)):
                continue
            if u2_values is not None and not np.any(
                    np.isclose(U2, np.atleast_1d(u2_values), rtol=1e-3, atol=0)):
                continue
            out.append((m_N, U2))
        return out

    def _sum(self, field_name, masses=None, u2_values=None):
        ks = self.keys(masses, u2_values)
        if not ks:
            raise ValueError("no (m_N, U2) points match the selection")
        vals = [self.points[k][field_name] for k in ks]
        return np.sum(vals, axis=0)


# --------------------------------------------------------------------------- #
# Core computation
# --------------------------------------------------------------------------- #
def compute(channel="dimuon", scan_dir=None, config=NOMINAL,
            min_pe=MIN_PHOTOELECTRONS, pde=SIPM_PDE,
            masses=None, u2_values=None,
            dist_edges=None, b_edges=None,
            detectors="all", best_detector_only=None,
            use_decay_pos_prob=True, verbose=True):
    """Stream the scan files and accumulate expected-tagged / candidate-event
    histograms per (m_N, U2), binned in distance and HNL impact parameter to
    each detector.

    Parameters
    ----------
    channel : {"dimuon", "detailed"}
    scan_dir : str or None
        Override the channel's default directory.
    config : BeamDumpConfig
        Metadata for plot labels (defaults to the nominal setup).
    min_pe : float or sequence of float
        PE threshold(s).  Pass a list to evaluate several thresholds in a SINGLE
        pass over the files (e.g. min_pe=[10, 20, 30]); the raw-photon threshold
        is min_pe / pde.  The result stores all of them (see thresholds_pe).
    pde : float
        SiPM photon detection efficiency.
    masses, u2_values : iterable or None
        Restrict to these HNL masses [GeV] / U2 values (None = all found).
    dist_edges, b_edges : array or None
        Bin edges [m].  Defaults: distance 0-110 km (2.5 km bins), impact
        parameter 0-1 km (20 m bins).
    detectors : {"all", "best"} or int or iterable of int
        Which detector(s) enter the (event, detector) pairing:
          "all"  (default) -- every detector; an event is counted once per
                              detector it illuminates (per-detector semantics).
          "best"           -- only each event's brightest TAGGING detector
                              (distinct-event semantics; the tagging set, hence
                              the "best", is resolved per threshold).
          int / list[int]  -- restrict to that detector ID / those IDs
                              (indices into detector_positions, 0..N_det-1).
        Distance and impact parameter are always measured to whichever detector
        the pair belongs to.
    best_detector_only : bool or None
        Deprecated alias.  True maps to detectors="best", False to "all".
        Ignored unless set (None).
    use_decay_pos_prob : bool
        Include the ``decay_pos_probability`` importance weight (default True).
        The shipped files use UNIFORM decay-distance sampling, so this factor is
        required to recover the physical exponential decay profile -- essential
        for the distance / impact-parameter shapes.  Set False only to reproduce
        the (uniform-unaware) collect_results*.py convention.

    Returns
    -------
    EfficiencyResult
    """
    if channel not in CHANNELS:
        raise ValueError(f"unknown channel {channel!r}; choose from {list(CHANNELS)}")
    spec = CHANNELS[channel]
    scan_dir = scan_dir or spec["scan_dir"]
    metric_fn = spec["metric_fn"]

    # resolve the detector selection -> (best_mode, det_indices).  det_indices is
    # None for "all detectors"; otherwise an explicit list of detector IDs.
    if best_detector_only is not None:                 # deprecated alias wins if set
        detectors = "best" if best_detector_only else "all"
    if isinstance(detectors, str):
        if detectors not in ("all", "best"):
            raise ValueError('detectors must be "all", "best", an int, or a list of ints')
        best_mode = detectors == "best"
        det_indices = None
    else:
        best_mode = False
        det_indices = ([int(detectors)] if np.isscalar(detectors)
                       else [int(d) for d in detectors])
    thresholds_pe = np.atleast_1d(np.asarray(min_pe, float))
    thr_photons = thresholds_pe / pde          # (n_thr,) raw-photon thresholds
    n_thr = len(thr_photons)

    files = sorted(glob.glob(os.path.join(scan_dir, spec["pattern"])))
    if not files:
        raise FileNotFoundError(f"no scan files matching {spec['pattern']} in {scan_dir}")

    if dist_edges is None:
        dist_edges = np.linspace(0.0, 110e3, 45)     # 0-110 km, 2.5 km bins
    if b_edges is None:
        b_edges = np.linspace(0.0, 1e3, 51)          # 0-1 km, 20 m bins
    dist_edges = np.asarray(dist_edges, float)
    b_edges = np.asarray(b_edges, float)

    masses = None if masses is None else np.atleast_1d(masses).astype(float)
    u2_values = None if u2_values is None else np.atleast_1d(u2_values).astype(float)

    points = {}
    det_pos_ref = None

    for f in files:
        data = np.load(f, allow_pickle=True)
        m_N = float(data["m_N"])
        if masses is not None and not np.any(np.isclose(m_N, masses)):
            continue

        det_pos = np.asarray(data["detector_positions"], float)    # (N_det, 3)
        det_pos_ref = det_pos
        N_det = det_pos.shape[0]
        loop_dets = range(N_det) if det_indices is None else det_indices
        if det_indices is not None:
            bad = [i for i in det_indices if not 0 <= i < N_det]
            if bad:
                raise ValueError(f"detector ID(s) {bad} out of range 0..{N_det - 1}")
        N_samples = int(data["N_samples"])
        U2_batch = np.atleast_1d(np.asarray(data["U2_batch"], float))
        decay = np.asarray(data["decay_positions"], float)          # (nU2, N, 3)
        prod = np.asarray(data["production_positions"], float)
        decay_prob = np.asarray(data["decay_probability"], float)   # (nU2, N)
        decay_pos_prob = (np.asarray(data["decay_pos_probability"], float)
                          if (use_decay_pos_prob and "decay_pos_probability" in data.files)
                          else np.ones_like(decay_prob))
        cw = (np.atleast_1d(np.asarray(data["cherenkov_weights"], float))
              if "cherenkov_weights" in data.files else np.ones(len(U2_batch)))
        # dimuon files fold BR into N_HNLs_per_muon; detailed files (older format)
        # only store interaction_probability -> use it directly.
        nhnl = np.atleast_1d(np.asarray(
            data["N_HNLs_per_muon"] if "N_HNLs_per_muon" in data.files
            else data["interaction_probability"], float))

        for iu, U2 in enumerate(U2_batch):
            if u2_values is not None and not np.any(np.isclose(U2, u2_values, rtol=1e-3)):
                continue

            metric = metric_fn(data, iu)         # (N_det, N_samples) photon metric
            dp = decay[iu]                        # (N_samples, 3)
            pp = prod[iu]
            wd = decay_prob[iu]
            w_event = (nhnl[iu] * cw[iu] * wd * decay_pos_prob[iu]
                       / N_samples * N_muon_decays)     # (N_samples,)
            valid = (dp[:, 2] > 0) & (wd > 0)

            key = (m_N, float(U2))
            acc = points.setdefault(key, _empty_acc(dist_edges, b_edges, n_thr))

            # "best" mode: brightest TAGGING detector per event, per threshold
            # (the tagging set shifts with the threshold).
            if best_mode:
                best_det = _best_tagging_detector(metric, thr_photons)  # (n_thr, N_samples)

            for idet in loop_dets:
                D = det_pos[idet]
                cand = valid & (dp[:, 2] < D[2])
                if not cand.any():
                    continue
                dist = decay_detector_distance(dp[cand], D)
                bimp = hnl_impact_parameter(pp[cand], dp[cand], D)
                w = w_event[cand]
                m_det = metric[idet][cand]        # (n_cand,) photon metric

                # candidate histograms (threshold-independent)
                acc["cand_dist"] += np.histogram(dist, dist_edges, weights=w)[0]
                acc["cand_b"] += np.histogram(bimp, b_edges, weights=w)[0]
                acc["cand_2d"] += np.histogram2d(dist, bimp, [dist_edges, b_edges],
                                                 weights=w)[0]
                acc["cand_tot"] += w.sum()
                acc["over_dist"] += w.sum() - w[(dist >= dist_edges[0]) &
                                                (dist <= dist_edges[-1])].sum()
                acc["over_b"] += w.sum() - w[(bimp >= b_edges[0]) &
                                             (bimp <= b_edges[-1])].sum()

                # tagged histograms, one per threshold
                for it, thrp in enumerate(thr_photons):
                    t = m_det >= thrp
                    if best_mode:
                        t = t & (best_det[it][cand] == idet)
                    if t.any():
                        acc["tag_dist"][it] += np.histogram(dist[t], dist_edges,
                                                            weights=w[t])[0]
                        acc["tag_b"][it] += np.histogram(bimp[t], b_edges, weights=w[t])[0]
                        acc["tag_2d"][it] += np.histogram2d(dist[t], bimp[t],
                                                            [dist_edges, b_edges],
                                                            weights=w[t])[0]
                        acc["tag_tot"][it] += w[t].sum()

    detector_select = "best" if best_mode else ("all" if det_indices is None
                                                 else det_indices)
    res = EfficiencyResult(channel=channel, config=config, thresholds_pe=thresholds_pe,
                           pde=pde, dist_edges=dist_edges, b_edges=b_edges,
                           detector_positions=det_pos_ref, points=points,
                           detector_select=detector_select)
    if verbose:
        _print_summary(res)
    return res


def _best_tagging_detector(metric, thr_photons):
    """For each threshold, the detector index with the largest tag metric among
    those clearing the threshold, per event (-1 if none).

    metric : (N_det, N_samples) per-detector tag metric.
    Returns (n_thr, N_samples) int array of detector indices.
    """
    n_thr = len(thr_photons)
    out = np.full((n_thr, metric.shape[1]), -1, dtype=np.int64)
    for it, thrp in enumerate(thr_photons):
        tagged = metric >= thrp                           # (N_det, N_samples)
        masked = np.where(tagged, metric, -np.inf)
        best = np.argmax(masked, axis=0)
        best[~tagged.any(axis=0)] = -1
        out[it] = best
    return out


# --------------------------------------------------------------------------- #
# Summaries
# --------------------------------------------------------------------------- #
def average_efficiency(result, masses=None, u2_values=None, threshold_pe=None):
    """Scalar rate-weighted average tagging efficiency = expected tagged events
    / expected candidate events, over the selected (m_N, U2) points, at the
    threshold nearest ``threshold_pe`` (default: first)."""
    it = result.thr_index(threshold_pe)
    tag = result._sum("tag_tot", masses, u2_values)[it]
    cand = result._sum("cand_tot", masses, u2_values)
    return float(tag / cand) if cand > 0 else 0.0


def expected_events(result, masses=None, u2_values=None, threshold_pe=None):
    """Total expected tagged events over the selected (m_N, U2) points, at the
    threshold nearest ``threshold_pe`` (default: first)."""
    it = result.thr_index(threshold_pe)
    return float(result._sum("tag_tot", masses, u2_values)[it])


def _print_summary(result):
    thr_str = ", ".join(f"{pe:.0f}" for pe in result.thresholds_pe)
    print(f"[hnl_efficiency] channel={result.channel}  "
          f"thresholds={thr_str} PE (PDE={result.pde})  "
          f"detectors={result.detector_label()}  "
          f"{len(result.points)} (m_N,U2) points")
    for (m_N, U2) in sorted(result.points):
        a = result.points[(m_N, U2)]
        over = (a["over_dist"] / a["cand_tot"]) if a["cand_tot"] > 0 else 0.0
        effs = (a["tag_tot"] / a["cand_tot"]) if a["cand_tot"] > 0 else np.zeros_like(a["tag_tot"])
        eff_str = ", ".join(f"{e:.2e}" for e in np.atleast_1d(effs))
        print(f"  m_N={m_N:5.1f} GeV  U2={U2:.2e}  avg eff per threshold=[{eff_str}]"
              + (f"  (dist overflow {over:.1%})" if over > 0.01 else ""))


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


def _curve(result, key, var, quantity, it=0):
    """(x, y) for one (m_N, U2) point at threshold index ``it``: var in
    {'dist','b'}, quantity in {'events','efficiency'}."""
    a = result.points[key]
    if var == "dist":
        x = result.dist_centers / 1e3          # km
        tag, cand = a["tag_dist"][it], a["cand_dist"]
    else:
        x = result.b_centers                   # m
        tag, cand = a["tag_b"][it], a["cand_b"]
    if quantity == "efficiency":
        with np.errstate(divide="ignore", invalid="ignore"):
            y = np.where(cand > 0, tag / cand, np.nan)
    else:
        y = tag
    return x, y


def plot_efficiency(result, quantity="events", masses=None, u2_values=None,
                    threshold_pe=None, output=None, ax=None):
    """Two-panel figure: quantity vs distance (left) and vs HNL impact parameter
    (right), one curve per selected (m_N, U2), at a single threshold.

    quantity : {"events", "efficiency"}
        "events"     -> expected tagged events per bin (absolute).
        "efficiency" -> tagged / candidate per bin (0-1).
    threshold_pe : float or None
        Which PE threshold to plot (nearest match; default: first).  To overlay
        several thresholds, use ``plot_threshold_comparison``.
    """
    import matplotlib.pyplot as plt
    _apply_style()

    ks = result.keys(masses, u2_values)
    if not ks:
        raise ValueError("no (m_N, U2) points match the selection")
    it = result.thr_index(threshold_pe)

    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    else:
        fig = ax[0].figure

    ylabel = ("expected tagged events / bin" if quantity == "events"
              else "tagging efficiency")
    for (m_N, U2) in ks:
        label = f"$m_N$={m_N:.0f} GeV, $U^2$={U2:.1e}"
        x, y = _curve(result, (m_N, U2), "dist", quantity, it)
        ax[0].step(x, y, where="mid", label=label)
        xb, yb = _curve(result, (m_N, U2), "b", quantity, it)
        ax[1].step(xb, yb, where="mid", label=label)

    ax[0].set_xlabel("decay-to-detector distance [km]")
    ax[1].set_xlabel("HNL impact parameter to detector [m]")
    for a in ax:
        a.set_ylabel(ylabel)
        a.set_yscale("log")
        a.legend(fontsize=8)
    ax[0].set_title(CHANNELS[result.channel]["desc"])
    ax[1].set_title(f"{result.config.title()}\nthreshold = "
                    f"{result.thresholds_pe[it]:.0f} PE, {result.detector_label()}",
                    fontsize=9)
    fig.tight_layout()

    if output:
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        fig.savefig(output, dpi=150, bbox_inches="tight")
        print(f"[hnl_efficiency] saved {output}")
    return fig, ax


def plot_efficiency_2d(result, quantity="efficiency", masses=None, u2_values=None,
                       threshold_pe=None, output=None, ax=None, cmap="viridis"):
    """2D map of the quantity in (distance, HNL impact parameter).

    Unlike ``plot_efficiency`` (which overlays one curve per point), the 2D map
    AGGREGATES the selected (m_N, U2) points: summed expected events for
    ``quantity="events"``, and pooled ratio sum(tagged)/sum(candidate) for
    ``quantity="efficiency"``.  Filter to a single point for a per-point map.

    quantity : {"events", "efficiency"}
    threshold_pe : float or None
        Which PE threshold to map (nearest match; default: first).
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    _apply_style()

    it = result.thr_index(threshold_pe)
    tag2d = result._sum("tag_2d", masses, u2_values)[it]      # (n_dist, n_b)
    cand2d = result._sum("cand_2d", masses, u2_values)

    x = result.dist_edges / 1e3        # km (n_dist+1,)
    y = result.b_edges                 # m  (n_b+1,)

    if quantity == "efficiency":
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(cand2d > 0, tag2d / cand2d, np.nan)
        norm, cbar_label = LogNorm(vmin=1e-5,vmax=1), "tagging efficiency"
    else:
        z = np.where(tag2d > 0, tag2d, np.nan)
        vmax = np.nanmax(z) if np.isfinite(z).any() else 1.0
        vmin = np.nanmin(z) if np.isfinite(z).any() else vmax
        norm = LogNorm(vmin=max(vmin, vmax * 1e-4), vmax=vmax)
        cbar_label = "expected tagged events / bin"

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 4.6))
    else:
        fig = ax.figure

    # z is (n_dist, n_b); pcolormesh wants C with shape (len(y)-1, len(x)-1) -> z.T
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(alpha=0.0)        # leave empty (no-candidate) bins blank
    mesh = ax.pcolormesh(x, y, np.ma.masked_invalid(z.T), cmap=cmap_obj,
                         norm=norm, shading="flat")
    cb = fig.colorbar(mesh, ax=ax)
    cb.set_label(cbar_label)

    ax.set_xlabel("decay-to-detector distance [km]")
    ax.set_ylabel("HNL impact parameter to detector [m]")
    ks = result.keys(masses, u2_values)
    sub = (f"$m_N$={ks[0][0]:.0f} GeV, $U^2$={ks[0][1]:.1e}" if len(ks) == 1
           else f"{len(ks)} (m_N,U2) points pooled")
    ax.set_title(f"{CHANNELS[result.channel]['desc']}\n{sub}, "
                 f"threshold {result.thresholds_pe[it]:.0f} PE\n{result.detector_label()}",
                 fontsize=9)
    fig.tight_layout()

    if output:
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        fig.savefig(output, dpi=150, bbox_inches="tight")
        print(f"[hnl_efficiency] saved {output}")
    return fig, ax


def plot_threshold_comparison(result, quantity="efficiency", masses=None,
                              u2_values=None, thresholds_pe=None, output=None, ax=None):
    """Two-panel figure (vs distance, vs impact parameter) overlaying one curve
    per PE threshold, for the selected (m_N, U2) points pooled together.

    thresholds_pe : iterable or None
        Which thresholds to overlay (nearest matches; default: all in result).
    """
    import matplotlib.pyplot as plt
    _apply_style()

    ks = result.keys(masses, u2_values)
    if not ks:
        raise ValueError("no (m_N, U2) points match the selection")
    its = (list(range(len(result.thresholds_pe))) if thresholds_pe is None
           else [result.thr_index(pe) for pe in np.atleast_1d(thresholds_pe)])

    # pool the selected (m_N, U2) points once
    td = result._sum("tag_dist", masses, u2_values)     # (n_thr, n_dist)
    cd = result._sum("cand_dist", masses, u2_values)    # (n_dist,)
    tb = result._sum("tag_b", masses, u2_values)        # (n_thr, n_b)
    cb = result._sum("cand_b", masses, u2_values)       # (n_b,)

    def y_of(tag, cand):
        if quantity == "efficiency":
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(cand > 0, tag / cand, np.nan)
        return tag

    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    else:
        fig = ax[0].figure

    ylabel = ("expected tagged events / bin" if quantity == "events"
              else "tagging efficiency")
    for it in its:
        label = f"{result.thresholds_pe[it]:.0f} PE"
        ax[0].step(result.dist_centers / 1e3, y_of(td[it], cd), where="mid", label=label)
        ax[1].step(result.b_centers, y_of(tb[it], cb), where="mid", label=label)

    ax[0].set_xlabel("decay-to-detector distance [km]")
    ax[1].set_xlabel("HNL impact parameter to detector [m]")
    pooled = (f"$m_N$={ks[0][0]:.0f} GeV, $U^2$={ks[0][1]:.1e}" if len(ks) == 1
              else f"{len(ks)} (m_N,U2) points pooled")
    for a in ax:
        a.set_ylabel(ylabel)
        a.set_yscale("log")
        a.legend(fontsize=8, title="threshold")
    ax[0].set_title(CHANNELS[result.channel]["desc"])
    ax[1].set_title(f"{pooled}\n{result.detector_label()}", fontsize=9)
    fig.tight_layout()

    if output:
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        fig.savefig(output, dpi=150, bbox_inches="tight")
        print(f"[hnl_efficiency] saved {output}")
    return fig, ax


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--channel", choices=list(CHANNELS), default="dimuon")
    p.add_argument("--scan-dir", default=None, help="override channel's default dir")
    p.add_argument("--min-pe", type=float, nargs="+", default=[MIN_PHOTOELECTRONS],
                   help="one or more PE thresholds (evaluated in a single pass)")
    p.add_argument("--pde", type=float, default=SIPM_PDE)
    p.add_argument("--masses", type=float, nargs="+", default=None)
    p.add_argument("--u2", type=float, nargs="+", default=None, dest="u2_values")
    p.add_argument("--quantity", choices=["events", "efficiency"], default="events")
    p.add_argument("--twod", action="store_true",
                   help="make the 2D distance x impact-parameter map instead of "
                        "the 1D projections")
    p.add_argument("--detectors", nargs="+", default=["all"],
                   help='detector selection: "all", "best", or one/more detector '
                        'IDs (e.g. --detectors 2, or --detectors 2 5 8)')
    p.add_argument("--no-decay-pos-prob", action="store_true",
                   help="drop the decay_pos_probability importance weight "
                        "(reproduces the collect_results*.py convention)")
    p.add_argument("--output", default=None, help="figure path (e.g. figures/eff.png)")
    args = p.parse_args()

    # --detectors: "all"/"best" string, or integer detector ID(s)
    if args.detectors in (["all"], ["best"]):
        detectors = args.detectors[0]
    else:
        try:
            detectors = [int(d) for d in args.detectors]
        except ValueError:
            p.error('--detectors must be "all", "best", or integer detector IDs')

    res = compute(channel=args.channel, scan_dir=args.scan_dir,
                  min_pe=args.min_pe, pde=args.pde,
                  masses=args.masses, u2_values=args.u2_values,
                  detectors=detectors,
                  use_decay_pos_prob=not args.no_decay_pos_prob)
    print("[hnl_efficiency] pooled over all selected points:")
    for pe in res.thresholds_pe:
        print(f"    {pe:5.0f} PE:  avg tag efficiency={average_efficiency(res, threshold_pe=pe):.3e}"
              f"   expected tagged events={expected_events(res, threshold_pe=pe):.3g}")

    if args.output:
        multi = len(res.thresholds_pe) > 1
        if args.twod:
            if multi:      # one map per threshold, PE appended to the filename
                base, ext = os.path.splitext(args.output)
                for pe in res.thresholds_pe:
                    plot_efficiency_2d(res, quantity=args.quantity, threshold_pe=pe,
                                       output=f"{base}_{pe:.0f}pe{ext}")
            else:
                plot_efficiency_2d(res, quantity=args.quantity, output=args.output)
        elif multi:        # overlay thresholds on one 1D figure
            plot_threshold_comparison(res, quantity=args.quantity, output=args.output)
        else:
            plot_efficiency(res, quantity=args.quantity, output=args.output)


if __name__ == "__main__":
    _main()
