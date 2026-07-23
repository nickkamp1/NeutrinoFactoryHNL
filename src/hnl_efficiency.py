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

    def keys(self, masses=None, u2min=None, u2max=None):
        """Selected (m_N, U2) keys, filtered by mass (exact list) and by a U2
        RANGE [u2min, u2max] (either bound None = open).

        NOTE: mass matching uses atol=0 -- the default np.isclose atol (1e-8)
        would swamp the small values in play.
        """
        out = []
        for (m_N, U2) in sorted(self.points):
            if masses is not None and not np.any(
                    np.isclose(m_N, np.atleast_1d(masses), rtol=1e-3, atol=0)):
                continue
            if u2min is not None and U2 < u2min:
                continue
            if u2max is not None and U2 > u2max:
                continue
            out.append((m_N, U2))
        return out

    def masses_in(self, masses=None, u2min=None, u2max=None):
        """Sorted unique HNL masses among the selected keys."""
        return sorted({m for (m, _U2) in self.keys(masses, u2min, u2max)})

    def _sum(self, field_name, masses=None, u2min=None, u2max=None):
        """Sum a per-point field over the selected keys (pools the U2 range)."""
        ks = self.keys(masses, u2min, u2max)
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
            if u2_values is not None and not np.any(
                    np.isclose(U2, u2_values, rtol=1e-3, atol=0)):
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
def average_efficiency(result, masses=None, u2min=None, u2max=None, threshold_pe=None):
    """Scalar rate-weighted average tagging efficiency = expected tagged events
    / expected candidate events, over the selected points (masses + U2 range),
    at the threshold nearest ``threshold_pe`` (default: first)."""
    it = result.thr_index(threshold_pe)
    tag = result._sum("tag_tot", masses, u2min, u2max)[it]
    cand = result._sum("cand_tot", masses, u2min, u2max)
    return float(tag / cand) if cand > 0 else 0.0


def expected_events(result, masses=None, u2min=None, u2max=None, threshold_pe=None):
    """Total expected tagged events over the selected points (masses + U2 range),
    at the threshold nearest ``threshold_pe`` (default: first)."""
    it = result.thr_index(threshold_pe)
    return float(result._sum("tag_tot", masses, u2min, u2max)[it])


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


_LINESTYLES = ["-", "--", ":", "-."]


def _y_from(tag, cand, quantity):
    """Per-bin y: efficiency ratio (NaN where no candidates) or raw expected
    tagged events."""
    if quantity == "efficiency":
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(cand > 0, tag / cand, np.nan)
    return tag


def _mass_colors(masses):
    import matplotlib.pyplot as plt
    cyc = plt.rcParams["axes.prop_cycle"].by_key().get(
        "color", [f"C{i}" for i in range(10)])
    return {m: cyc[i % len(cyc)] for i, m in enumerate(masses)}


def plot_efficiency(result, quantity="events", masses=None, u2min=None, u2max=None,
                    thresholds_pe=None, output=None, ax=None):
    """Two-panel figure: quantity vs distance (left) and vs HNL impact parameter
    (right).

    One line per HNL mass (distinguished by COLOR) and per PE threshold
    (distinguished by LINESTYLE).  Within each mass the U2 range [u2min, u2max]
    is pooled (both bounds None -> all U2).

    quantity : {"events", "efficiency"}
        "events"     -> expected tagged events per bin (absolute).
        "efficiency" -> tagged / candidate per bin (0-1).
    thresholds_pe : iterable or None
        Thresholds drawn as distinct linestyles (nearest matches; default: all).
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    _apply_style()

    ms = result.masses_in(masses, u2min, u2max)
    if not ms:
        raise ValueError("no (m_N, U2) points match the selection")
    its = (list(range(len(result.thresholds_pe))) if thresholds_pe is None
           else [result.thr_index(pe) for pe in np.atleast_1d(thresholds_pe)])
    colors = _mass_colors(ms)

    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    else:
        fig = ax[0].figure

    dx, bx = result.dist_centers / 1e3, result.b_centers
    for m in ms:
        td = result._sum("tag_dist", [m], u2min, u2max)      # (n_thr, n_dist)
        cd = result._sum("cand_dist", [m], u2min, u2max)     # (n_dist,)
        tb = result._sum("tag_b", [m], u2min, u2max)
        cb = result._sum("cand_b", [m], u2min, u2max)
        for j, it in enumerate(its):
            ls = _LINESTYLES[j % len(_LINESTYLES)]
            ax[0].step(dx, _y_from(td[it], cd, quantity), where="mid",
                       color=colors[m], ls=ls)
            ax[1].step(bx, _y_from(tb[it], cb, quantity), where="mid",
                       color=colors[m], ls=ls)

    ax[0].set_ylabel("expected tagged events / bin" if quantity == "events"
                     else "dimuon tagging efficiency")
    ax[0].set_xlabel("decay-to-detector distance [km]")
    ax[1].set_xlabel("HNL impact parameter [m]")
    ax[0].set_xlim(result.dist_edges[0]*1e-3, result.dist_edges[-1]*1e-3)
    ax[1].set_xlim(result.b_edges[0], result.b_edges[-1])
    for a in ax:
        a.set_yscale("log")
        if quantity == "efficiency":
            a.set_ylim(1e-6, 1)

    # two legends: color = HNL mass (left panel), linestyle = threshold (right)
    ax[0].legend(handles=[Line2D([], [], color=colors[m], ls="-",
                                 label=f"$m_N$={m:.0f} GeV") for m in ms],
                 fontsize=8, title="HNL mass")
    ax[1].legend(handles=[Line2D([], [], color="k",
                                 ls=_LINESTYLES[j % len(_LINESTYLES)],
                                 label=f"{result.thresholds_pe[it]:.0f} PE")
                          for j, it in enumerate(its)],
                 fontsize=8, title="threshold")

    #ax[0].set_title(CHANNELS[result.channel]["desc"])
    #ax[1].set_title(f"{result.config.title()}\n{result.detector_label()}", fontsize=9)
    fig.tight_layout()

    if output:
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        fig.savefig(output, dpi=150, bbox_inches="tight")
        print(f"[hnl_efficiency] saved {output}")
    return fig, ax


def plot_efficiency_2d(result, quantity="efficiency", masses=None, u2min=None,
                       u2max=None, threshold_pe=None, ncols=3, output=None,
                       axes=None, cmap="viridis"):
    """Grid of 2D (distance, HNL impact parameter) maps, ONE PANEL PER HNL MASS.

    Within each mass panel the U2 range [u2min, u2max] is pooled, at the single
    threshold nearest ``threshold_pe`` (default: first).  A shared color scale
    and colorbar make the mass panels directly comparable.  No-candidate bins
    are left blank.

    quantity : {"events", "efficiency"}
    ncols : int
        Panels per row.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    _apply_style()

    it = result.thr_index(threshold_pe)
    ms = result.masses_in(masses, u2min, u2max)
    if not ms:
        raise ValueError("no (m_N, U2) points match the selection")

    x, y = result.dist_edges / 1e3, result.b_edges

    zs = {}
    for m in ms:
        tag2d = result._sum("tag_2d", [m], u2min, u2max)[it]
        cand2d = result._sum("cand_2d", [m], u2min, u2max)
        if quantity == "efficiency":
            with np.errstate(divide="ignore", invalid="ignore"):
                zs[m] = np.where(cand2d > 0, tag2d / cand2d, np.nan)
        else:
            zs[m] = np.where(tag2d > 0, tag2d, np.nan)

    # shared color normalization across mass panels
    if quantity == "efficiency":
        norm, cbar_label = LogNorm(vmin=1e-6, vmax=1), "tagging efficiency"
    else:
        finite = np.concatenate([z[np.isfinite(z)] for z in zs.values()
                                 if np.isfinite(z).any()] or [np.array([1.0])])
        vmax = finite.max()
        norm = LogNorm(vmin=vmax * 1e-4, vmax=vmax)
        cbar_label = "expected tagged events / bin"

    nrows = int(np.ceil(len(ms) / ncols))
    ncol = min(ncols, len(ms))
    if axes is None:
        fig, axes = plt.subplots(nrows, ncol, squeeze=False,
                                 figsize=(4.3 * ncol, 3.7 * nrows),
                                 layout="constrained")
    else:
        fig = np.ravel(axes)[0].figure
    axflat = np.ravel(axes)

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(alpha=0.0)
    mesh = None
    for k, m in enumerate(ms):
        a = axflat[k]
        # z is (n_dist, n_b); pcolormesh wants (n_b, n_dist) -> transpose
        mesh = a.pcolormesh(x, y, np.ma.masked_invalid(zs[m].T), cmap=cmap_obj,
                            norm=norm, shading="flat")
        a.set_title(f"$m_N$={m:.0f} GeV", fontsize=9)
        a.set_xlabel("distance [km]")
        a.set_ylabel("impact parameter [m]")
    for k in range(len(ms), len(axflat)):     # hide any unused panels
        axflat[k].axis("off")

    cb = fig.colorbar(mesh, ax=axes, fraction=0.046, pad=0.02)
    cb.set_label(cbar_label)
    fig.suptitle(f"{CHANNELS[result.channel]['desc']}  --  threshold "
                 f"{result.thresholds_pe[it]:.0f} PE, {result.detector_label()}",
                 fontsize=10)

    if output:
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        fig.savefig(output, dpi=150, bbox_inches="tight")
        print(f"[hnl_efficiency] saved {output}")
    return fig, axes


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
    p.add_argument("--masses", type=float, nargs="+", default=None,
                   help="HNL masses to include (one line/panel each; default all)")
    p.add_argument("--u2min", type=float, default=None,
                   help="lower U2 bound to pool over (default: no bound)")
    p.add_argument("--u2max", type=float, default=None,
                   help="upper U2 bound to pool over (default: no bound)")
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
                  masses=args.masses, detectors=detectors,
                  use_decay_pos_prob=not args.no_decay_pos_prob)
    print(f"[hnl_efficiency] pooled over U2 in "
          f"[{args.u2min}, {args.u2max}]:")
    for pe in res.thresholds_pe:
        print(f"    {pe:5.0f} PE:  avg tag efficiency="
              f"{average_efficiency(res, u2min=args.u2min, u2max=args.u2max, threshold_pe=pe):.3e}"
              f"   expected tagged events="
              f"{expected_events(res, u2min=args.u2min, u2max=args.u2max, threshold_pe=pe):.3g}")

    if args.output:
        if args.twod:
            # 2D is single-threshold: one figure (of per-mass panels) per threshold
            multi = len(res.thresholds_pe) > 1
            base, ext = os.path.splitext(args.output)
            for pe in res.thresholds_pe:
                out = f"{base}_{pe:.0f}pe{ext}" if multi else args.output
                plot_efficiency_2d(res, quantity=args.quantity, u2min=args.u2min,
                                   u2max=args.u2max, threshold_pe=pe, output=out)
        else:
            # 1D: one line/mass (color), one linestyle/threshold -> single figure
            plot_efficiency(res, quantity=args.quantity, u2min=args.u2min,
                            u2max=args.u2max, output=args.output)


if __name__ == "__main__":
    _main()
