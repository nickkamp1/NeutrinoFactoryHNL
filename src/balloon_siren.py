# SIREN-integrated Balloon dimuon HNL simulation.
#
# Hybrid model: HNL *production* geometry and kinematics are reused unchanged
# from HNLFluxGeometry (src/balloon.py); the HNL *decay* is sampled with SIREN's
# native HNLDecay class for the leptonic dimuon channel  N4 -> nu_mu mu- mu+.
# The Cherenkov signal is the sum of the two muon tracks (the neutrino is
# invisible and there is no hadronic shower for this leptonic channel).
#
# Requires SIREN, pip-installed into the spack "lienv" python (Python 3.10):
#     source .../spack/share/spack/setup-env.sh && spacktivate lienv
import numpy as np

from src.balloon import (
    HNLFluxGeometry,
    summarize_signal,
    muon_range_in_air,
)
from src.cherenkov import cherenkov_photons_multi_detector, cherenkov_transmission
from src.constants import R_det
from src.xs_and_decays import HNL_decay_length
from src.muon_beam_dump_helpers import muon_energy_in_earth, d_max_curved_earth

import siren
from siren import interactions, dataclasses, utilities

M_MU = 0.105658  # muon mass [GeV]
PT = dataclasses.Particle.ParticleType


class SIRENDimuonGeometry(HNLFluxGeometry):
    """HNLFluxGeometry specialised to the SIREN-sampled dimuon decay channel.

    Production (cross-section-weighted production points + HNL lab 4-momentum
    from the Momentum*.dat tables) is inherited verbatim.  Only the decay
    kinematics and the Cherenkov signal model are overridden:

      * decay kinematics  -> SIREN HNLDecay, N4 -> nu_mu mu- mu+
      * signal            -> Cherenkov from the two muon tracks (no hadron shower)

    Parameters
    ----------
    *args, **kwargs
        Forwarded to HNLFluxGeometry (e.g. E_mu, dump_depth, dump_angle).
    nature : {"Majorana", "Dirac"}
        HNL chiral nature passed to SIREN's HNLDecay.
    seed : int
        Seed for SIREN's RNG used in decay sampling.
    """

    def __init__(self, *args, nature="Majorana", seed=42, **kwargs):
        super().__init__(*args, **kwargs)
        self._rng = utilities.SIREN_random(seed)
        self._nature = (interactions.HNLDecay.ChiralNature.Majorana
                        if nature == "Majorana"
                        else interactions.HNLDecay.ChiralNature.Dirac)
        # cache: m_N -> (HNLDecay, dimuon_signature, BR_mumu)
        self._decay_cache = {}

    # ------------------------------------------------------------------ #
    # SIREN decay setup
    # ------------------------------------------------------------------ #
    def _get_decay(self, m_N):
        """Build (and cache) the SIREN HNLDecay, the N4 dimuon signature, and
        the branching ratio BR(N4 -> nu_mu mu- mu+).

        Mixing is taken to be muon-dominated ([Ue4, Umu4, Utau4] = [0, 1, 0]).
        The dimuon branching ratio is independent of the overall mixing scale
        for a single-flavour coupling, so a unit Umu4 is sufficient here; the
        absolute lifetime / decay length is handled separately by
        HNL_decay_length(m_N, U2, E_N).
        """
        if m_N in self._decay_cache:
            return self._decay_cache[m_N]

        decay = interactions.HNLDecay(float(m_N), [0.0, 1.0, 0.0], self._nature)

        dimuon_sig = None
        for s in decay.GetPossibleSignatures():
            st = list(s.secondary_types)
            if (s.primary_type == PT.N4 and len(st) == 3
                    and st[1] == PT.MuMinus and st[2] == PT.MuPlus):
                dimuon_sig = s
                break
        if dimuon_sig is None:
            raise RuntimeError(
                f"No N4 -> nu mu- mu+ signature for m_N={m_N} GeV "
                f"(channel closed below 2*m_mu?)")

        rec = dataclasses.InteractionRecord()
        rec.signature = dimuon_sig
        w_mumu = decay.TotalDecayWidth(rec)
        w_tot = decay.TotalDecayWidthAllFinalStates(rec)
        BR_mumu = w_mumu / w_tot if w_tot > 0 else 0.0

        self._decay_cache[m_N] = (decay, dimuon_sig, BR_mumu)
        return self._decay_cache[m_N]

    def dimuon_branching_ratio(self, m_N):
        """BR(N4 -> nu_mu mu- mu+) from SIREN partial widths."""
        return self._get_decay(m_N)[2]

    def sample_dimuon_decay(self, hnl_energy, hnl_dirs, m_N):
        """Sample the two muon kinematics for each HNL via SIREN.

        Parameters
        ----------
        hnl_energy : ndarray, shape (N,)
            HNL lab energies [GeV].
        hnl_dirs : ndarray, shape (N, 3)
            HNL lab directions (unit vectors).
        m_N : float
            HNL mass [GeV].

        Returns
        -------
        E_mu1, dir_mu1, E_mu2, dir_mu2 : ndarrays
            Lab-frame energies [GeV] and unit directions of the two muons.
            Index 1 is mu-, index 2 is mu+ (ordering is irrelevant for the
            symmetric Cherenkov sum).
        """
        decay, dimuon_sig, _ = self._get_decay(m_N)
        N = len(hnl_energy)

        E_mu1 = np.zeros(N)
        E_mu2 = np.zeros(N)
        dir_mu1 = np.zeros((N, 3))
        dir_mu2 = np.zeros((N, 3))

        p_mag = np.sqrt(np.maximum(hnl_energy**2 - m_N**2, 0.0))

        for i in range(N):
            p3 = p_mag[i] * hnl_dirs[i]

            rec = dataclasses.InteractionRecord()
            rec.signature = dimuon_sig
            rec.primary_mass = float(m_N)
            rec.primary_momentum = [float(hnl_energy[i]),
                                    float(p3[0]), float(p3[1]), float(p3[2])]
            rec.primary_helicity = 0.0
            rec.secondary_masses = [0.0, M_MU, M_MU]
            rec.secondary_helicities = [0.0, 0.0, 0.0]

            csdr = dataclasses.CrossSectionDistributionRecord(rec)
            decay.SampleFinalState(csdr, self._rng)
            csdr.finalize(rec)

            secs = list(rec.secondary_momenta)
            for k, (E_arr, dir_arr) in enumerate(
                    ((E_mu1, dir_mu1), (E_mu2, dir_mu2))):
                p4 = np.asarray(secs[k + 1], dtype=float)  # skip the neutrino (index 0)
                E_arr[i] = p4[0]
                pv = p4[1:]
                n = np.linalg.norm(pv)
                dir_arr[i] = pv / n if n > 0 else np.array([0.0, 0.0, 1.0])

        return E_mu1, dir_mu1, E_mu2, dir_mu2

    # ------------------------------------------------------------------ #
    # Signal computation
    # ------------------------------------------------------------------ #
    def compute_dimuon_signal_at_satellite(self, m_N, U2,
                                           detector_positions,
                                           N_samples=1000,
                                           use_energy_loss=True,
                                           max_cherenkov_events=None,
                                           uniform_gen=False):
        """Dimuon analogue of HNLFluxGeometry.compute_signal_at_satellite.

        Identical production geometry and decay-position weighting; the decay
        kinematics come from SIREN and the Cherenkov signal is the sum of the
        two muon tracks.

        Like compute_signal_at_satellite but with an extra per-muon photon
        array inserted at index 2 (mu_photon_counts, shape (2, N_det, N_samples))
        for both-muon tagging, the hadronic array always zero, and a trailing
        BR_mumu scalar.  Full layout:
            0  photon_counts            (N_det, N_samples)
            1  muon_photon_counts       (N_det, N_samples)   [sum of both muons]
            2  mu_photon_counts         (2, N_det, N_samples)[per-muon]
            3  hadronic_photon_counts   (N_det, N_samples)   [zero]
            4  prod_points, 5 decay_points, 6 decay_probability,
            7  decay_pos_probability, 8 interaction_probability,
            9  cherenkov_weight, 10 hnl_energy, 11 decay_dist,
            12 decay_length, 13 d_max, 14 BR_mumu
        The expected number of signal events is

            N_HNLs_per_muon * BR_mumu * <decay-weighted detection efficiency>

        so multiply interaction_probability by BR_mumu before passing it to
        summarize_signal as N_HNLs_per_muon (see run_dimuon_scan).
        """
        detector_positions = [np.asarray(p, dtype=float) for p in detector_positions]
        N_det = len(detector_positions)
        max_det_height = max(p[2] for p in detector_positions)

        BR_mumu = self.dimuon_branching_ratio(m_N)

        # --- 1. Production points (reused) ---
        if use_energy_loss:
            prod_points, interaction_probability, _, _ = \
                self.sample_production_points_weighted(m_N, U2, N_samples)
        else:
            from src.xs_and_decays import sigma
            HNL_xs = sigma(self.E_mu, m_N, U2)
            interaction_probability = HNL_xs * self.L_target * self.n_earth_m3
            prod_points = self.sample_production_points(N_samples)

        E_muon_local = muon_energy_in_earth(self.E_mu, prod_points[:, -1] + self.L_target)

        # --- 2. HNL production kinematics (reused; muon columns discarded) ---
        hnl_energy, hnl_dirs, _, _ = self.sample_kinematics(
            prod_points[:, -1], E_muon_local, m_N=m_N
        )

        # --- 3. Dimuon decay kinematics (SIREN) ---
        E_mu1, dir_mu1, E_mu2, dir_mu2 = self.sample_dimuon_decay(
            hnl_energy, hnl_dirs, m_N
        )

        # --- 4. Decay-position weighting (reused) ---
        decay_length = HNL_decay_length(m_N, U2, hnl_energy)
        cos_z = hnl_dirs[:, 2]
        upward = cos_z > 0
        d_max = np.where(upward, d_max_curved_earth(cos_z, max_det_height), 0.0)
        d_max = np.maximum(d_max, 0.0)

        with np.errstate(divide='ignore', invalid='ignore'):
            decay_probability = np.where(
                d_max > 0, 1.0 - np.exp(-d_max / decay_length), 0.0)

        if uniform_gen:
            decay_dist = np.random.uniform(0, d_max, N_samples)
            with np.errstate(divide='ignore', invalid='ignore'):
                decay_pos_probability = np.where(
                    decay_probability > 0,
                    np.exp(-decay_dist / decay_length) * d_max
                    / (decay_length * (1 - np.exp(-d_max / decay_length))),
                    0.0)
        else:
            u = np.random.uniform(0, 1, N_samples)
            with np.errstate(divide='ignore', invalid='ignore'):
                exp_term = np.exp(-d_max / decay_length)
                decay_dist = np.where(
                    decay_probability > 0,
                    -decay_length * np.log(1.0 - u * (1.0 - exp_term)),
                    0.0)
                decay_pos_probability = np.ones_like(decay_dist)

        decay_points = prod_points + decay_dist[:, np.newaxis] * hnl_dirs

        above_surface = decay_points[:, 2] > 0
        valid_decay = above_surface & (decay_probability > 0)

        photon_counts = np.zeros((N_det, N_samples))
        muon_photon_counts = np.zeros((N_det, N_samples))      # sum of the two muons
        mu_photon_counts = np.zeros((2, N_det, N_samples))     # per-muon, for both-tagging
        hadronic_photon_counts = np.zeros((N_det, N_samples))  # always zero (leptonic channel)

        if not np.any(valid_decay):
            return (photon_counts, muon_photon_counts, mu_photon_counts,
                    hadronic_photon_counts,
                    prod_points, decay_points, decay_probability,
                    decay_pos_probability, interaction_probability, 1.0,
                    hnl_energy, decay_dist, decay_length, d_max, BR_mumu)

        valid_indices = np.where(valid_decay)[0]
        N_valid = len(valid_indices)

        if max_cherenkov_events is not None and N_valid > max_cherenkov_events:
            eval_indices = np.random.choice(valid_indices, max_cherenkov_events, replace=False)
            cherenkov_weight = N_valid / max_cherenkov_events
        else:
            eval_indices = valid_indices
            cherenkov_weight = 1.0

        muon_kin = ((E_mu1, dir_mu1), (E_mu2, dir_mu2))

        for idx in eval_indices:
            decay_pos = decay_points[idx]
            decay_altitude = max(0.0, decay_pos[2])

            valid_dets = [i for i, dp in enumerate(detector_positions)
                          if decay_pos[2] < dp[2]]
            if not valid_dets:
                continue
            valid_det_pos = [detector_positions[i] for i in valid_dets]

            # --- Cherenkov from each of the two muons (tracked separately) ---
            for k, (E_arr, dir_arr) in enumerate(muon_kin):
                mu_dir = dir_arr[idx]
                if mu_dir[2] <= 0:
                    continue
                dir_cosine = mu_dir[2]
                zenith_angle = np.arccos(np.clip(dir_cosine, -1.0, 1.0))
                transmission = cherenkov_transmission(decay_altitude, zenith_angle)

                track_length = muon_range_in_air(
                    E_arr[idx], decay_altitude, direction_cosine=dir_cosine)
                N_track = min(1000, max(300, int(track_length / 100)))
                try:
                    N_ph = cherenkov_photons_multi_detector(
                        decay_pos, mu_dir, track_length, R_det,
                        valid_det_pos, N_psi=300, N_track=N_track)
                    N_ph = np.asarray(N_ph) * transmission
                except Exception:
                    N_ph = np.zeros(len(valid_dets))

                for j, i_det in enumerate(valid_dets):
                    mu_photon_counts[k, i_det, idx] = N_ph[j]

        muon_photon_counts = mu_photon_counts.sum(axis=0)
        photon_counts = muon_photon_counts + hadronic_photon_counts
        return (photon_counts,
                muon_photon_counts,
                mu_photon_counts,        # shape (2, N_det, N_samples): per-muon photons
                hadronic_photon_counts,
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
                BR_mumu)


# ---------------------------------------------------------------------- #
# Photon -> photoelectron threshold (informed by TRINITY, arXiv:1811.09287,
# and the Trinity Demonstrator papers 2503.11864 / 2406.08274).
#
# SiPM camera (Trinity design): 0.3 deg/pixel (the Demonstrator achieved 0.24),
# NSB ~200 MHz/pixel (5 MHz/mm^2 x ~39 mm^2 SiPM), per-pixel signal-fluctuation
# noise 1.9 PE RMS for above-horizon pixels (Demonstrator, on the ~30 ns
# shaped-signal gate).  The Demonstrator triggers on a SINGLE pixel above
# threshold; its stable dark-night operating point is 20 PE (150 DAC, <0.5 Hz
# accidental rate) -- NOT a nearest-neighbour cluster trigger.
#
# Each muon of the dimuon pair images to essentially a single pixel: ray-tracing
# the Cherenkov photons that reach the 2 m disk gives an image RMS ~0.007 deg,
# ~40x smaller than a pixel (decay altitudes 10-90 km).  So the total disk photon
# count PER MUON is a good proxy for that muon's peak-pixel count, and the
# per-pixel PE threshold is applied per muon (see run_dimuon_scan).  The two
# muons are usually in DIFFERENT pixels (lab opening angle ~m_N/E_N: median
# ~0.2 deg at m_N=5 GeV rising to ~2 deg at 70 GeV), so they are treated
# separately rather than summed.
#
# Default threshold = 20 PE to match the Demonstrator's demonstrated operating
# point; 12 PE (~6 sigma over the 1.9 PE noise) is a more optimistic alternative
# -- pass min_photoelectrons=12 to run it as a systematic.
# ---------------------------------------------------------------------- #
SIPM_PDE = 0.40             # SiPM photon detection efficiency (band-averaged, 300-1000 nm)
MIN_PHOTOELECTRONS = 20.0   # single-pixel PE threshold (Trinity Demonstrator operating point)
MIN_PHOTONS_DEFAULT = MIN_PHOTOELECTRONS / SIPM_PDE  # threshold in raw photons (~50)


def run_dimuon_scan(geom, masses, U2_grid, detector_positions,
                    N_samples=1000, min_photoelectrons=MIN_PHOTOELECTRONS,
                    pde=SIPM_PDE, max_cherenkov_events=None, verbose=True):
    """Scan (m_N, U2) and return detection efficiency, mean photons, and event
    counts per detector, threshold expressed in photoelectrons.

    The expected event count folds in the SIREN dimuon branching ratio:
        N_HNLs_per_muon_effective = interaction_probability * BR_mumu.

    Parameters
    ----------
    geom : SIRENDimuonGeometry
    masses : iterable of float
        HNL masses [GeV].
    U2_grid : iterable of float
        Mixing-squared values.
    detector_positions : list of array-like
        Detector positions [m].
    min_photoelectrons : float
        Single-pixel PE detection threshold (default 20, the Trinity
        Demonstrator operating point; pass 12 for the optimistic case).
    pde : float
        SiPM photon detection efficiency used to convert the raw photon counts
        from the Cherenkov calculation into photoelectrons.

    Returns
    -------
    dict keyed by (m_N, U2) -> dict with:
        'efficiency', 'mean_photons', 'events'         -- single-muon tag (>=1 muon
                                                          above threshold), per detector
        'events_both', 'efficiency_both'               -- BOTH muons above threshold
                                                          on the same detector (the
                                                          background-free sample)
        'BR_mumu'
    """
    min_photons = min_photoelectrons / pde  # convert PE threshold to raw-photon threshold
    results = {}
    for m_N in masses:
        for U2 in U2_grid:
            out = geom.compute_dimuon_signal_at_satellite(
                m_N, U2, detector_positions,
                N_samples=N_samples,
                max_cherenkov_events=max_cherenkov_events)
            (_photon_sum, _muon_sum, mu_photon_counts, _had, _prod, _decay,
             decay_probability, _decay_pos, interaction_probability,
             cherenkov_weight, *_rest) = out
            BR_mumu = out[-1]

            N_HNLs_eff = interaction_probability * BR_mumu

            # Single-muon tag: >=1 muon above threshold in ITS OWN pixel.  Each
            # muon images to ~one pixel (image RMS ~0.007 deg << 0.3 deg pixel)
            # and the two muons usually land in different pixels, so the correct
            # per-pixel quantity is the brighter muon, not the summed disk count.
            single_muon_max = np.maximum(mu_photon_counts[0], mu_photon_counts[1])
            eff, mean_ph, n_evt = summarize_signal(
                single_muon_max, N_HNLs_eff, N_samples,
                min_photons=min_photons,
                cherenkov_weight=cherenkov_weight,
                decay_weights=decay_probability)

            # Both-muon tag (background-free): require BOTH muons above threshold
            # on the same detector.  We build a per-detector "tagged" pseudo-count
            # (1 if both muons clear threshold, else 0) and reuse summarize_signal.
            both_tagged = ((mu_photon_counts[0] >= min_photons) &
                           (mu_photon_counts[1] >= min_photons)).astype(float)
            eff_both, _mp_both, n_evt_both = summarize_signal(
                both_tagged, N_HNLs_eff, N_samples,
                min_photons=0.5,  # both_tagged is 0/1
                cherenkov_weight=cherenkov_weight,
                decay_weights=decay_probability)

            results[(m_N, U2)] = {
                'efficiency': eff,
                'mean_photons': mean_ph,
                'events': n_evt,
                'efficiency_both': eff_both,
                'events_both': n_evt_both,
                'BR_mumu': BR_mumu,
            }
            if verbose:
                print(f"m_N={m_N:6.2f} GeV  U2={U2:.1e}  BR_mumu={BR_mumu:.3e}  "
                      f"max events={np.max(n_evt):.3g}  "
                      f"max events(both-tag)={np.max(n_evt_both):.3g}")
    return results
