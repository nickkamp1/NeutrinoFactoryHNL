"""
Table-based mu- N -> HNL X cross section, interpolating the MadGraph5 + CT18NNLO
isoscalar scan (data/HNL_xs_2D_majorana.txt).

Replaces the crude analytic form previously in src/xs_and_decays.sigma(). The
table is computed for U_mu^2 = 1 on an isoscalar target (Earth-like); the cross
section scales linearly with the mixing U2.

Physics of the interpolation
----------------------------
sigma(E, m) is interpolated bilinearly in (log10 E, m) of log10(sigma) -- log-log
in energy keeps the ~linear-then-propagator-softened E dependence smooth, and
log in sigma keeps the steep high-mass PDF suppression smooth. Above the DIS
threshold sqrt(s) = sqrt(2 m_p E + m_p^2) the cross section is zero. Below the
lowest tabulated energy the cross section is extrapolated as sigma proportional
to E (the low-energy linear-DIS regime).
"""
import os
import numpy as np
from scipy.interpolate import LinearNDInterpolator

M_NUCLEON = 0.938  # GeV
PB_TO_M2 = 1e-40   # 1 pb = 1e-40 m^2

_DEFAULT_TABLE = os.path.join(os.path.dirname(__file__), "..", "data", "HNL_xs_2D_majorana.txt")


class HNLCrossSection:
    def __init__(self, table_path=_DEFAULT_TABLE):
        d = np.loadtxt(table_path)
        # columns: ebeam1[GeV]  mN[GeV]  sigma_iso[pb]  ...
        self.E = d[:, 0]
        self.m = d[:, 1]
        self.xs = d[:, 2]  # pb, U2=1, isoscalar
        good = self.xs > 0
        # Interpolate in (log10 E, r=m^2/s): the mass suppression aligns along r
        # (the DIS cliff sits at ~fixed r for every energy), while a mild residual
        # E-dependence -- strong at low E -- is retained by the 2nd axis.
        logE = np.log10(self.E[good])
        r = self.m[good]**2 / self._s(self.E[good])
        lxs = np.log10(self.xs[good])
        # Add an r=0 "plateau" anchor per energy (value = smallest-mass point) so
        # the (logE, r) convex hull always extends down to r=0. Without it, the
        # slightly-curved small-r boundary can leave intermediate-energy plateau
        # queries just outside the hull -> NaN.
        aE, ar, ax = [], [], []
        for E0 in np.unique(self.E[good]):
            sel = self.E[good] == E0
            i0 = np.argmin(r[sel])
            aE.append(np.log10(E0)); ar.append(0.0); ax.append(lxs[sel][i0])
        logE = np.concatenate([logE, aE])
        r = np.concatenate([r, ar])
        lxs = np.concatenate([lxs, ax])
        self._interp = LinearNDInterpolator(np.column_stack([logE, r]), lxs)
        self.E_min = self.E.min()
        self.E_max = self.E.max()

    @staticmethod
    def _s(E):
        return 2 * M_NUCLEON * E + M_NUCLEON**2

    @staticmethod
    def sqrt_s(E):
        return np.sqrt(2 * M_NUCLEON * E + M_NUCLEON**2)

    def _interp_pb(self, E, m):
        """Interpolated sigma [pb] for broadcastable arrays E, m at U2=1."""
        E, m = np.broadcast_arrays(np.asarray(E, dtype=float),
                                   np.asarray(m, dtype=float))
        shape = E.shape
        E = E.ravel(); m = m.ravel()
        out = np.zeros_like(E)
        r = m**2 / self._s(E)
        alive = m < (self.sqrt_s(E) - M_NUCLEON)          # DIS threshold
        inrange = alive & (E >= self.E_min) & (E <= self.E_max)
        if inrange.any():
            v = self._interp(np.log10(E[inrange]), r[inrange])
            out[inrange] = np.where(np.isnan(v), 0.0, 10**v)
        # low-E extrapolation: sigma(E,m) ~ sigma(E_min,m) * E/E_min (linear DIS
        # regime); anchor is evaluated at the same fixed mass -> its own r.
        low = alive & (E < self.E_min)
        if low.any():
            r_anchor = m[low]**2 / self._s(self.E_min)
            base = self._interp(np.full(low.sum(), np.log10(self.E_min)), r_anchor)
            base = np.where(np.isnan(base), 0.0, 10**base)
            out[low] = base * (E[low] / self.E_min)
        # high-E clamp: hold the top-grid value rather than returning zero
        high = alive & (E > self.E_max)
        if high.any():
            r_top = m[high]**2 / self._s(self.E_max)
            top = self._interp(np.full(high.sum(), np.log10(self.E_max)), r_top)
            out[high] = np.where(np.isnan(top), 0.0, 10**top)
        return out.reshape(shape)

    def sigma(self, E_mu, m_N, U2=1.0):
        """Cross section [m^2] for mu- N -> HNL X, isoscalar target.

        E_mu, m_N : scalar or array [GeV] (broadcast together); U2 : mixing (linear).
        Returns a float if both inputs are scalar, else an ndarray.
        """
        scalar = np.isscalar(E_mu) and np.isscalar(m_N)
        res = self._interp_pb(E_mu, m_N) * PB_TO_M2 * U2
        return float(res) if scalar else res


_default = None


def sigma(E_mu, m_N, U2=1.0):
    """Module-level convenience wrapper using the default table (lazy-loaded)."""
    global _default
    if _default is None:
        _default = HNLCrossSection()
    return _default.sigma(E_mu, m_N, U2)
