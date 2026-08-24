#!/bin/bash
# Small-N smoke test for the pixel + decay-nu campaign: confirms the new pixel
# fields populate, the 5 km detector produces events, and the decay-nu charm
# flux runs.  Writes throwaway det*_R4 / det11 / seed_999 files.
set -e
ROOT=/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/Pheno/HNLs/NeutrinoFactoryHNL
source /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/spack/share/spack/setup-env.sh
spack env activate lienv
export SIREN_CHARM_SPLINE_DIR=/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/pzhelnin/DiMuons/Simulation/Resources/Splines/M_Muon
cd "$ROOT"

echo "===== HNL smoke: DET_INDEX=2 R_DET=4 (m=20, N=20000, max=2000) ====="
DET_INDEX=2 R_DET=4 python cluster/run_hnl_signal.py 13 20000 2000
echo "===== HNL smoke: DET_INDEX=11 (5 km) R_DET=2 ====="
DET_INDEX=11 R_DET=2 python cluster/run_hnl_signal.py 13 20000 2000
echo "===== charm smoke: scattering DET_INDEX=2 R_DET=2 (seed 999, N=50000) ====="
DET_INDEX=2 R_DET=2 CHARM_MODE=scattering python cluster/run_charm_background.py 999 50000 5000
echo "===== charm smoke: decay (nu_mu-bar) DET_INDEX=2 R_DET=2 (seed 999) ====="
DET_INDEX=2 R_DET=2 CHARM_MODE=decay python cluster/run_charm_background.py 999 50000 5000

echo "===== INSPECT OUTPUTS ====="
python - <<'PY'
import numpy as np
def finite_count(a):
    a=np.asarray(a,float); return int(np.isfinite(a).sum()), a.size
def show_hnl(f):
    d=np.load(f, allow_pickle=True)
    n=len(d["weight"])
    fx,_=finite_count(d["mu1_pix_x"]); print(f"{f}\n   events={n}  mu1_pix_x finite={fx}  "
        f"beam_axis_pixel={d['beam_axis_pixel']}  meta_N_samples={d['meta_N_samples']}")
    if fx: print(f"   mu1 pixel sample (x,y): "
        f"({np.nanmedian(d['mu1_pix_x']):.2f},{np.nanmedian(d['mu1_pix_y']):.2f})")
def show_charm(f):
    d=np.load(f, allow_pickle=True)
    fx,tot=finite_count(d["mu1_pixel"][:,0]); hx,_=finite_count(d["had_pixel"][:,0])
    print(f"{f}\n   mode={d['mode']}  R_det={d['R_det']}  N_nu_per_muon={float(d['N_nu_per_muon']):.3e}  "
        f"mu1_pixel finite={fx}  had_pixel finite={hx}")
    up=np.asarray(d['int_pos'])  # just presence
    print(f"   has beam_axis_pixel={ 'beam_axis_pixel' in d.files }  same_detector sum={int(np.sum(d['same_detector']))}")
show_hnl("data/hnl_signal_det2_R4_20.npz")
show_hnl("data/hnl_signal_det11_20.npz")
show_charm("data/scan_results_balloon_charm_det2/charm_bkg_seed_999.npz")
show_charm("data/scan_results_balloon_charm_decay_det2/charm_bkg_seed_999.npz")
PY
echo "===== SMOKE DONE ====="
