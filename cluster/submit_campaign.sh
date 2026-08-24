#!/bin/bash
# Launch the expanded pixel + decay-nu campaign.
#
# Matrix: detector DISTANCES {5, 20, 50, 100} km (central detectors 11/2/5/8 in
# run_*.py's ALL_DETECTORS) x R_det {2, 4} m, for three samples each:
#   * HNL signal            (23 masses, cluster/run_hnl_signal.py)
#   * charm scattering-nu   (50 seeds,  cluster/run_charm_background.py)
#   * charm decay-nu (nu_mu-bar; 50 seeds, CHARM_MODE=decay)
#
# File-name tags: R=2 is UNTAGGED (keeps old names); R=4 adds _R4/R4_; decay adds
# _decay.  Detector tag det<i> (i = index into ALL_DETECTORS).
#
# Run from the project ROOT on the cluster AFTER the smoke tests pass:
#   bash cluster/submit_campaign.sh
set -euo pipefail
ROOT=/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/Pheno/HNLs/NeutrinoFactoryHNL
LOG=$ROOT/cluster/logs
mkdir -p "$LOG"
cd "$ROOT"

# central detectors: 1 km ->14, 5 km -> 11, 20 km -> 2, 50 km -> 5, 100 km -> 8
DETS=(14 11 2 5 8)
RADII=(2 4)

for R in "${RADII[@]}"; do
  for DET in "${DETS[@]}"; do
    tag="d${DET}_R${R}"

    # --- HNL signal (23 masses) ---
    sbatch --job-name="hnl_${tag}" --array=0-22 \
           --output="$LOG/img_${tag}_%a.out" --error="$LOG/img_${tag}_%a.err" \
           --export=ALL,DET_INDEX=$DET,R_DET=$R \
           cluster/submit_hnl_signal.sh

    # --- charm scattering-neutrino background (50 seeds) ---
    sbatch --job-name="charm_${tag}" --array=0-49 \
           --output="$LOG/charm_${tag}_%a.out" --error="$LOG/charm_${tag}_%a.err" \
           --export=ALL,DET_INDEX=$DET,R_DET=$R,CHARM_MODE=scattering \
           cluster/submit_charm_background.sh

    # --- charm decay-neutrino (nu_mu-bar) background (50 seeds) ---
    sbatch --job-name="charmdec_${tag}" --array=0-49 \
           --output="$LOG/charmdec_${tag}_%a.out" --error="$LOG/charmdec_${tag}_%a.err" \
           --export=ALL,DET_INDEX=$DET,R_DET=$R,CHARM_MODE=decay \
           cluster/submit_charm_background.sh
  done
done

echo "Submitted 8 configs x {HNL 23 masses, charm-scatt 50 seeds, charm-decay 50 seeds}."
