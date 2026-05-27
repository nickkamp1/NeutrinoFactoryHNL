#!/bin/bash
#SBATCH --job-name=hnl_dimuon
#SBATCH --partition=serial_requeue,shared,sapphire,arguelles_delgado
#SBATCH --array=0-319
#SBATCH --time=0-2:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=8
#SBATCH --output=/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/Pheno/HNLs/NeutrinoFactoryHNL/cluster/logs/dimuon_%a.out
#SBATCH --error=/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/Pheno/HNLs/NeutrinoFactoryHNL/cluster/logs/dimuon_%a.err
#SBATCH --requeue

# 320 tasks = 16 masses x 20 U2 batches (5 U2 points each, ~19 min/point => ~95 min/batch).
# Keep U2_BATCH_SIZE=5 in run_scan_dimuon.py and collect_results_dimuon.py in sync:
#   N_U2_BATCHES = ceil(len(U2_RANGE)/U2_BATCH_SIZE) = ceil(100/5) = 20
#   array range  = N_MASSES*N_U2_BATCHES - 1 = 16*20 - 1 = 319
N_U2_BATCHES=20

MASS_IDX=$(( SLURM_ARRAY_TASK_ID / N_U2_BATCHES ))
U2_BATCH_IDX=$(( SLURM_ARRAY_TASK_ID % N_U2_BATCHES ))

# Setup spack environment (SIREN is installed in lienv)
source /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/spack/share/spack/setup-env.sh
spack env activate lienv

cd /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/Pheno/HNLs/NeutrinoFactoryHNL

python3 cluster/run_scan_dimuon.py ${MASS_IDX} ${U2_BATCH_IDX}
