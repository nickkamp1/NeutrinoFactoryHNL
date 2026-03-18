#!/bin/bash
#SBATCH --job-name=hnl_balloon
#SBATCH --partition=serial_requeue,shared,sapphire
#SBATCH --array=0-359
#SBATCH --time=0-3:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=8
#SBATCH --output=cluster/logs/scan_%a.out
#SBATCH --error=cluster/logs/scan_%a.err
#SBATCH --requeue

# 360 tasks = 18 masses x 20 U2 batches (5 U2 points each)
N_U2_BATCHES=20

MASS_IDX=$(( SLURM_ARRAY_TASK_ID / N_U2_BATCHES ))
U2_BATCH_IDX=$(( SLURM_ARRAY_TASK_ID % N_U2_BATCHES ))

# Setup spack environment
source /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/spack/share/spack/setup-env.sh
spack env activate lienv

cd /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/Pheno/HNLs/NeutrinoFactoryHNL

python3 cluster/run_scan.py ${MASS_IDX} ${U2_BATCH_IDX}
