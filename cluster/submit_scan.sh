#!/bin/bash
#SBATCH --job-name=hnl_balloon
#SBATCH --partition=serial_requeue,shared,sapphire
#SBATCH --array=0-1800
#SBATCH --time=0-4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=cluster/logs/scan_%a.out
#SBATCH --error=cluster/logs/scan_%a.err
#SBATCH --requeue

# 1800 tasks = 18 masses x 100 U2 batches (1 U2 points each)
N_U2_BATCHES=100

MASS_IDX=$(( SLURM_ARRAY_TASK_ID / N_U2_BATCHES ))
U2_BATCH_IDX=$(( SLURM_ARRAY_TASK_ID % N_U2_BATCHES ))
if [ $MASS_IDX -eq 18 ] && [ $U2_BATCH_IDX -eq 0 ]; then
    COMPUTE_BKG=1
else
    COMPUTE_BKG=0
fi

# Setup spack environment
source /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/spack/share/spack/setup-env.sh
spack env activate lienv

cd /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/Pheno/HNLs/NeutrinoFactoryHNL

python3 cluster/run_scan.py ${MASS_IDX} ${U2_BATCH_IDX} ${COMPUTE_BKG}
