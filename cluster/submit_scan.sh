#!/bin/bash
#SBATCH --job-name=hnl_balloon
#SBATCH --partition=serial_requeue,shared,sapphire,arguelles_delgado
#SBATCH --array=0-279
#SBATCH --time=0-2:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=8
#SBATCH --output=/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/Pheno/HNLs/NeutrinoFactoryHNL/cluster/logs/scan_%a.out
#SBATCH --error=/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/Pheno/HNLs/NeutrinoFactoryHNL/cluster/logs/scan_%a.err
#SBATCH --requeue

# 1800 tasks = 14 masses x 20 U2 batches (1 U2 points each)
N_U2_BATCHES=20

# Setup spack environment
source /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/spack/share/spack/setup-env.sh
spack env activate lienv

cd /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/Pheno/HNLs/NeutrinoFactoryHNL

python3 cluster/run_scan.py ${MASS_IDX} ${U2_BATCH_IDX}
