#!/bin/bash
#SBATCH --job-name=hnl_balloon_bkg
#SBATCH --partition=serial_requeue,shared,sapphire
#SBATCH --time=0-8:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=cluster/logs/scan_bkg.out
#SBATCH --error=cluster/logs/scan_bkg.err
#SBATCH --requeue

# Setup spack environment
source /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/spack/share/spack/setup-env.sh
spack env activate lienv

cd /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/Pheno/HNLs/NeutrinoFactoryHNL

python3 cluster/run_scan.py --bkg
