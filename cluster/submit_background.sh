#!/bin/bash
#SBATCH --job-name=hnl_balloon_bkg
#SBATCH --partition=serial_requeue,shared,sapphire,arguelles_delgado
#SBATCH --time=0-1:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=8
#SBATCH --output=/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/Pheno/HNLs/NeutrinoFactoryHNL/cluster/logs/scan_bkg.out
#SBATCH --error=/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/Pheno/HNLs/NeutrinoFactoryHNL/cluster/logs/scan_bkg.err
#SBATCH --requeue

# Setup spack environment
source /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/spack/share/spack/setup-env.sh
spack env activate lienv

cd /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/Pheno/HNLs/NeutrinoFactoryHNL

python3 cluster/run_scan.py --bkg
