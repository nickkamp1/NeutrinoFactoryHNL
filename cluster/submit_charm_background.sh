#!/bin/bash
#SBATCH --job-name=hnl_charm_bkg
#SBATCH --partition=serial_requeue,shared,sapphire,arguelles_delgado
#SBATCH --array=0-49
#SBATCH --time=0-2:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=8
#SBATCH --output=/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/Pheno/HNLs/NeutrinoFactoryHNL/cluster/logs/charm_bkg_%a.out
#SBATCH --error=/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/Pheno/HNLs/NeutrinoFactoryHNL/cluster/logs/charm_bkg_%a.err
#SBATCH --requeue

# Charm dimuon background: NO parameter scan -- each array task is an independent
# random SEED that accumulates MC statistics.  50 seeds x N_SAMPLES each; combine
# the per-seed data/scan_results_balloon_charm/charm_bkg_seed_*.npz downstream.
# The SLURM array index IS the seed.

# Setup spack environment (SIREN + charm splines are installed in lienv)
source /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/spack/share/spack/setup-env.sh
spack env activate lienv

# Charm splines (override the module default here if they move)
export SIREN_CHARM_SPLINE_DIR=/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/pzhelnin/DiMuons/Simulation/Resources/Splines/M_Muon

cd /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/Pheno/HNLs/NeutrinoFactoryHNL

python cluster/run_charm_background.py $SLURM_ARRAY_TASK_ID
