#!/bin/bash
#SBATCH --job-name=hnl_img_spread
#SBATCH --partition=serial_requeue,shared,sapphire,arguelles_delgado
#SBATCH --array=0-15
#SBATCH --time=0-4:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=8
#SBATCH --output=/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/Pheno/HNLs/NeutrinoFactoryHNL/cluster/logs/img_spread_%a.out
#SBATCH --error=/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/Pheno/HNLs/NeutrinoFactoryHNL/cluster/logs/img_spread_%a.err
#SBATCH --requeue

# One array task per HNL mass (16 masses).  The array index IS the mass index.
# Produces data/muon_image_spread_<m>.npz with per-event opening_deg for the
# HNL-vs-charm discriminator comparison (src/charm_vs_hnl.py).

source /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/spack/share/spack/setup-env.sh
spack env activate lienv

cd /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/Pheno/HNLs/NeutrinoFactoryHNL

python cluster/run_muon_image_spread.py $SLURM_ARRAY_TASK_ID
