#!/bin/bash
#SBATCH --job-name=balloon_hnl
#SBATCH --output=logs/balloon_%A_%a.out
#SBATCH --error=logs/balloon_%A_%a.err
#SBATCH --time=4:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --partition=shared
#SBATCH --array=0-29

# Balloon HNL Detection Simulation - SLURM Array Job
# Uses SIREN with muon primaries and MuonHNLDISFromSpline cross section

# Load modules
module load python/3.10.13-fasrc01

# Base directory
BASEDIR=/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/Pheno/HNLs/NeutrinoFactoryHNL/SIREN_Balloon
SIREN_DIR=/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/LIV2/sources/SIREN

# Add SIREN to path
export PYTHONPATH=${SIREN_DIR}/build:$PYTHONPATH

# Create logs directory if needed
mkdir -p ${BASEDIR}/logs

# Mass scan points (1-100 GeV, 30 points, log-spaced)
MASSES=(1.0 1.17 1.37 1.61 1.89 2.21 2.59 3.04 3.56 4.18 
        4.89 5.74 6.72 7.88 9.24 10.83 12.69 14.87 17.43 20.43 
        23.95 28.07 32.90 38.57 45.20 52.98 62.10 72.79 85.32 100.0)

# Get mass for this array job
M_HNL=${MASSES[$SLURM_ARRAY_TASK_ID]}

# Fixed mixing angle for initial scan
U2=1e-10

# Number of events
N_EVENTS=100000

echo "Running SIREN balloon HNL simulation"
echo "  Array task: ${SLURM_ARRAY_TASK_ID}"
echo "  HNL mass: ${M_HNL} GeV"
echo "  Mixing U^2: ${U2}"
echo "  Events: ${N_EVENTS}"

# Run simulation
python ${BASEDIR}/scripts/Balloon_SIREN_Simulation.py \
    --n_events ${N_EVENTS} \
    --m_hnl ${M_HNL} \
    --u2 ${U2} \
    --muon_input ${BASEDIR}/inputs/muon_beam_5TeV.csv \
    --output_dir ${BASEDIR}/output \
    --experiment Balloon \
    --seed $((42 + SLURM_ARRAY_TASK_ID))

echo "Job completed"
