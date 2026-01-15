#!/usr/bin/env python3
"""
SIREN Simulation for Balloon HNL Detection Experiment

This script runs a full SIREN simulation with:
- Muon primaries (5 TeV beam traveling upward)
- HNL production via muon-nucleon DIS (mu + N -> HNL + X)
- HNL decay in atmosphere
- Event output for Cherenkov analysis

The muon→HNL cross section is handled by SIREN, which properly computes
the HNL angular distribution from DIS kinematics.

Usage:
    python Balloon_SIREN_Simulation.py --m_hnl 20 --u2 1e-10 --n_events 10000
"""

import os
import sys
import argparse
import numpy as np
import awkward as ak

# Add SIREN to path
SIREN_dir = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/LIV2/sources/SIREN"
sys.path.insert(0, os.path.join(SIREN_dir, "build"))

import siren
from siren.SIREN_Controller import SIREN_Controller

# Base directory for this simulation
BALLOON_dir = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/Pheno/HNLs/NeutrinoFactoryHNL/SIREN_Balloon"

# Physical constants
N_muon_decays = 1e22  # muons in beam lifetime


def RunMuonHNLSimulation(events_to_inject, outfile, experiment,
                         m4, Ue4, Umu4, Utau4,
                         muon_input_file,
                         lumi=3000, seed=42):
    """
    Run SIREN simulation for HNL production from muon beam.
    
    This function mirrors the structure of RunHNLSimulation from the
    Geneva Lake_Geneva_Neutrinos/SIREN_Simulation.py, but uses muon
    primaries instead of neutrino primaries.
    
    Parameters
    ----------
    events_to_inject : int
        Number of events to simulate
    outfile : str
        Output file path (without extension)
    experiment : str
        Experiment name (should match detector file, e.g., "Balloon")
    m4 : float
        HNL mass in MeV
    Ue4, Umu4, Utau4 : float
        Mixing angles squared for each flavor
    muon_input_file : str
        Path to muon beam input file (CSV format)
    lumi : float
        Luminosity factor for event weighting
    seed : int
        Random seed
    """
    
    # Define the controller
    controller = SIREN_Controller(events_to_inject, experiment, seed=seed)
    
    # Primary particle type: MUON
    # Use mu- (PDG 13) or mu+ (PDG -13) depending on beam
    primary_type = siren.dataclasses.Particle.ParticleType.MuMinus
    
    # Secondary particle: HNL
    hnl_type = siren.dataclasses.Particle.ParticleType.N4
    
    # =========================================================================
    # CROSS SECTION SETUP
    # =========================================================================
    # 
    # TODO: Implement MuonHNLDISFromSpline class in SIREN
    # 
    # This should be analogous to HNLDISFromSpline but for muon primaries:
    #   mu + N -> HNL + X
    # 
    # The cross section class should:
    # 1. Load differential cross section splines dσ/dxdy for muon-nucleon DIS
    # 2. Properly sample the HNL kinematics (energy, angle) from DIS variables
    # 3. Handle the threshold behavior for HNL production
    # 
    # Expected interface (matching HNLDISFromSpline pattern):
    #
    #   cross_section_model = "MuonHNLDISSplines"
    #   xsfiledir = siren.utilities.get_cross_section_model_path(cross_section_model)
    #   
    #   m4_str = f"{int(m4):07d}"
    #   DIS_xs = siren.interactions.MuonHNLDISFromSpline(
    #       os.path.join(xsfiledir, "M_0000000MeV/dsdxdy-mu-N-nc.fits"),
    #       os.path.join(xsfiledir, f"M_{m4_str}MeV/sigma-mu-N-nc.fits"),
    #       float(m4) * 1e-3,  # mass in GeV
    #       [Ue4, Umu4, Utau4],
    #       siren.utilities.Constants.isoscalarMass,
    #       1,  # minQ2
    #       [primary_type],
    #       [target_type],
    #   )
    #
    # =========================================================================
    
    # Target type
    target_type = siren.dataclasses.Particle.ParticleType.Nucleon
    
    # Placeholder: Load the muon→HNL DIS cross section
    # REPLACE THIS SECTION when MuonHNLDISFromSpline is implemented
    cross_section_model = "MuonHNLDISSplines"  # TODO: create this
    xsfiledir = siren.utilities.get_cross_section_model_path(cross_section_model)
    
    m4_str = f"{int(m4):07d}"
    DIS_xs = siren.interactions.MuonHNLDISFromSpline(
        os.path.join(xsfiledir, "M_0000000MeV/dsdxdy-mu-N-nc.fits"),
        os.path.join(xsfiledir, f"M_{m4_str}MeV/sigma-mu-N-nc.fits"),
        float(m4) * 1e-3,  # mass in GeV
        [Ue4, Umu4, Utau4],
        siren.utilities.Constants.isoscalarMass,
        1,  # minQ2
        [primary_type],
        [target_type],
    )
    
    print(f"Cross section threshold: {DIS_xs.InteractionThreshold(siren.dataclasses.InteractionRecord())}")
    
    DIS_interaction_collection = siren.interactions.InteractionCollection(primary_type, [DIS_xs])
    
    # =========================================================================
    # PRIMARY DISTRIBUTIONS
    # =========================================================================
    
    primary_injection_distributions = {}
    primary_physical_distributions = {}
    
    # Load muon beam from external file
    # File format: PDG,hPDG,x0,y0,z0,thx,thy,E,wgt,px,py,pz
    assert os.path.isfile(muon_input_file), f"Muon input file not found: {muon_input_file}"
    
    energy_threshold = DIS_xs.InteractionThreshold(siren.dataclasses.InteractionRecord())
    primary_external_dist = siren.distributions.PrimaryExternalDistribution(
        muon_input_file, 
        1.1 * energy_threshold  # filter muons above threshold
    )
    primary_injection_distributions["external"] = primary_external_dist
    
    num_input_events = primary_external_dist.GetPhysicalNumEvents()
    print(f"Number of input muons above threshold: {num_input_events}")
    
    # Position distribution: bounded to target region
    # Target is underground, from z=-400m to z=0 (surface)
    # Using max_length to bound the interaction region
    max_length = 1000  # meters - covers the target region
    position_distribution = siren.distributions.PrimaryBoundedVertexDistribution(max_length)
    primary_injection_distributions["position"] = position_distribution
    
    # =========================================================================
    # SET UP PROCESSES
    # =========================================================================
    
    controller.SetProcesses(
        primary_type, 
        primary_injection_distributions, 
        primary_physical_distributions,
        [hnl_type],  # secondary types from primary interaction
        [[]],        # secondary injection distributions
        [[]]         # secondary physical distributions
    )
    
    # =========================================================================
    # HNL DECAY MODEL
    # =========================================================================
    
    hnl_decay = siren.interactions.HNLDecay(
        float(m4) * 1e-3,  # mass in GeV
        [Ue4, Umu4, Utau4],
        siren.interactions.HNLDecay.ChiralNature.Majorana
    )
    Decay_interaction_collection = siren.interactions.InteractionCollection(hnl_type, [hnl_decay])
    
    # Set interactions
    controller.SetInteractions(primary_interaction_collection=DIS_interaction_collection)
    controller.SetInteractions(secondary_interaction_collections=[Decay_interaction_collection])
    
    # =========================================================================
    # RUN SIMULATION
    # =========================================================================
    
    controller.Initialize()
    
    # Stopping condition: stop tracking when we get final state particles
    def stop(datum, i):
        secondary_type = datum.record.signature.secondary_types[i]
        return (secondary_type != siren.dataclasses.Particle.ParticleType.N4 and
                secondary_type != siren.dataclasses.Particle.ParticleType.N4Bar and
                secondary_type != siren.dataclasses.Particle.ParticleType.WPlus and
                secondary_type != siren.dataclasses.Particle.ParticleType.WMinus and
                secondary_type != siren.dataclasses.Particle.ParticleType.Z0)
    
    controller.SetInjectorStoppingCondition(stop)
    
    print("Generating events...")
    controller.GenerateEvents(fill_tables_at_exit=False, verbose=True)
    
    print("Saving events...")
    controller.SaveEvents(
        outfile,
        save_int_probs=True,
        save_int_params=True,
        save_survival_probs=True,
        fill_tables_at_exit=False,
        hdf5=False,
        siren_events=False,
        verbose=True
    )
    
    # =========================================================================
    # POST-PROCESSING: COMPUTE WEIGHTS
    # =========================================================================
    
    data = ak.from_parquet(f"{outfile}.parquet")
    
    # Compute event weights
    weights = np.array(
        np.squeeze(data.wgt) * lumi * 1000 * 
        np.prod(data.int_probs, axis=-1) * 
        np.prod(data.survival_probs, axis=-1)
    )
    weights *= num_input_events / events_to_inject  # correct for sampled events
    data["weights"] = weights
    
    # Save with weights
    ak.to_parquet(data, f"{outfile}.parquet")
    
    print(f"Saved {len(data)} events to {outfile}.parquet")
    
    return data


def main():
    parser = argparse.ArgumentParser(
        description='SIREN Balloon HNL Simulation with Muon Primaries'
    )
    parser.add_argument('--n_events', type=int, default=10000,
                        help='Number of events to inject')
    parser.add_argument('--m_hnl', type=float, required=True,
                        help='HNL mass in GeV')
    parser.add_argument('--u2', type=float, required=True,
                        help='Muon mixing angle squared (Umu4^2)')
    parser.add_argument('--ue4', type=float, default=0.0,
                        help='Electron mixing squared (default: 0)')
    parser.add_argument('--utau4', type=float, default=0.0,
                        help='Tau mixing squared (default: 0)')
    parser.add_argument('--muon_input', type=str, 
                        default=os.path.join(BALLOON_dir, "inputs/muon_beam_5TeV.csv"),
                        help='Path to muon input file')
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(BALLOON_dir, "output"),
                        help='Output directory')
    parser.add_argument('--experiment', type=str, default='Balloon',
                        help='Experiment name (must match detector file)')
    parser.add_argument('--lumi', type=float, default=3000,
                        help='Luminosity factor')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Convert mass to MeV for SIREN
    m4_MeV = args.m_hnl * 1000
    
    # Output filename
    outfile = os.path.join(
        args.output_dir,
        f"balloon_m{args.m_hnl:.2f}GeV_U2_{args.u2:.0e}"
    )
    
    print("="*70)
    print("SIREN Balloon HNL Simulation")
    print("="*70)
    print(f"HNL mass:        {args.m_hnl} GeV ({m4_MeV} MeV)")
    print(f"Mixing Umu4^2:   {args.u2:.2e}")
    print(f"Mixing Ue4^2:    {args.ue4:.2e}")
    print(f"Mixing Utau4^2:  {args.utau4:.2e}")
    print(f"Events:          {args.n_events}")
    print(f"Muon input:      {args.muon_input}")
    print(f"Experiment:      {args.experiment}")
    print(f"Output:          {outfile}")
    print("="*70)
    print()
    
    # Run simulation
    data = RunMuonHNLSimulation(
        events_to_inject=args.n_events,
        outfile=outfile,
        experiment=args.experiment,
        m4=m4_MeV,
        Ue4=args.ue4,
        Umu4=args.u2,
        Utau4=args.utau4,
        muon_input_file=args.muon_input,
        lumi=args.lumi,
        seed=args.seed
    )
    
    print("\nSimulation complete!")
    print(f"Output saved to: {outfile}.parquet")


if __name__ == '__main__':
    main()
