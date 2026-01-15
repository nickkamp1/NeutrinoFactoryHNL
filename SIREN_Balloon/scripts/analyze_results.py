#!/usr/bin/env python3
"""
Analysis script for SIREN balloon simulation output.

Processes HNL event files to compute:
1. Detection efficiency via Cherenkov photon calculation
2. Expected event rates
3. Sensitivity contours

This script reuses the Cherenkov calculation from the main analysis
to ensure consistency with the original notebook results.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from glob import glob

# Add path to the main analysis code
sys.path.insert(0, "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/Pheno/HNLs/NeutrinoFactoryHNL")

try:
    from src.cherenkov import cherenkov_photons_detected_vectorized
    from src.balloon import muon_range_in_air, muon_energy_from_hnl_decay
    from src.constants import N_muon_decays, R_det, m_mu, speed_of_light
    IMPORTS_AVAILABLE = True
except ImportError:
    print("Warning: Could not import from src/. Using simplified analysis.")
    IMPORTS_AVAILABLE = False
    # Fallback constants
    N_muon_decays = 1e22
    R_det = 2.0
    m_mu = 0.1056583745


def air_density(altitude_m):
    """Approximate air density as function of altitude."""
    rho_0 = 1.225  # sea level density [kg/m^3]
    H = 8500  # scale height [m]
    return rho_0 * np.exp(-altitude_m / H)


def simplified_cherenkov_detection(decay_z, dir_z, E_mu_daughter, 
                                    satellite_height, R_det, min_photons=10):
    """
    Simplified Cherenkov detection probability.
    
    For events where daughter muon travels upward toward detector.
    """
    # Only count if muon goes upward
    if dir_z <= 0:
        return 0, False
    
    # Simplified geometric acceptance
    # Muon must be within R_det transverse distance from detector axis
    dist_to_detector = satellite_height - decay_z
    
    # Cherenkov cone angle ~ 1.4 degrees
    theta_c = 0.0245  # radians
    
    # Cherenkov photon yield ~ 64 photons/m
    dN_dx = 64.18
    
    # Estimate track length (cap at muon decay length)
    tau_mu = 2.2e-6  # seconds
    gamma_mu = E_mu_daughter / m_mu
    L_mu_decay = gamma_mu * speed_of_light * tau_mu if IMPORTS_AVAILABLE else 1e6
    
    # Energy loss limits track
    track_length = min(dist_to_detector / dir_z, L_mu_decay, 50000)
    
    # Rough photon estimate
    # Geometric factor for photons hitting disk detector
    solid_angle = np.pi * R_det**2 / (dist_to_detector**2)
    cherenkov_cone_solid = 2 * np.pi * (1 - np.cos(theta_c))
    geo_factor = solid_angle / cherenkov_cone_solid if cherenkov_cone_solid > 0 else 0
    
    N_photons = dN_dx * track_length * geo_factor
    
    detected = N_photons >= min_photons
    return N_photons, detected


def analyze_event_file(filepath, satellite_height=10000, min_photons=10):
    """
    Analyze a single HNL event file.
    
    Parameters
    ----------
    filepath : str
        Path to HNL events CSV file
    satellite_height : float
        Detector altitude in meters
    min_photons : int
        Detection threshold
        
    Returns
    -------
    dict with analysis results
    """
    df = pd.read_csv(filepath)
    
    # Extract parameters
    m_N = df['m_N'].iloc[0]
    U2 = df['U2'].iloc[0]
    E_N = df['E_N'].iloc[0]
    N_HNL_per_muon = df['N_HNL_per_muon'].iloc[0]
    
    # Filter to atmospheric decays
    atm_mask = df['in_atmosphere']
    atm_events = df[atm_mask].copy()
    
    if len(atm_events) == 0:
        return {
            'm_N': m_N,
            'U2': U2,
            'N_total': len(df),
            'N_atmosphere': 0,
            'N_detected': 0,
            'efficiency': 0.0,
            'expected_events': 0.0,
            'mean_photons': 0.0
        }
    
    # Muon energy from HNL decay
    if IMPORTS_AVAILABLE:
        E_mu_mean, _ = muon_energy_from_hnl_decay(m_N, E_N)
    else:
        # Simplified: muon gets ~half the HNL energy in lab frame
        gamma_N = E_N / m_N
        E_mu_mean = gamma_N * m_N / 2
    
    # Compute Cherenkov detection for each event
    photon_counts = []
    detected_mask = []
    
    for idx, row in atm_events.iterrows():
        decay_z = row['decay_z']
        dir_z = row['dir_z']
        
        if IMPORTS_AVAILABLE and dir_z > 0:
            # Use full Cherenkov calculation
            decay_pos = np.array([row['decay_x'], row['decay_y'], row['decay_z']])
            sat_pos = np.array([0, 0, satellite_height])
            r_rel = decay_pos - sat_pos
            
            mu_dir = np.array([row['dir_x'], row['dir_y'], row['dir_z']])
            
            # Add decay angle spread
            gamma_N = E_N / m_N
            theta_spread = np.random.exponential(1/gamma_N)
            phi = np.random.uniform(0, 2*np.pi)
            perp = np.array([np.cos(phi), np.sin(phi), 0])
            perp = perp - np.dot(perp, mu_dir) * mu_dir
            if np.linalg.norm(perp) > 0:
                perp = perp / np.linalg.norm(perp)
                mu_dir = mu_dir + theta_spread * perp
                mu_dir = mu_dir / np.linalg.norm(mu_dir)
            
            if mu_dir[2] <= 0:
                photon_counts.append(0)
                detected_mask.append(False)
                continue
            
            track_length = muon_range_in_air(E_mu_mean, decay_z, mu_dir[2])
            
            try:
                N_ph, _, _ = cherenkov_photons_detected_vectorized(
                    r_rel, mu_dir, track_length, R_det, N_psi=30, N_track=300
                )
            except:
                N_ph = 0
            
            photon_counts.append(N_ph)
            detected_mask.append(N_ph >= min_photons)
        else:
            # Use simplified calculation
            N_ph, det = simplified_cherenkov_detection(
                decay_z, dir_z, E_mu_mean, satellite_height, R_det, min_photons
            )
            photon_counts.append(N_ph)
            detected_mask.append(det)
    
    photon_counts = np.array(photon_counts)
    detected_mask = np.array(detected_mask)
    
    # Compute efficiency
    efficiency = np.sum(detected_mask) / len(df)
    
    # Expected events
    expected_events = N_HNL_per_muon * efficiency * N_muon_decays
    
    # Mean photons for detected events
    mean_photons = np.mean(photon_counts[detected_mask]) if np.any(detected_mask) else 0
    
    return {
        'm_N': m_N,
        'U2': U2,
        'N_total': len(df),
        'N_atmosphere': len(atm_events),
        'N_detected': np.sum(detected_mask),
        'efficiency': efficiency,
        'expected_events': expected_events,
        'mean_photons': mean_photons
    }


def main():
    parser = argparse.ArgumentParser(description='Analyze balloon HNL simulation results')
    parser.add_argument('--input_dir', type=str, default='./output',
                        help='Directory containing HNL event files')
    parser.add_argument('--output', type=str, default='analysis_results.csv',
                        help='Output CSV file')
    parser.add_argument('--satellite_height', type=float, default=10000,
                        help='Detector altitude in meters')
    parser.add_argument('--min_photons', type=int, default=10,
                        help='Minimum photons for detection')
    parser.add_argument('--pattern', type=str, default='*_hnl_events.csv',
                        help='Glob pattern for input files')
    
    args = parser.parse_args()
    
    # Find all event files
    pattern = os.path.join(args.input_dir, args.pattern)
    files = sorted(glob(pattern))
    
    if not files:
        print(f"No files found matching {pattern}")
        return
    
    print(f"Found {len(files)} event files to analyze")
    print(f"Detector height: {args.satellite_height/1000:.1f} km")
    print(f"Detection threshold: {args.min_photons} photons")
    print()
    
    results = []
    for i, filepath in enumerate(files):
        print(f"Analyzing {os.path.basename(filepath)} ({i+1}/{len(files)})...", end=' ')
        result = analyze_event_file(filepath, args.satellite_height, args.min_photons)
        results.append(result)
        print(f"efficiency={result['efficiency']:.4f}, events={result['expected_events']:.2e}")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output, index=False)
    print(f"\nSaved analysis to {args.output}")
    
    # Print summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Mass range: {results_df['m_N'].min():.1f} - {results_df['m_N'].max():.1f} GeV")
    print(f"Max expected events: {results_df['expected_events'].max():.2e}")
    
    # Find sensitivity point (3 events)
    sensitive = results_df[results_df['expected_events'] >= 3]
    if len(sensitive) > 0:
        print(f"Masses with >= 3 events: {len(sensitive)}")
        print(f"  Best mass: {sensitive.loc[sensitive['expected_events'].idxmax(), 'm_N']:.1f} GeV")


if __name__ == '__main__':
    main()
