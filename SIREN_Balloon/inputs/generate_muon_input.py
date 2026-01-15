#!/usr/bin/env python3
"""
Generate muon beam input file for SIREN simulation.

Creates a CSV file with fixed-energy muon beam parameters for the
balloon HNL detection experiment.

Output format matches SIREN's PrimaryExternalDistribution:
PDG,hPDG,x0,y0,z0,thx,thy,E,wgt,px,py,pz
"""

import numpy as np
import pandas as pd
import argparse

# Physical constants
m_mu = 0.1056583745  # muon mass in GeV

def generate_muon_beam(N_muons, E_mu, z_start, theta_spread=0.001, 
                       x_spread=0.1, y_spread=0.1, pdg=13):
    """
    Generate a collimated muon beam traveling upward.
    
    Parameters
    ----------
    N_muons : int
        Number of muons to generate
    E_mu : float
        Muon energy [GeV]
    z_start : float
        Starting z position [m] (negative = below surface)
    theta_spread : float
        Angular spread from vertical [rad]
    x_spread : float
        Transverse position spread [m]
    y_spread : float
        Transverse position spread [m]
    pdg : int
        PDG code (13 for mu-, -13 for mu+)
        
    Returns
    -------
    DataFrame with columns matching SIREN input format
    """
    p_mu = np.sqrt(E_mu**2 - m_mu**2)
    
    data = []
    for i in range(N_muons):
        # Sample angular deviation from vertical (beam goes upward, +z)
        theta = np.abs(np.random.normal(0, theta_spread))
        phi = np.random.uniform(0, 2*np.pi)
        
        # Direction components (beam goes upward, +z)
        px = p_mu * np.sin(theta) * np.cos(phi)
        py = p_mu * np.sin(theta) * np.sin(phi)
        pz = p_mu * np.cos(theta)  # positive = upward
        
        # Angular variables for SIREN format
        thx = np.arctan2(px, pz)
        thy = np.arctan2(py, pz)
        
        # Position with transverse spread
        x0 = np.random.normal(0, x_spread)
        y0 = np.random.normal(0, y_spread)
        z0 = z_start
        
        data.append({
            'PDG': pdg,
            'hPDG': 0,  # no associated hadron
            'x0': x0,
            'y0': y0,
            'z0': z0,
            'thx': thx,
            'thy': thy,
            'E': E_mu,
            'wgt': 1.0,
            'px': px,
            'py': py,
            'pz': pz
        })
    
    return pd.DataFrame(data)


def main():
    parser = argparse.ArgumentParser(description='Generate muon beam input for SIREN')
    parser.add_argument('--n_muons', type=int, default=100000,
                        help='Number of muons to generate')
    parser.add_argument('--energy', type=float, default=5000.0,
                        help='Muon energy in GeV')
    parser.add_argument('--z_start', type=float, default=-600.0,
                        help='Starting z position in meters (negative = below surface)')
    parser.add_argument('--theta_spread', type=float, default=0.001,
                        help='Angular spread from vertical in radians')
    parser.add_argument('--output', type=str, default='muon_beam_5TeV.csv',
                        help='Output filename')
    parser.add_argument('--pdg', type=int, default=13,
                        help='PDG code (13 for mu-, -13 for mu+)')
    
    args = parser.parse_args()
    
    print(f"Generating {args.n_muons} muons at {args.energy} GeV")
    print(f"Starting position: z = {args.z_start} m")
    print(f"Angular spread: {args.theta_spread} rad ({np.degrees(args.theta_spread):.3f} deg)")
    
    df = generate_muon_beam(
        args.n_muons, 
        args.energy, 
        args.z_start,
        args.theta_spread,
        pdg=args.pdg
    )
    
    df.to_csv(args.output, index=False)
    print(f"Saved to {args.output}")
    
    # Print summary
    print(f"\nSummary:")
    print(f"  Mean pz: {df['pz'].mean():.2f} GeV")
    print(f"  Mean theta from vertical: {np.mean(np.arctan2(np.sqrt(df['px']**2 + df['py']**2), df['pz'])):.6f} rad")


if __name__ == '__main__':
    main()
