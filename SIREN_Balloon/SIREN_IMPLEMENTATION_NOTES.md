# SIREN Implementation Notes for Muon→HNL DIS

This document describes what needs to be implemented in SIREN to support
muon-induced HNL production for the balloon experiment.

## Required: MuonHNLDISFromSpline Class

You need to implement a cross section class analogous to `HNLDISFromSpline`
but for muon primaries:

```
μ + N → HNL + X
```

### Interface (should match HNLDISFromSpline pattern)

```cpp
// In SIREN/projects/interactions/public/SIREN/interactions/MuonHNLDISFromSpline.h

class MuonHNLDISFromSpline : public CrossSection {
public:
    MuonHNLDISFromSpline(
        std::string differential_xs_file,  // dσ/dxdy spline
        std::string total_xs_file,          // σ(E) spline
        double hnl_mass,                    // HNL mass in GeV
        std::vector<double> mixing,         // [Ue4, Umu4, Utau4]
        double target_mass,                 // isoscalar nucleon mass
        double min_Q2,                      // minimum Q² cut
        std::vector<ParticleType> primary_types,   // {MuMinus} or {MuPlus}
        std::vector<ParticleType> target_types     // {Nucleon}
    );
    
    // Required virtual methods
    double TotalCrossSection(InteractionRecord const &) const override;
    double DifferentialCrossSection(InteractionRecord const &) const override;
    void SampleFinalState(InteractionRecord &, std::shared_ptr<Detector>) const override;
    double InteractionThreshold(InteractionRecord const &) const override;
    
    // ... other methods
};
```

### Key Physics

The differential cross section for μN → HNL X involves:

1. **DIS kinematics**: 
   - Bjorken x = Q²/(2p·q)
   - Inelasticity y = (p·q)/(p·k) where k is muon momentum
   
2. **HNL production**:
   - Threshold: s > (m_HNL + m_μ)²
   - Cross section scales as |U_μ|²
   
3. **Angular distribution**:
   - The HNL angle is determined by the DIS kinematics
   - NOT a simple forward-peaked approximation
   - `SampleFinalState` should properly sample (x, y) and compute HNL 4-momentum

### Cross Section Spline Files

Create spline files in the format SIREN expects:

```
resources/CrossSections/MuonHNLDISSplines/
├── M_0000000MeV/
│   └── dsdxdy-mu-N-nc.fits     # differential cross section template
├── M_0001000MeV/
│   └── sigma-mu-N-nc.fits      # total XS for m_HNL = 1 GeV
├── M_0005000MeV/
│   └── sigma-mu-N-nc.fits      # total XS for m_HNL = 5 GeV
...
```

## Python Binding

Add Python binding in `SIREN/python/`:

```python
# In SIREN bindings
siren.interactions.MuonHNLDISFromSpline(
    differential_file,
    total_file, 
    hnl_mass,
    mixing_angles,
    target_mass,
    min_Q2,
    primary_types,
    target_types
)
```

## Reference Implementation

Look at these existing files for patterns:

1. `SIREN/projects/interactions/private/HNLDISFromSpline.cxx`
   - Similar structure for neutrino→HNL
   
2. `SIREN/projects/interactions/private/DISFromSpline.cxx`
   - Standard DIS implementation
   
3. `SIREN/resources/CrossSections/HNLDISSplines/`
   - Existing HNL spline file structure

## Alternative: Modify Existing Classes

If implementing a new class is too heavy, you could potentially:

1. Modify `HNLDISFromSpline` to accept muon primaries
2. Create muon DIS splines with same format as neutrino ones
3. The kinematics are similar (both are charged lepton DIS)

The main difference is the primary particle type and potentially
some kinematic factors in the cross section calculation.

## Testing

Once implemented, test with:

```bash
cd /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/Pheno/HNLs/NeutrinoFactoryHNL/SIREN_Balloon

# Generate muon input first
python inputs/generate_muon_input.py --n_muons 10000 --output inputs/muon_beam_5TeV.csv

# Run simulation
python scripts/Balloon_SIREN_Simulation.py --m_hnl 20 --u2 1e-10 --n_events 1000
```
