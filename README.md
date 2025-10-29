# cluster_properties

A Python package for analyzing galaxy cluster properties from cosmological simulations, with a focus on f(R) modified gravity models.

## Overview

This repository contains tools for calculating and analyzing various properties of galaxy clusters from hydrodynamical cosmological simulations. The code is designed to work with Arepo simulation data and supports both General Relativity (GR) and f(R) modified gravity models, particularly the Hu-Sawicki f(R) model.

## Features

- **Cluster Property Calculation**: Compute fundamental cluster properties including:
  - Mass measurements (M200, M500, gas mass, stellar mass)
  - Temperature profiles (volume-weighted and mass-weighted)
  - Density profiles
  - Gas fraction profiles
  - Electron pressure profiles

- **Sunyaev-Zel'dovich (SZ) Analysis**:
  - Generate mock SZ maps
  - Calculate integrated Compton-y parameter (Y_SZ)
  - SZ signal with and without core regions

- **X-ray Luminosity**:
  - Calculate X-ray luminosity in various energy bands
  - Support for core-excised measurements

- **Scaling Relations**:
  - Generate and analyze proxy scaling relations (M-T, M-Y_SZ, M-L_X)
  - Compare different gravity models
  - Analytical fitting and jackknife error analysis

- **Modified Gravity Support**:
  - Built-in support for f(R) gravity models
  - Effective mass calculations for screened regions
  - Background scalaron field computations

## Installation

### Prerequisites

This package requires Python 3.x and the following dependencies:

```bash
# Core scientific libraries
numpy
scipy
matplotlib
h5py
astropy

# Additional requirements
pickle (standard library)
```

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Andrew-Pick/cluster_properties.git
cd cluster_properties
```

2. Install required packages:
```bash
pip install numpy scipy matplotlib h5py astropy
```

3. The code expects access to simulation data directories. Update paths in the code to match your data location.

## Usage

### Basic Cluster Properties Analysis

```python
from cluster_properties import ClusterProperties

# Initialize cluster properties calculator
cp = ClusterProperties(
    simulation="L302_N1136",    # Simulation name
    model="GR",                  # Model: "GR" or f(R) models like "F45", "F5", "F6"
    realisation="1",             # Realization number
    snapshot=12,                 # Snapshot number
    mass_cut=1e13,              # Minimum halo mass in M_sun
    delta=500,                   # Overdensity criterion (200 or 500)
    file_ending="all",          # Output file label
    rescaling="true",           # Use true or effective masses for f(R)
    core_frac=0.15              # Core fraction for core-excised measurements
)

# Calculate all cluster properties
cp.cluster_properties()
```

### Generating SZ Maps

```python
from cluster_properties import ClusterProperties

# Initialize and compute properties
cp = ClusterProperties(
    simulation="L302_N1136",
    model="GR",
    realisation="1",
    snapshot=12
)

# Generate SZ map for a specific cluster
cp.cluster_SZ(group_id=0)  # group_id=-1 for all clusters
```

### Scaling Relations

```python
from cluster_properties import ClusterProperties

cp = ClusterProperties(
    simulation="L302_N1136",
    model="GR",
    realisation="1",
    snapshot=12
)

# Compute cluster properties first
cp.cluster_properties()

# Generate scaling relations
cp.proxy_scaling_relation(
    proxy_type=["T", "Ysz", "Lx"],  # Temperature, SZ, X-ray
    temp_weight="mass",              # "mass" or "volume" weighting
    no_core=True,                    # Use core-excised measurements
    use_analytical=True              # Fit analytical relations
)
```

### Comparing Multiple Models

```python
from scaling_relations import ScalingRelations

# Compare different gravity models
sr = ScalingRelations(
    simulation="L302_N1136",
    models=["GR", "F45", "F5", "F6"],
    realisations=["1", "1", "1", "1"],
    snapshot=12,
    file_ending="all",
    labels=["GR", "F4.5", "F5", "F6"],
    colors=["black", "blue", "green", "red"],
    defaults=["GR", 0, 0, 0],  # Which model to use as reference
    plot_name="comparison"
)

# Generate comparison plots
sr.gas_mass_fraction(ax)
sr.stellar_mass_fraction(ax)
sr.black_hole_mass(ax)
```

## Module Descriptions

### Core Modules

- **cluster_properties.py**: Main module containing the `ClusterProperties` class for computing cluster properties, temperature profiles, gas fractions, and scaling relations.

- **scaling_relations.py**: Tools for analyzing and plotting scaling relations between cluster observables (mass, temperature, SZ signal, X-ray luminosity).

- **mock_SZ_map.py**: Generate mock Sunyaev-Zel'dovich effect maps from simulation data, including realistic observational effects.

- **SZ_map_volume.py**: Create volume-based SZ maps from cosmological simulation volumes.

- **electron_pressure.py**: Calculate electron pressure profiles and related quantities for cluster analysis.

- **temp_relation.py**: Specialized tools for temperature-based scaling relations and analysis.

### Arepo Library

The `Arepo/` directory contains utilities for reading and processing Arepo simulation data:

- **read_hdf5.py**: HDF5 file reader for Arepo snapshots
- **readsnap.py**: Snapshot reading utilities
- **readsubf.py**: Subhalo finder output reader
- **group_particles.py**: Group particle data by halos
- **cluster_properties.py**: Specialized cluster property calculations

### Execution Scripts

- **run_ClusterProperties.py**: Execute cluster property calculations
- **run_SZ_map.py**: Generate SZ maps
- **run_Scaling_Relation.py**: Compute scaling relations
- **run_SimulationTesting.py**: Test and validate simulation data

## Supported Simulations

The code supports various simulation suites including:

- L302_N1136 (high-resolution cluster suite)
- L62_N512, L100_N256, L54_N256, L68_N256, L86_N256, L136_N512
- TNG300 (IllustrisTNG)
- Custom f(R) gravity simulations

## Physical Constants and Conventions

The code uses the following units and conventions:

- **Masses**: Solar masses (M☉)
- **Distances**: Kiloparsecs (kpc) and Megaparsecs (Mpc)
- **Temperatures**: Kilo-electron volts (keV)
- **Hydrogen mass fraction**: X_H = 0.76
- **Adiabatic index**: γ = 5/3

## Modified Gravity Models

The code implements the Hu-Sawicki f(R) model with parameter f_R0:

- **F45**: |f_R0| = 10^-4.5
- **F5**: |f_R0| = 10^-5
- **F6**: |f_R0| = 10^-6
- **GR**: General Relativity (f_R0 = 0)

## Output

The code generates:

- Pickle files containing computed cluster properties
- PDF plots of scaling relations and profiles
- SZ maps in various formats
- Temperature and density profiles
- Comparison plots between different models

## Project Structure

```
cluster_properties/
├── cluster_properties.py      # Main analysis module
├── scaling_relations.py       # Scaling relation tools
├── mock_SZ_map.py            # SZ map generation
├── SZ_map_volume.py          # Volume SZ maps
├── electron_pressure.py      # Pressure profile calculations
├── temp_relation.py          # Temperature relations
├── tiling.py                 # Spatial tiling utilities
├── test.py                   # Testing utilities
├── run_*.py                  # Execution scripts
├── Arepo/                    # Arepo data reading library
│   ├── read_hdf5.py
│   ├── readsnap.py
│   ├── readsubf.py
│   ├── group_particles.py
│   └── cluster_properties.py
└── plots/                    # Output plots directory
```

## Notes

- The code is designed to run on high-performance computing systems with access to large simulation datasets
- File paths are hardcoded for specific computing systems (COSMA) and may need adjustment for local use
- Many functions assume specific simulation box sizes and cosmological parameters

## Author

Andrew Pickett

## Acknowledgments

This project builds upon Arepo simulation data and uses standard cosmological analysis techniques for cluster property measurements.

## License

Please contact the author for licensing information.
