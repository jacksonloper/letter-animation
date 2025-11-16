# letter-animation

Exploration of diffusion models using Hamiltonian dynamics and instanton path optimization.

## Overview

This repository contains scripts for analyzing transition paths in diffusion models (DDPM MNIST) using two complementary approaches:

1. **Hamiltonian Dynamics**: Navigate image space using energy-conserving dynamics driven by the gradient of log probability
2. **Instanton Paths**: Optimize paths to minimize the Freidlin-Wentzell action functional, finding the most probable transitions

## Quick Start

Install dependencies:
```bash
pip install -e .
```

Download the DDPM MNIST model:
```bash
python scripts/download_model.py
```

Run the comparison:
```bash
python scripts/compare_methods.py
```

## Scripts

See the [scripts README](scripts/README.md) for detailed documentation of each script:
- `download_model.py`: Download and cache the DDPM MNIST model
- `hamiltonian_dynamics.py`: Explore via Hamiltonian mechanics
- `instanton_path.py`: Optimize instanton paths
- `compare_methods.py`: Compare both approaches

## Theory

These scripts implement two fundamental approaches to understanding transitions in stochastic systems:

- **Hamiltonian dynamics** treats the score function ∇log p(x) as a force field, generating momentum-driven trajectories
- **Instanton optimization** finds the most probable transition path by minimizing the action functional S[x] = (1/2) ∫ |dx/dt - ∇log p(x)|² dt

## Requirements

- Python 3.8+
- PyTorch
- Diffusers
- NumPy, SciPy, Matplotlib