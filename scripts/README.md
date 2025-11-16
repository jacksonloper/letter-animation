# Scripts Directory

This directory contains scripts for exploring diffusion models using Hamiltonian dynamics and instanton path optimization.

## Overview

These scripts use the DDPM MNIST model to analyze two different approaches for navigating image space:

1. **Hamiltonian Dynamics**: Uses momentum and the gradient of log probability to explore image space through energy-conserving dynamics
2. **Instanton Paths**: Optimizes paths to minimize the Freidlin-Wentzell action functional, finding the most probable transition paths under diffusion

## Scripts

### 1. `download_model.py`
Downloads and caches the DDPM MNIST model from HuggingFace.

```bash
python scripts/download_model.py
```

This will:
- Download the model to `./ddpm-mnist/` (gitignored)
- Generate a test image to verify the model works

### 2. `hamiltonian_dynamics.py`
Explores image space using Hamiltonian mechanics.

```bash
python scripts/hamiltonian_dynamics.py
```

This script:
- Uses the small-t limit to approximate ∇log p(x) (the score function)
- Picks a random initial velocity direction
- Integrates the Hamiltonian equations: dx/dt = v, dv/dt = ∇log p(x)
- Visualizes the trajectory and energy conservation

Output:
- `hamiltonian_trajectory.png`: Visualization of the path through image space
- `hamiltonian_trajectory_energy.png`: Energy over time

### 3. `instanton_path.py`
Finds optimal transition paths using the Freidlin-Wentzell functional.

```bash
python scripts/instanton_path.py
```

This script:
- Generates two random endpoint images
- Optimizes the path between them to minimize the action:
  ```
  S[x] = (1/2) ∫ |dx/dt - ∇log p(x)|² dt
  ```
- Fixes the endpoints and optimizes intermediate points
- Visualizes the optimized path and convergence

Output:
- `instanton_path.png`: The optimized path
- `instanton_path_history.png`: Action value during optimization

### 4. `compare_methods.py`
Compares both approaches on the same data.

```bash
python scripts/compare_methods.py
```

This script:
- Runs Hamiltonian dynamics to get a trajectory
- Optimizes an instanton path between the same endpoints
- Compares the paths in terms of:
  - Freidlin-Wentzell action (lower = more probable)
  - Path length
  - Visual appearance
- Generates detailed comparison plots

Output:
- `detailed_comparison.png`: Side-by-side visualization
- `metrics_comparison.png`: Quantitative metrics
- `comparison_summary.txt`: Statistical summary

## Installation

Install dependencies using pip:

```bash
pip install -e .
```

Or install specific dependencies:

```bash
pip install torch diffusers numpy scipy matplotlib pillow huggingface-hub
```

## Theory

### Hamiltonian Dynamics
The Hamiltonian system treats the image space as having "positions" (pixel values) and "velocities" (momentum):
- Kinetic energy: (1/2) |v|²
- Potential energy: -log p(x)
- Energy is conserved along trajectories

### Instanton Paths
The Freidlin-Wentzell functional measures the "cost" of a transition in the presence of noise:
- Paths that follow the gradient (∇log p) have low cost
- Paths that go against the gradient have high cost
- The optimal path (instanton) is the most probable transition

### Key Differences
- **Hamiltonian**: Energy-conserving, explores naturally, no optimization needed
- **Instanton**: Explicitly optimized, finds most probable path, respects noise structure

## Notes

- The model is cached locally in `ddpm-mnist/` (gitignored)
- Scripts use GPU if available, otherwise CPU
- The small-t approximation (t ≈ 0.001) is used to estimate the score function
- Optimization can be computationally intensive for high-resolution images
