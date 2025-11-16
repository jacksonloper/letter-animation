#!/usr/bin/env python3
"""
Script to navigate image space using Hamiltonian dynamics with the DDPM model.

This script uses the small t limit to approximate ∇log p(x), picks a random direction
in image space, and surfs via Hamiltonian mechanics (single ODE solve without resampling velocity).
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from diffusers import DDPMPipeline


def load_model(cache_dir="./ddpm-mnist"):
    """Load the DDPM MNIST model from cache."""
    pipeline = DDPMPipeline.from_pretrained(
        "1aurent/ddpm-mnist",
        cache_dir=cache_dir
    )
    return pipeline


def score_function(pipeline, x, t_small=0.001):
    """
    Approximate ∇log p(x) using the small t limit of the diffusion model.
    
    For DDPM, the score function is approximately:
    ∇log p(x) ≈ -ε_θ(x, t) / σ(t)
    
    At small t, this gives us the gradient of the log density.
    
    Args:
        pipeline: DDPM pipeline
        x: Current image (torch tensor)
        t_small: Small timestep for approximation
        
    Returns:
        score: ∇log p(x) as numpy array
    """
    device = x.device
    
    # Create timestep tensor
    t = torch.tensor([t_small], device=device)
    
    # Get noise prediction from model
    with torch.no_grad():
        noise_pred = pipeline.unet(x, t).sample
    
    # For small t, the score is approximately -noise_pred / sigma(t)
    # For DDPM, at small t, sigma(t) ≈ sqrt(2*t) for typical noise schedules
    # But we can use the scheduler's scaling
    alpha_t = pipeline.scheduler.alphas_cumprod[int(t_small * len(pipeline.scheduler.alphas_cumprod))]
    sigma_t = np.sqrt(1 - alpha_t)
    
    score = -noise_pred / (sigma_t + 1e-8)
    
    return score.cpu().numpy()


def hamiltonian_dynamics(pipeline, x0, v0, T=10.0, n_steps=100):
    """
    Integrate Hamiltonian dynamics in image space.
    
    The Hamiltonian system is:
        dx/dt = v
        dv/dt = ∇log p(x)
    
    Args:
        pipeline: DDPM pipeline
        x0: Initial position (image)
        v0: Initial velocity (random direction)
        T: Total integration time
        n_steps: Number of evaluation points
        
    Returns:
        trajectory: Dictionary with 't', 'x', 'v', 'energy'
    """
    device = next(pipeline.unet.parameters()).device
    
    # Flatten initial conditions
    x0_flat = x0.flatten()
    v0_flat = v0.flatten()
    shape = x0.shape
    
    # Store trajectory
    trajectory = {
        't': [],
        'x': [],
        'v': [],
        'energy': []
    }
    
    def compute_potential(x_flat):
        """Negative log probability (potential energy)."""
        x_img = torch.tensor(x_flat.reshape(shape), dtype=torch.float32).to(device)
        x_img = x_img.unsqueeze(0)  # Add batch dimension
        score = score_function(pipeline, x_img)
        score_flat = score.flatten()
        # Potential U(x) = -log p(x), so ∇U = -∇log p = -score
        return -score_flat
    
    def hamiltonian_rhs(t, state):
        """Right-hand side of Hamiltonian equations."""
        n = len(state) // 2
        x = state[:n]
        v = state[n:]
        
        # Compute force (gradient of log probability)
        force = -compute_potential(x)
        
        # Hamiltonian equations
        dx_dt = v
        dv_dt = force
        
        return np.concatenate([dx_dt, dv_dt])
    
    # Initial state [x, v]
    state0 = np.concatenate([x0_flat, v0_flat])
    
    # Solve ODE
    print(f"Integrating Hamiltonian dynamics for T={T} with {n_steps} steps...")
    t_eval = np.linspace(0, T, n_steps)
    
    sol = solve_ivp(
        hamiltonian_rhs,
        (0, T),
        state0,
        t_eval=t_eval,
        method='RK45',
        rtol=1e-6,
        atol=1e-8
    )
    
    # Extract trajectory
    n = len(x0_flat)
    for i, t in enumerate(sol.t):
        x = sol.y[:n, i]
        v = sol.y[n:, i]
        
        # Compute energy (kinetic + potential)
        kinetic = 0.5 * np.sum(v**2)
        # Potential is harder to compute exactly, so we approximate
        potential_grad = compute_potential(x)
        potential = np.sum(x * potential_grad)  # Approximate
        
        trajectory['t'].append(t)
        trajectory['x'].append(x.reshape(shape))
        trajectory['v'].append(v.reshape(shape))
        trajectory['energy'].append(kinetic + potential)
    
    return trajectory


def visualize_trajectory(trajectory, output_path="hamiltonian_trajectory.png"):
    """Visualize the Hamiltonian trajectory."""
    n_frames = len(trajectory['t'])
    n_display = min(10, n_frames)
    indices = np.linspace(0, n_frames-1, n_display, dtype=int)
    
    fig, axes = plt.subplots(2, n_display, figsize=(2*n_display, 4))
    
    for i, idx in enumerate(indices):
        x = trajectory['x'][idx]
        v = trajectory['v'][idx]
        t = trajectory['t'][idx]
        
        # Display image
        if len(x.shape) == 4:  # (1, C, H, W)
            img = x[0, 0]  # Take first channel
        elif len(x.shape) == 3:  # (C, H, W)
            img = x[0]
        else:
            img = x
            
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].axis('off')
        axes[0, i].set_title(f't={t:.2f}')
        
        # Display velocity magnitude
        if len(v.shape) == 4:
            v_mag = np.abs(v[0, 0])
        elif len(v.shape) == 3:
            v_mag = np.abs(v[0])
        else:
            v_mag = np.abs(v)
            
        axes[1, i].imshow(v_mag, cmap='hot')
        axes[1, i].axis('off')
        axes[1, i].set_title('velocity')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Trajectory visualization saved to: {output_path}")
    
    # Plot energy over time
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(trajectory['t'], trajectory['energy'])
    ax.set_xlabel('Time')
    ax.set_ylabel('Energy')
    ax.set_title('Hamiltonian Energy over Time')
    ax.grid(True)
    
    energy_path = output_path.replace('.png', '_energy.png')
    plt.savefig(energy_path, dpi=150, bbox_inches='tight')
    print(f"Energy plot saved to: {energy_path}")


def main():
    """Main function to run Hamiltonian dynamics."""
    print("Loading DDPM MNIST model...")
    pipeline = load_model()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.unet.to(device)
    pipeline.unet.eval()
    
    print(f"Using device: {device}")
    
    # Generate initial image
    print("Generating initial image...")
    with torch.no_grad():
        x0 = pipeline(num_inference_steps=25).images[0]
    
    # Convert to tensor
    x0_array = np.array(x0).astype(np.float32) / 255.0
    # Reshape to (1, C, H, W) format
    if len(x0_array.shape) == 2:
        x0_array = x0_array[np.newaxis, np.newaxis, :, :]
    elif len(x0_array.shape) == 3:
        x0_array = x0_array.transpose(2, 0, 1)[np.newaxis, :, :, :]
    
    # Generate random initial velocity
    v0 = np.random.randn(*x0_array.shape) * 0.1
    
    print(f"Initial image shape: {x0_array.shape}")
    print(f"Initial velocity shape: {v0.shape}")
    
    # Run Hamiltonian dynamics
    trajectory = hamiltonian_dynamics(
        pipeline,
        x0_array,
        v0,
        T=5.0,
        n_steps=50
    )
    
    # Visualize results
    visualize_trajectory(trajectory)
    
    print("\nDone! Hamiltonian dynamics completed.")


if __name__ == "__main__":
    main()
