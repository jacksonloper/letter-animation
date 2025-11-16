#!/usr/bin/env python3
"""
Script to find instanton paths using the Freidlin-Wentzell functional.

This script optimizes a path between two fixed endpoints to minimize the
Freidlin-Wentzell action functional for isotropic homogeneous Langevin flow
with potential -log p(x).

The Freidlin-Wentzell action for a path x(t) from time 0 to T is:
    S[x] = (1/2) ∫_0^T |dx/dt - ∇log p(x)|^2 dt

This represents the "cost" of a transition in the presence of noise.
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.optimize import minimize
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
    
    Args:
        pipeline: DDPM pipeline
        x: Current image (torch tensor with batch dimension)
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
    
    # Compute score from noise prediction
    alpha_t = pipeline.scheduler.alphas_cumprod[int(t_small * len(pipeline.scheduler.alphas_cumprod))]
    sigma_t = np.sqrt(1 - alpha_t)
    
    score = -noise_pred / (sigma_t + 1e-8)
    
    return score.cpu().numpy()


def freidlin_wentzell_action(path, pipeline, device):
    """
    Compute the Freidlin-Wentzell action for a path.
    
    S[x] = (1/2) ∫_0^T |dx/dt - ∇log p(x)|^2 dt
    
    Args:
        path: Array of shape (n_steps, *image_shape) representing the path
        pipeline: DDPM pipeline
        device: torch device
        
    Returns:
        action: The Freidlin-Wentzell action value
    """
    n_steps = len(path)
    dt = 1.0 / (n_steps - 1)
    
    action = 0.0
    
    for i in range(n_steps - 1):
        x_curr = path[i]
        x_next = path[i + 1]
        
        # Compute velocity
        velocity = (x_next - x_curr) / dt
        
        # Compute drift (score function)
        x_tensor = torch.tensor(x_curr, dtype=torch.float32).to(device)
        x_tensor = x_tensor.unsqueeze(0)  # Add batch dimension
        score = score_function(pipeline, x_tensor)
        drift = score.squeeze(0)
        
        # Compute integrand: |dx/dt - ∇log p(x)|^2
        diff = velocity - drift
        integrand = np.sum(diff**2)
        
        action += 0.5 * integrand * dt
    
    return action


def optimize_instanton_path(pipeline, x_start, x_end, n_steps=20, max_iter=100):
    """
    Optimize a path to minimize the Freidlin-Wentzell action.
    
    The path endpoints are fixed, and we optimize the intermediate points.
    
    Args:
        pipeline: DDPM pipeline
        x_start: Starting image
        x_end: Ending image
        n_steps: Number of discretization points along the path
        max_iter: Maximum optimization iterations
        
    Returns:
        optimized_path: The optimized instanton path
        action_history: History of action values during optimization
    """
    device = next(pipeline.unet.parameters()).device
    
    # Initialize path with linear interpolation
    print("Initializing path with linear interpolation...")
    path = np.zeros((n_steps,) + x_start.shape)
    for i in range(n_steps):
        alpha = i / (n_steps - 1)
        path[i] = (1 - alpha) * x_start + alpha * x_end
    
    # Fix endpoints
    path[0] = x_start
    path[-1] = x_end
    
    # Flatten internal points for optimization
    internal_shape = path[1:-1].shape
    internal_flat = path[1:-1].flatten()
    
    action_history = []
    
    def objective(internal_flat):
        """Objective function: Freidlin-Wentzell action."""
        # Reconstruct full path
        internal = internal_flat.reshape(internal_shape)
        full_path = np.concatenate([
            path[0:1],
            internal,
            path[-1:]
        ], axis=0)
        
        # Compute action
        action = freidlin_wentzell_action(full_path, pipeline, device)
        action_history.append(action)
        
        if len(action_history) % 10 == 0:
            print(f"  Iteration {len(action_history)}: action = {action:.6f}")
        
        return action
    
    def gradient(internal_flat):
        """Numerical gradient of the action."""
        eps = 1e-5
        grad = np.zeros_like(internal_flat)
        
        for i in range(len(internal_flat)):
            internal_plus = internal_flat.copy()
            internal_plus[i] += eps
            
            internal_minus = internal_flat.copy()
            internal_minus[i] -= eps
            
            grad[i] = (objective(internal_plus) - objective(internal_minus)) / (2 * eps)
        
        return grad
    
    print(f"Optimizing instanton path with {n_steps} steps...")
    print(f"Initial action: {objective(internal_flat):.6f}")
    
    # Optimize using L-BFGS-B
    result = minimize(
        objective,
        internal_flat,
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'disp': True}
    )
    
    # Reconstruct optimized path
    optimized_internal = result.x.reshape(internal_shape)
    optimized_path = np.concatenate([
        path[0:1],
        optimized_internal,
        path[-1:]
    ], axis=0)
    
    print(f"Optimization completed. Final action: {result.fun:.6f}")
    
    return optimized_path, action_history


def compare_paths(hamiltonian_path, instanton_path, output_path="path_comparison.png"):
    """
    Compare Hamiltonian and instanton paths.
    
    Args:
        hamiltonian_path: Path from Hamiltonian dynamics
        instanton_path: Optimized instanton path
        output_path: Where to save the comparison figure
    """
    n_steps = min(len(hamiltonian_path), len(instanton_path))
    n_display = min(8, n_steps)
    indices = np.linspace(0, n_steps-1, n_display, dtype=int)
    
    fig, axes = plt.subplots(2, n_display, figsize=(2*n_display, 4))
    
    for i, idx in enumerate(indices):
        # Hamiltonian path
        h_img = hamiltonian_path[idx]
        if len(h_img.shape) == 4:
            h_img = h_img[0, 0]
        elif len(h_img.shape) == 3:
            h_img = h_img[0]
            
        axes[0, i].imshow(h_img, cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('Hamiltonian', fontsize=10)
        
        # Instanton path
        i_img = instanton_path[idx]
        if len(i_img.shape) == 4:
            i_img = i_img[0, 0]
        elif len(i_img.shape) == 3:
            i_img = i_img[0]
            
        axes[1, i].imshow(i_img, cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel('Instanton', fontsize=10)
    
    plt.suptitle('Comparison: Hamiltonian vs Instanton Paths')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Path comparison saved to: {output_path}")


def visualize_instanton_path(path, action_history, output_path="instanton_path.png"):
    """Visualize the optimized instanton path."""
    n_steps = len(path)
    n_display = min(10, n_steps)
    indices = np.linspace(0, n_steps-1, n_display, dtype=int)
    
    fig, axes = plt.subplots(1, n_display, figsize=(2*n_display, 2))
    
    for i, idx in enumerate(indices):
        img = path[idx]
        if len(img.shape) == 4:
            img = img[0, 0]
        elif len(img.shape) == 3:
            img = img[0]
            
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f't={idx/(n_steps-1):.2f}')
    
    plt.suptitle('Optimized Instanton Path')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Instanton path visualization saved to: {output_path}")
    
    # Plot action history
    if action_history:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(action_history)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Freidlin-Wentzell Action')
        ax.set_title('Action Optimization History')
        ax.grid(True)
        
        history_path = output_path.replace('.png', '_history.png')
        plt.savefig(history_path, dpi=150, bbox_inches='tight')
        print(f"Action history plot saved to: {history_path}")


def main():
    """Main function to find and visualize instanton paths."""
    print("Loading DDPM MNIST model...")
    pipeline = load_model()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.unet.to(device)
    pipeline.unet.eval()
    
    print(f"Using device: {device}")
    
    # Generate two endpoint images
    print("Generating endpoint images...")
    with torch.no_grad():
        img1 = pipeline(num_inference_steps=25).images[0]
        img2 = pipeline(num_inference_steps=25).images[0]
    
    # Convert to arrays
    x_start = np.array(img1).astype(np.float32) / 255.0
    x_end = np.array(img2).astype(np.float32) / 255.0
    
    # Reshape to (1, C, H, W) format
    if len(x_start.shape) == 2:
        x_start = x_start[np.newaxis, np.newaxis, :, :]
        x_end = x_end[np.newaxis, np.newaxis, :, :]
    elif len(x_start.shape) == 3:
        x_start = x_start.transpose(2, 0, 1)[np.newaxis, :, :, :]
        x_end = x_end.transpose(2, 0, 1)[np.newaxis, :, :, :]
    
    print(f"Start image shape: {x_start.shape}")
    print(f"End image shape: {x_end.shape}")
    
    # Optimize instanton path
    instanton_path, action_history = optimize_instanton_path(
        pipeline,
        x_start,
        x_end,
        n_steps=10,
        max_iter=50
    )
    
    # Visualize results
    visualize_instanton_path(instanton_path, action_history)
    
    print("\nDone! Instanton path optimization completed.")


if __name__ == "__main__":
    main()
