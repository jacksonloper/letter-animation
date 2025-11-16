#!/usr/bin/env python3
"""
Script to compare Hamiltonian dynamics and instanton path approaches.

This script runs both methods on the same initial/final states and compares:
1. The paths taken through image space
2. The computational cost
3. The physical interpretation (energy vs action)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

# Import from other scripts
import sys
sys.path.append(str(Path(__file__).parent))

from hamiltonian_dynamics import (
    load_model,
    score_function,
    hamiltonian_dynamics as run_hamiltonian
)
from instanton_path import (
    optimize_instanton_path,
    freidlin_wentzell_action
)


def extract_endpoints_from_hamiltonian(trajectory):
    """Extract start and end points from Hamiltonian trajectory."""
    x_start = trajectory['x'][0]
    x_end = trajectory['x'][-1]
    return x_start, x_end


def compare_paths_detailed(ham_trajectory, inst_path, pipeline, device, output_dir="."):
    """
    Detailed comparison of the two paths.
    
    Args:
        ham_trajectory: Dictionary with Hamiltonian trajectory data
        inst_path: Array with instanton path
        pipeline: DDPM pipeline
        device: torch device
        output_dir: Directory to save outputs
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Extract Hamiltonian path positions
    ham_path = np.array(ham_trajectory['x'])
    
    # Compute Freidlin-Wentzell action for both paths
    print("\nComputing Freidlin-Wentzell actions...")
    ham_action = freidlin_wentzell_action(ham_path, pipeline, device)
    inst_action = freidlin_wentzell_action(inst_path, pipeline, device)
    
    print(f"  Hamiltonian path action: {ham_action:.6f}")
    print(f"  Instanton path action: {inst_action:.6f}")
    print(f"  Action ratio (Hamiltonian/Instanton): {ham_action/inst_action:.4f}")
    
    # Visualize paths side by side
    n_display = 10
    ham_indices = np.linspace(0, len(ham_path)-1, n_display, dtype=int)
    inst_indices = np.linspace(0, len(inst_path)-1, n_display, dtype=int)
    
    fig, axes = plt.subplots(3, n_display, figsize=(2*n_display, 6))
    
    for i, (h_idx, i_idx) in enumerate(zip(ham_indices, inst_indices)):
        # Hamiltonian path
        h_img = ham_path[h_idx]
        if len(h_img.shape) == 4:
            h_img = h_img[0, 0]
        elif len(h_img.shape) == 3:
            h_img = h_img[0]
            
        axes[0, i].imshow(h_img, cmap='gray', vmin=0, vmax=1)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('Hamiltonian', fontsize=12, rotation=0, ha='right')
        
        # Instanton path
        i_img = inst_path[i_idx]
        if len(i_img.shape) == 4:
            i_img = i_img[0, 0]
        elif len(i_img.shape) == 3:
            i_img = i_img[0]
            
        axes[1, i].imshow(i_img, cmap='gray', vmin=0, vmax=1)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel('Instanton', fontsize=12, rotation=0, ha='right')
        
        # Difference
        # Interpolate to same size if needed
        if h_img.shape != i_img.shape:
            from scipy.ndimage import zoom
            if h_img.size < i_img.size:
                factor = np.array(i_img.shape) / np.array(h_img.shape)
                h_img = zoom(h_img, factor)
            else:
                factor = np.array(h_img.shape) / np.array(i_img.shape)
                i_img = zoom(i_img, factor)
        
        diff = np.abs(h_img - i_img)
        axes[2, i].imshow(diff, cmap='hot', vmin=0, vmax=0.5)
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_ylabel('Difference', fontsize=12, rotation=0, ha='right')
    
    plt.suptitle(f'Path Comparison\nHamiltonian Action: {ham_action:.4f} | Instanton Action: {inst_action:.4f}', 
                 fontsize=14)
    plt.tight_layout()
    
    comparison_path = output_dir / "detailed_comparison.png"
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    print(f"\nDetailed comparison saved to: {comparison_path}")
    
    # Plot metrics over time
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    
    # Hamiltonian energy
    if 'energy' in ham_trajectory:
        axes[0].plot(ham_trajectory['t'], ham_trajectory['energy'], 'b-', linewidth=2)
        axes[0].set_xlabel('Time', fontsize=12)
        axes[0].set_ylabel('Energy', fontsize=12)
        axes[0].set_title('Hamiltonian Energy Conservation', fontsize=12)
        axes[0].grid(True, alpha=0.3)
    
    # Path length comparison
    ham_length = compute_path_length(ham_path)
    inst_length = compute_path_length(inst_path)
    
    axes[1].bar(['Hamiltonian', 'Instanton'], 
                [ham_length, inst_length],
                color=['blue', 'red'],
                alpha=0.7)
    axes[1].set_ylabel('Path Length', fontsize=12)
    axes[1].set_title('Total Path Length Comparison', fontsize=12)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    metrics_path = output_dir / "metrics_comparison.png"
    plt.savefig(metrics_path, dpi=150, bbox_inches='tight')
    print(f"Metrics comparison saved to: {metrics_path}")
    
    # Save summary statistics
    summary_path = output_dir / "comparison_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("COMPARISON SUMMARY: Hamiltonian vs Instanton Paths\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("FREIDLIN-WENTZELL ACTION (Lower is better):\n")
        f.write(f"  Hamiltonian path: {ham_action:.6f}\n")
        f.write(f"  Instanton path:   {inst_action:.6f}\n")
        f.write(f"  Ratio (H/I):      {ham_action/inst_action:.4f}\n\n")
        
        f.write("PATH LENGTH:\n")
        f.write(f"  Hamiltonian path: {ham_length:.6f}\n")
        f.write(f"  Instanton path:   {inst_length:.6f}\n")
        f.write(f"  Ratio (H/I):      {ham_length/inst_length:.4f}\n\n")
        
        f.write("INTERPRETATION:\n")
        f.write("- Hamiltonian dynamics: Conserves energy, explores based on\n")
        f.write("  momentum and gradient of log probability.\n")
        f.write("- Instanton path: Minimizes action functional, finds most\n")
        f.write("  probable transition path in the presence of noise.\n\n")
        
        if inst_action < ham_action:
            improvement = (ham_action - inst_action) / ham_action * 100
            f.write(f"The instanton path has {improvement:.1f}% lower action,\n")
            f.write("making it the more probable transition under diffusion.\n")
        else:
            f.write("The Hamiltonian path achieved lower action in this case.\n")
    
    print(f"Summary statistics saved to: {summary_path}")


def compute_path_length(path):
    """Compute the total Euclidean length of a path."""
    total_length = 0.0
    for i in range(len(path) - 1):
        diff = path[i+1] - path[i]
        total_length += np.sqrt(np.sum(diff**2))
    return total_length


def main():
    """Main comparison function."""
    print("=" * 60)
    print("COMPARING HAMILTONIAN DYNAMICS AND INSTANTON PATHS")
    print("=" * 60)
    
    # Load model
    print("\nLoading DDPM MNIST model...")
    pipeline = load_model()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.unet.to(device)
    pipeline.unet.eval()
    print(f"Using device: {device}")
    
    # Generate initial image and random velocity
    print("\n" + "-" * 60)
    print("STEP 1: Running Hamiltonian Dynamics")
    print("-" * 60)
    
    with torch.no_grad():
        img0 = pipeline(num_inference_steps=25).images[0]
    
    x0 = np.array(img0).astype(np.float32) / 255.0
    if len(x0.shape) == 2:
        x0 = x0[np.newaxis, np.newaxis, :, :]
    elif len(x0.shape) == 3:
        x0 = x0.transpose(2, 0, 1)[np.newaxis, :, :, :]
    
    v0 = np.random.randn(*x0.shape) * 0.05
    
    print(f"Initial state shape: {x0.shape}")
    
    start_time = time.time()
    ham_trajectory = run_hamiltonian(
        pipeline,
        x0,
        v0,
        T=3.0,
        n_steps=20
    )
    ham_time = time.time() - start_time
    print(f"Hamiltonian dynamics completed in {ham_time:.2f} seconds")
    
    # Extract endpoints for instanton path
    x_start, x_end = extract_endpoints_from_hamiltonian(ham_trajectory)
    
    # Run instanton optimization
    print("\n" + "-" * 60)
    print("STEP 2: Optimizing Instanton Path")
    print("-" * 60)
    
    start_time = time.time()
    inst_path, action_history = optimize_instanton_path(
        pipeline,
        x_start,
        x_end,
        n_steps=20,
        max_iter=30
    )
    inst_time = time.time() - start_time
    print(f"Instanton optimization completed in {inst_time:.2f} seconds")
    
    # Compare the two approaches
    print("\n" + "-" * 60)
    print("STEP 3: Detailed Comparison")
    print("-" * 60)
    
    compare_paths_detailed(ham_trajectory, inst_path, pipeline, device)
    
    print("\n" + "=" * 60)
    print("COMPARISON COMPLETE")
    print("=" * 60)
    print(f"\nComputational time:")
    print(f"  Hamiltonian: {ham_time:.2f}s")
    print(f"  Instanton:   {inst_time:.2f}s")
    print(f"\nSee generated plots and summary for detailed analysis.")


if __name__ == "__main__":
    main()
