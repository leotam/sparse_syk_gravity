import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# Style constants
COLOR_PRIMARY = '#D32F2F'  # Red
COLOR_SECONDARY = '#1976D2' # Blue
COLOR_TERTIARY = '#388E3C'  # Green
CMAP_HEATMAP = plt.cm.magma
CMAP_DIVERGING = plt.cm.RdBu_r

def plot_benchmark_results(results_table, sff_history, sff_times, save_path=None):
    """Plots Figure 1: OTOC Robustness and SFF."""
    sparsities = [r['sp'] for r in results_table]
    peaks = [r['peak'] for r in results_table]
    
    plt.figure(figsize=(14, 6))

    # Plot 1: Wormhole Robustness
    plt.subplot(1, 2, 1)
    plt.plot(sparsities, peaks, 'o-', linewidth=2, color=COLOR_PRIMARY)
    plt.gca().invert_xaxis()
    plt.axhline(y=0.005, color='k', linestyle='--', alpha=0.5, label='Classical Limit')
    plt.title("Wormhole Robustness (OTOC)")
    plt.xlabel("Sparsity")
    plt.ylabel("Peak Height")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: SFF
    plt.subplot(1, 2, 2)
    # Select a target index (e.g., 10% sparsity or similar)
    target_idx = min(3, len(sparsities)-1)
    raw_sff = sff_history[target_idx]
    smooth_sff = gaussian_filter1d(raw_sff, sigma=2.0)
    
    plt.loglog(sff_times, raw_sff, color='gray', alpha=0.3, label='Raw')
    plt.loglog(sff_times, smooth_sff, color=COLOR_SECONDARY, linewidth=2, label=f'Smoothed')
    plt.title("Spectral Form Factor")
    plt.xlabel("Time")
    plt.ylabel("K(t)")
    plt.legend()
    plt.grid(True, which="both", alpha=0.2)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved figure to {save_path}")
    plt.show()

def plot_phase_diagram(otoc_map, chaos_map, x_labels, y_labels, save_path=None):
    """Plots Figure 2: The Phase Diagram Heatmaps."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 1. OTOC Heatmap
    im1 = axes[0].imshow(otoc_map, cmap=CMAP_HEATMAP, aspect='auto')
    axes[0].set_xticks(np.arange(len(x_labels)))
    axes[0].set_yticks(np.arange(len(y_labels)))
    axes[0].set_xticklabels(x_labels, rotation=45)
    axes[0].set_yticklabels(y_labels)
    axes[0].set_title("Wormhole Traversability (OTOC Peak)")
    plt.colorbar(im1, ax=axes[0])

    # 2. Chaos Heatmap
    im2 = axes[1].imshow(chaos_map, cmap=CMAP_DIVERGING, aspect='auto', vmin=1.0, vmax=3.0)
    axes[1].set_xticks(np.arange(len(x_labels)))
    axes[1].set_yticks(np.arange(len(y_labels)))
    axes[1].set_xticklabels(x_labels, rotation=45)
    axes[1].set_yticklabels(y_labels)
    axes[1].set_title("Quantum Chaos (SFF Ramp Ratio)")
    plt.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def plot_shapiro_delay(strengths, delays, errors, save_path=None):
    """Plots Figure 4: Shapiro Delay."""
    plt.figure(figsize=(6, 5))
    plt.errorbar(strengths, delays, yerr=errors, 
                 fmt='o-', color=COLOR_PRIMARY, linewidth=2.5, 
                 capsize=5, markersize=10, label='Simulation')
    plt.title(f"Shapiro Delay vs. Strength")
    plt.xlabel("Perturbation Strength (Angle)")
    plt.ylabel("Time Delay $\Delta t$")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()