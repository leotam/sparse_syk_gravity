import numpy as np

def compute_sff(evals, time_points, beta):
    """
    Computes the Spectral Form Factor (SFF).
    SFF(t) = |Z(beta + it)|^2 / |Z(beta)|^2
    """
    # Partition function at beta (normalization)
    Z_beta = np.sum(np.exp(-beta * evals))
    
    # Vectorized calculation over time points
    # Z(beta + it) = sum exp(-(beta + it)E_n)
    #              = sum exp(-beta*E_n/2) * exp(-i*E_n*t) * exp(-beta*E_n/2) ???
    # Usually SFF is defined with beta/2 or just beta. Z_complex_t uses (beta/2) for real part.
    
    Z_complex_t = np.sum(
        np.exp(-(beta/2)*evals[:, None] - 1j*evals[:, None]*time_points[None, :]), 
        axis=0
    )
    
    sff_vals = (np.abs(Z_complex_t)**2) / (Z_beta**2)
    return sff_vals

def compute_gap_ratio(evals):
    """
    Computes the adjacent level spacing ratio <r> for a spectrum.
    Used to distinguish Chaos (GUE ~0.60) from Integrability (Poisson ~0.39).
    """
    # 1. Sort and Uniquify (remove degeneracy)
    evals_sorted = np.sort(evals.real)
    evals_unique = np.unique(np.round(evals_sorted, decimals=8))
    
    # 2. Spacings
    spacings = np.diff(evals_unique)
    # Filter tiny spacings (numerical noise)
    spacings = spacings[spacings > 1e-10]
    
    if len(spacings) < 20:
        return np.nan, np.nan
        
    # 3. Focus on middle spectrum (unfold interaction effects at edges)
    start_idx = len(spacings) // 4
    end_idx = 3 * len(spacings) // 4
    spacings_middle = spacings[start_idx:end_idx]
    
    # 4. Ratio calculation
    # r_n = min(s_n, s_{n+1}) / max(s_n, s_{n+1})
    s_n = spacings_middle[:-1]
    s_np1 = spacings_middle[1:]
    r_vals = np.minimum(s_n, s_np1) / np.maximum(s_n, s_np1)
    
    return np.mean(r_vals), np.std(r_vals)

def compute_otoc_peak(otoc_curve):
    """Simple helper to extract peak and check validity."""
    if len(otoc_curve) == 0:
        return 0.0
    return np.max(otoc_curve)