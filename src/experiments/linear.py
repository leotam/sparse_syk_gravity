import os
import time
import pickle
import numpy as np
import scipy.linalg
from scipy import sparse
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from src.physics.dynamics import apply_coupling_matrix_op
from src.config import BETA, WORMHOLE_STR, TIME_STEPS, TIME_RANGE
from src.physics.algebra import get_majorana
from src.physics.hamiltonians import generate_syk_couplings

def run_linear_universe(seed, sparsity, n_total_majoranas):
    # Setup Dimensions
    n_half = n_total_majoranas // 2 
    dim_half = 2**n_half
    
    # 1. LOCAL HAMILTONIAN (Linear Connectivity = 4)
    kept_terms, _ = generate_syk_couplings(n_total_majoranas, seed, sparsity, linear_connectivity=4)
    
    # Build Local H (Sparse is fine here, it's small)
    H_local_sp = sparse.csc_matrix((dim_half, dim_half), dtype=np.complex64)
    chis_local_sp = [get_majorana(idx, n_half) for idx in range(n_total_majoranas)]
    
    for _, J, i, j, k, l in kept_terms:
        H_local_sp += J * (chis_local_sp[i] @ chis_local_sp[j] @ chis_local_sp[k] @ chis_local_sp[l])
        
    evals, evecs = scipy.linalg.eigh(H_local_sp.toarray(), overwrite_a=True)
    
    # 2. FAST TFD CONSTRUCTION
    sqrt_rho_diag = np.exp(-BETA * evals / 2)
    norm = np.sqrt(np.sum(sqrt_rho_diag**2)) 
    sqrt_rho_diag /= norm
    
    Psi_tfd_site = evecs @ np.diag(sqrt_rho_diag) @ evecs.T

    # 3. PREPARE OPERATORS
    X_0 = sparse.csr_matrix([[0, 1], [1, 0]], dtype=np.complex64)
    for _ in range(n_half - 1): X_0 = sparse.kron(X_0, sparse.eye(2))
    X_local = X_0.toarray()
    
    chis_local = [c.toarray() for c in chis_local_sp]

    # 4. FAST EVOLUTION FUNCTIONS
    def evolve_free(Psi_matrix, t):
        # U_local = U diag(e^{-itE}) U.T
        phases = np.exp(-1j * evals * t)
        U_loc = evecs @ (phases[:, None] * evecs.T)
        return U_loc @ Psi_matrix @ U_loc.T

    def apply_coupling(Psi_matrix):
            return apply_coupling_matrix_op(Psi_matrix, chis_local, WORMHOLE_STR)

    # 5. MAIN LOOP
    times = np.linspace(-TIME_RANGE, TIME_RANGE, TIME_STEPS)
    otoc_vals = []
    
    # Initial: X_L |TFD>
    Psi_0 = X_local @ Psi_tfd_site
    
    for t in times:
        Psi_back = evolve_free(Psi_0, -t)
        Psi_coup = apply_coupling(Psi_back)
        Psi_fwd = evolve_free(Psi_coup, t)
        
        # Path A
        Psi_A = Psi_fwd @ X_local.T
        val_A = np.vdot(Psi_tfd_site, Psi_A) 
        
        # Path B
        Psi_B_start = Psi_coup @ X_local.T
        Psi_B = evolve_free(Psi_B_start, t)
        val_B = np.vdot(Psi_tfd_site, Psi_B)
        
        otoc_vals.append(np.abs(val_A - val_B)**2)
        
    return np.max(otoc_vals)

def run(args):
    N_VALUES = [4, 6, 8, 10, 12]
    if args.sparsity is not None:
        SPARSITIES = [args.sparsity]
    else:
        SPARSITIES = [1.0, 0.6, 0.4, 0.2, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01]

    DATA_DIR = "v2syk_linear_data"
    os.makedirs(DATA_DIR, exist_ok=True)
    
    print(f"\nFIGURE 3: LINEAR GEOMETRY SWEEP")
    
    job_list = []
    for N in N_VALUES:
        for sp in SPARSITIES:
            filename = f"{DATA_DIR}/res_N{N}_sp{sp}.pkl"
            if not os.path.exists(filename):
                for k in range(args.ensemble):
                    job_list.append({'seed': 3000 + k, 'sp': sp, 'N': N})

    if job_list:
        print(f"Running {len(job_list)} simulations...")
        results_flat = Parallel(n_jobs=-1, verbose=5)(
            delayed(run_linear_universe)(job['seed'], job['sp'], job['N']) 
            for job in job_list
        )
        # Group and Save
        storage = {}
        for i, res in enumerate(results_flat):
            job = job_list[i]
            key = (job['N'], job['sp'])
            if key not in storage: storage[key] = []
            storage[key].append(res)
            
        for (N, sp), res_list in storage.items():
            filename = f"{DATA_DIR}/res_N{N}_sp{sp}.pkl"
            with open(filename, 'wb') as f:
                pickle.dump(res_list, f)

    # Plotting
    print("\nGenerating Figure 3...")
    plt.figure(figsize=(10, 7))
    colors = plt.cm.plasma(np.linspace(0, 0.9, len(N_VALUES)))
    
    for i, N in enumerate(N_VALUES):
        peaks = []
        valid_sparsities = []
        for sp in SPARSITIES:
            filename = f"{DATA_DIR}/res_N{N}_sp{sp}.pkl"
            if os.path.exists(filename):
                with open(filename, 'rb') as f:
                    res = pickle.load(f)
                peaks.append(np.mean(res))
                valid_sparsities.append(sp)
        
        if peaks:
            # Sort for line plot
            xy = sorted(zip(valid_sparsities, peaks))
            x = [val[0] for val in xy]
            y = [val[1] for val in xy]
            plt.plot(x, y, 'o-', linewidth=2.5, color=colors[i], label=f'N={N}')

    plt.xscale('log')
    plt.gca().invert_xaxis() 
    plt.axhline(y=0.005, color='k', linestyle='--', alpha=0.5, label='Classical Bound')
    plt.title("Holographic Phase Transition (Linear Geometry)")
    plt.xlabel("Sparsity")
    plt.ylabel("Wormhole Teleportation Signal")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/figure3_linear_phase.png", dpi=300)
    print("Saved figures/figure3_linear_phase.png")
    plt.show()