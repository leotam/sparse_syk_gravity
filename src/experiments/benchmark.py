import time, os
import numpy as np
import scipy.linalg
from joblib import Parallel, delayed
from scipy.ndimage import gaussian_filter1d

from src.config import BETA, WORMHOLE_STR, TIME_STEPS, TIME_RANGE, SFF_TIME_RANGE
from src.physics.hamiltonians import generate_syk_couplings, build_hamiltonian
from src.physics.dynamics import get_thermal_state, evolve_state_vector
from src.physics.algebra import get_pauli, get_majorana
from src.analysis.metrics import compute_sff
from src.analysis.visualization import plot_benchmark_results

def run_universe(seed, sparsity, n_per_side=6):
    n_majoranas = n_per_side * 2
    n_total_qubits = n_per_side * 2
    
    # 1. Hamiltonian
    kept_terms, total_terms = generate_syk_couplings(n_majoranas, seed, sparsity)
    H_local_sparse = build_hamiltonian(n_per_side, kept_terms, n_majoranas)
    H_local = H_local_sparse.toarray()
    
    evals, evecs = scipy.linalg.eigh(H_local)

    # 2. Metric: SFF (Using helper)
    sff_times = np.linspace(0, SFF_TIME_RANGE, TIME_STEPS)
    sff_vals = compute_sff(evals, sff_times, BETA)

    # 3. Setup TFD
    tfd = get_thermal_state(evals, evecs, BETA)

    # 4. Global Operators
    I_local = np.eye(2**n_per_side, dtype=complex)
    H_total = np.kron(H_local, I_local) + np.kron(I_local, H_local)
    
    # Tunneling Operator V
    V_op = np.zeros((2**n_total_qubits, 2**n_total_qubits), dtype=complex)
    for j in range(n_majoranas):
        chi = get_majorana(j, n_per_side).toarray()
        V_op += np.kron(chi, chi)
        
    # Operator X_L and X_R
    X_mat = get_pauli('X').toarray()
    X_0 = X_mat
    for _ in range(n_per_side - 1): X_0 = np.kron(X_0, np.eye(2))
    X_L = np.kron(X_0, np.eye(2**n_per_side))
    X_R = np.kron(np.eye(2**n_per_side), X_0)

    # Diagonalize Total H for fast evolution
    evals_H, evecs_H = scipy.linalg.eigh(H_total)
    U_couple = scipy.linalg.expm(1j * WORMHOLE_STR * V_op)

    # 5. OTOC Evolution
    times = np.linspace(-TIME_RANGE, TIME_RANGE, TIME_STEPS)
    otoc_vals = []

    psi_start_A = X_L @ tfd
    
    for t in times:
        # Path A: Perturbed
        step1_A = evolve_state_vector(psi_start_A, evals_H, evecs_H, -t)
        step2_A = U_couple @ step1_A
        psi_A = evolve_state_vector(step2_A, evals_H, evecs_H, t)
        val_A = np.vdot(tfd, X_R @ psi_A)

        # Path B: Unperturbed
        step1_B = evolve_state_vector(psi_start_A, evals_H, evecs_H, -t)
        step2_B = U_couple @ step1_B
        step3_B = X_R @ step2_B
        psi_B = evolve_state_vector(step3_B, evals_H, evecs_H, t)
        val_B = np.vdot(tfd, psi_B)
        
        otoc_vals.append(np.abs(val_A - val_B)**2)
        
    return np.array(otoc_vals), sff_vals, total_terms, len(kept_terms)

def run(args):
    print(f"\nFIGURE 1: SYK COMPRESSION BENCHMARK (Ensemble={args.ensemble})")
    
    if args.sparsity is not None:
        sparsities = [args.sparsity]
    else:
        sparsities = [1.0, 0.4, 0.2, 0.1, 0.05]
        
    results_table = []
    sff_history = [] 

    for sp in sparsities:
        start_time = time.time()
        results = Parallel(n_jobs=-1)(
            delayed(run_universe)(2000 + k, sp) for k in range(args.ensemble)
        )
        
        otocs = np.array([r[0] for r in results])
        sffs = np.array([r[1] for r in results])
        total_terms = results[0][2]
        kept_terms = results[0][3]
        
        avg_otoc = np.mean(otocs, axis=0)
        avg_sff = np.mean(sffs, axis=0)
        sff_history.append(avg_sff)
        peak_otoc = np.max(avg_otoc)
        
        # Simple Ramp Calculation
        smoothed_sff = gaussian_filter1d(avg_sff, sigma=2.0)
        ratio = smoothed_sff[-1] / np.min(smoothed_sff)
        
        print(f"Sp {sp*100:4.1f}% | Terms: {kept_terms}/{total_terms} | Peak: {peak_otoc:.4f} | SFF Ratio: {ratio:.2f}")
        results_table.append({"sp": sp, "peak": peak_otoc})

    # --- PLOTTING (Using Helper) ---
    print("\nGenerating Figure 1...")
    times = np.linspace(0, 20.0, len(sff_history[0]))
    os.makedirs("figures", exist_ok=True)
    plot_benchmark_results(results_table, sff_history, times, save_path="figures/figure1_benchmark.png")    