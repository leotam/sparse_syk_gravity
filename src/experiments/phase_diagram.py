import os
import time
import pickle
import numpy as np
import scipy.linalg
from joblib import Parallel, delayed
from scipy.ndimage import gaussian_filter1d

from src.config import BETA, WORMHOLE_STR, TIME_STEPS, TIME_RANGE, SFF_TIME_RANGE
from src.physics.hamiltonians import generate_syk_couplings, build_hamiltonian
from src.physics.algebra import get_majorana, get_pauli
from src.analysis.metrics import compute_sff
from src.analysis.visualization import plot_phase_diagram

def run_universe_variable_n(seed, sparsity, n_total_majoranas):
    n_per_side = n_total_majoranas // 2
    n_qubits = n_per_side
    n_total_qubits = n_qubits * 2 
    
    # 1. Hamiltonian
    kept_terms, _ = generate_syk_couplings(n_total_majoranas, seed, sparsity)
    H_local = build_hamiltonian(n_per_side, kept_terms, n_total_majoranas).toarray()
    
    evals, evecs = scipy.linalg.eigh(H_local)
    
    # 2. TFD State (Manual construction needed for variable N logic here)
    probs = np.exp(-BETA * evals / 2)
    probs /= np.linalg.norm(probs)
    tfd = np.zeros(2**n_total_qubits, dtype=np.complex128)
    for i, prob in enumerate(probs):
        tfd += prob * np.kron(evecs[:, i], np.conj(evecs[:, i]))

    # 3. Global Operators
    I_local = np.eye(2**n_qubits, dtype=complex)
    H_total = np.kron(H_local, I_local) + np.kron(I_local, H_local)
    
    V_op = np.zeros((2**n_total_qubits, 2**n_total_qubits), dtype=complex)
    chis = [get_majorana(idx, n_qubits) for idx in range(n_total_majoranas)]
    for j in range(n_total_majoranas):
        chi = chis[j].toarray()
        V_op += np.kron(chi, chi)
        
    X_mat = get_pauli('X').toarray()
    X_0 = X_mat.copy()
    for _ in range(n_qubits - 1): X_0 = np.kron(X_0, np.eye(2))
    
    X_L = np.kron(X_0, I_local)
    X_R = np.kron(I_local, X_0)
    
    evals_H, evecs_H = scipy.linalg.eigh(H_total)
    
    # 4. Metrics
    sff_times = np.linspace(0, SFF_TIME_RANGE, 100)
    sff_vals = compute_sff(evals, sff_times, BETA)
    
    # OTOC (Fast Basis)
    U_couple = scipy.linalg.expm(1j * WORMHOLE_STR * V_op)
    times = np.linspace(-TIME_RANGE, TIME_RANGE, TIME_STEPS)
    
    psi_start_E = evecs_H.conj().T @ (X_L @ tfd)
    U_couple_E = evecs_H.conj().T @ U_couple @ evecs_H
    X_R_E = evecs_H.conj().T @ X_R @ evecs_H
    tfd_E = evecs_H.conj().T @ tfd
    
    otoc_vals = []
    for t in times:
        exp_fwd = np.exp(-1j * evals_H * t)
        exp_back = np.exp(-1j * evals_H * (-t))
        
        v_A = U_couple_E @ (psi_start_E * exp_back)
        psi_A = v_A * exp_fwd
        val_A = np.vdot(tfd_E, X_R_E @ psi_A)
        
        v_B = X_R_E @ (U_couple_E @ (psi_start_E * exp_back))
        psi_B = v_B * exp_fwd
        val_B = np.vdot(tfd_E, psi_B)
        
        otoc_vals.append(np.abs(val_A - val_B)**2)
        
    return np.max(otoc_vals), sff_vals

def run(args):
    if args.sparsity is not None:
        SPARSITIES = [args.sparsity]
    else:
        SPARSITIES = [1.0, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, .01]

    os.makedirs(args.data_dir, exist_ok=True)
    
    print(f"\nFIGURE 2: PHASE DIAGRAM SWEEP")
    
    # --- 1. RUN SIMULATIONS ---
    for N in args.n_values:
        for sp in SPARSITIES:
            filename = f"{args.data_dir}/res_N{N}_sp{sp}.pkl"
            if os.path.exists(filename): continue
            
            print(f"Running N={N} Sp={sp}...")
            results = Parallel(n_jobs=-1)(
                delayed(run_universe_variable_n)(2000+k, sp, N) for k in range(args.ensemble)
            )
            with open(filename, 'wb') as f:
                pickle.dump(results, f)

    # --- 2. AGGREGATE DATA FOR PLOTTING ---
    phase_map_otoc = np.zeros((len(args.n_values), len(SPARSITIES)))
    phase_map_chaos = np.zeros((len(args.n_values), len(SPARSITIES)))
    
    for i, N in enumerate(args.n_values):
        for j, sp in enumerate(SPARSITIES):
            filename = f"{args.data_dir}/res_N{N}_sp{sp}.pkl"
            if os.path.exists(filename):
                with open(filename, 'rb') as f:
                    results = pickle.load(f)
                
                peaks = np.array([r[0] for r in results])
                sffs = np.array([r[1] for r in results])
                
                phase_map_otoc[i, j] = np.mean(peaks)
                
                # SFF Ramp Ratio Calculation
                avg_sff = np.mean(sffs, axis=0)
                smooth_sff = gaussian_filter1d(avg_sff, sigma=3.0)
                mid = len(smooth_sff) // 2
                phase_map_chaos[i, j] = np.max(smooth_sff[mid:]) / np.min(smooth_sff[:mid])
            else:
                phase_map_otoc[i, j] = np.nan
                phase_map_chaos[i, j] = np.nan

    # --- 3. PLOT (Using Helper) ---
    x_labels = [f"{s*100:g}%" for s in SPARSITIES]
    y_labels = [str(n) for n in args.n_values]
    
    print("Generating Figure 2...")
    os.makedirs("figures", exist_ok=True)
    plot_phase_diagram(phase_map_otoc, phase_map_chaos, x_labels, y_labels, save_path="figures/figure2_phase_diagram.png")