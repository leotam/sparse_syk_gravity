import os
import pickle
import numpy as np
import scipy.linalg
from scipy import sparse
from joblib import Parallel, delayed

from src.config import BETA, WORMHOLE_STR
from src.physics.algebra import get_pauli, get_majorana
from src.physics.hamiltonians import generate_syk_couplings, build_hamiltonian
from src.physics.dynamics import evolve_operator_matrix
from src.analysis.visualization import plot_shapiro_delay

def run_rotation_universe(seed, strength, sparsity=0.1, n_majoranas=12):
    n_half = n_majoranas // 2
    
    # 1. Hamiltonian
    kept_terms, _ = generate_syk_couplings(n_majoranas, seed, sparsity)
    H_local = build_hamiltonian(n_half, kept_terms, n_majoranas).toarray()
    
    evals, evecs = scipy.linalg.eigh(H_local)
    
    # 2. TFD
    probs = np.exp(-BETA * evals / 2)
    probs /= np.linalg.norm(probs)
    Psi_tfd = np.zeros(2**n_majoranas, dtype=np.complex128)
    for i, p in enumerate(probs):
        Psi_tfd += p * np.kron(evecs[:, i], np.conj(evecs[:, i]))

    # 3. Operators
    X_0 = sparse.kron(get_pauli('X'), sparse.eye(2**(n_half-1))).toarray()
    # W operator (I x Z)
    Z_part = sparse.kron(sparse.eye(2), get_pauli('Z'))
    W_0 = sparse.kron(Z_part, sparse.eye(2**(n_half-2))).toarray()
    U_rock = scipy.linalg.expm(-1j * strength * W_0)

    # 4. Dynamics
    ROCK_OFFSET = 3.0
    times = np.linspace(-7.0, 7.0, 140)
    
    # Re-diagonalize Global H for simplicity
    I_loc = np.eye(2**n_half)
    H_total = np.kron(H_local, I_loc) + np.kron(I_loc, H_local)
    evals_H, evecs_H = scipy.linalg.eigh(H_total)

    chis = [get_majorana(i, n_half).toarray() for i in range(n_majoranas)]
    V_op = sum([np.kron(c, c) for c in chis])
    
    # Pre-compute Unitaries
    def get_U(t):
        return evecs_H @ (np.exp(-1j * evals_H * t)[:, None] * evecs_H.conj().T)

    U_rock_global = np.kron(U_rock, I_loc) 
    X_op_global = np.kron(X_0, I_loc)      
    
    U_couple = scipy.linalg.expm(1j * WORMHOLE_STR * V_op)

    curve = []
    for t in times:
        psi = get_U(-ROCK_OFFSET) @ Psi_tfd
        psi = U_rock_global @ psi
        psi = get_U(ROCK_OFFSET) @ psi
        psi = X_op_global @ psi
        psi = U_couple @ psi
        psi = get_U(t) @ psi
        
        # Measure Right side X_R
        X_R = np.kron(I_loc, X_0)
        val = np.vdot(Psi_tfd, X_R @ psi)
        curve.append(np.abs(val)**2)
        
    return np.array(curve)

def run(args):
    ROCK_STRENGTHS = [0.0, 3.14/8, 3.14/4, 3.14/8*3, 1.57]
    DATA_DIR = "v2syk_shapiro_data"
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Use args.n_majoranas (defaulting to 12 in CLI if not set)
    N_MAJ = args.n_majoranas
    
    print(f"\nFIGURE 4: SHAPIRO DELAY (N={N_MAJ}, Ensemble={args.ensemble})")
    
    results_map = {st: [] for st in ROCK_STRENGTHS}
    
    for st in ROCK_STRENGTHS:
        # Include N in filename so smoke tests don't corrupt real data
        cache_file = f"{DATA_DIR}/shapiro_N{N_MAJ}_str{st:.3f}.pkl"
        
        jobs_to_run = []
        for k in range(args.ensemble):
            jobs_to_run.append(k)

        # Force re-run for simplicity of logic here, or add cache check
        print(f"Simulating Strength {st:.2f}...")
        batch_res = Parallel(n_jobs=-1)(
            delayed(run_rotation_universe)(k, st, args.sparsity, N_MAJ) 
            for k in jobs_to_run
        )
        results_map[st] = batch_res
        
        with open(cache_file, 'wb') as f:
            pickle.dump(batch_res, f)

    # Analyze
    times = np.linspace(-7.0, 7.0, 140)
    delays = []
    errors = []
    
    baseline_curve = np.mean(results_map[ROCK_STRENGTHS[0]], axis=0)
    t_0 = times[np.argmax(baseline_curve)]
    
    for st in ROCK_STRENGTHS:
        curves = np.array(results_map[st])
        avg_curve = np.mean(curves, axis=0)
        t_st = times[np.argmax(avg_curve)]
        delays.append(t_st - t_0)
        
        individual_delays = [times[np.argmax(c)] - t_0 for c in curves]
        errors.append(np.std(individual_delays) / np.sqrt(len(curves)))

    os.makedirs("figures", exist_ok=True)
    plot_shapiro_delay(ROCK_STRENGTHS, delays, errors, save_path="figures/figure4_shapiro.png")