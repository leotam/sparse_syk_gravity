import time, os
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt

def compute_gap_ratio_logic(evals_cpu):
    evals_cpu.sort()
    evals_unique = np.unique(np.round(evals_cpu, decimals=8))
    spacings = np.diff(evals_unique)
    spacings = spacings[spacings > 1e-10]
    
    if len(spacings) < 20: return np.nan, np.nan
        
    start_idx = len(spacings) // 4
    end_idx = 3 * len(spacings) // 4
    spacings_middle = spacings[start_idx:end_idx]
    
    s_n = spacings_middle[:-1]
    s_np1 = spacings_middle[1:]
    r_vals = np.minimum(s_n, s_np1) / np.maximum(s_n, s_np1)
    return np.mean(r_vals), np.std(r_vals)

def run_spectral_job(params):
    """
    Worker function executed by each process.
    """
    try:
        import cupy as cp
        import cupyx.scipy.sparse as cusparse
        
        # Internal helper for GPU ops (Self-contained for pickling)
        def get_pauli_gpu(char):
            if char == 'I': return cusparse.eye(2, dtype=cp.complex128, format='csr')
            if char == 'X': return cusparse.csr_matrix(cp.array([[0, 1], [1, 0]], dtype=cp.complex128))
            if char == 'Y': return cusparse.csr_matrix(cp.array([[0, -1j], [1j, 0]], dtype=cp.complex128))
            if char == 'Z': return cusparse.csr_matrix(cp.array([[1, 0], [0, -1]], dtype=cp.complex128))

        def get_majorana_gpu(idx, n_qubits):
            k = idx // 2
            is_odd = idx % 2
            ops = [get_pauli_gpu('Z')] * k
            ops.append(get_pauli_gpu('Y') if is_odd else get_pauli_gpu('X'))
            ops.extend([get_pauli_gpu('I')] * (n_qubits - k - 1))
            mat = ops[0]
            for i in range(1, len(ops)):
                mat = cusparse.kron(mat, ops[i], format='csr')
            return mat

        n_total, sparsity, seed, gpu_id = params
        
        with cp.cuda.Device(gpu_id):
            cp.get_default_memory_pool().free_all_blocks()
            
            rng = np.random.RandomState(seed)
            n_qubits = n_total // 2
            dim = 2**n_qubits
            
            var_base = 6.0 / (n_total**3)
            var_sparse = var_base / sparsity
            std_scaled = np.sqrt(var_sparse)
            
            total_possible = (n_total * (n_total-1) * (n_total-2) * (n_total-3)) // 24
            n_target = int(total_possible * sparsity)
            
            H_sparse = cusparse.csr_matrix((dim, dim), dtype=cp.complex128)
            chis_gpu = [get_majorana_gpu(i, n_qubits) for i in range(n_total)]
            
            for _ in range(n_target):
                idxs = rng.choice(n_total, 4, replace=False)
                J = rng.normal(0, std_scaled)
                term = chis_gpu[idxs[0]] @ chis_gpu[idxs[1]] @ chis_gpu[idxs[2]] @ chis_gpu[idxs[3]]
                H_sparse = H_sparse + term * J
            
            H_dense = H_sparse.toarray()
            del H_sparse, chis_gpu
            evals = cp.linalg.eigvalsh(H_dense)
            del H_dense
            
            evals_cpu = cp.asnumpy(evals).real
            r_mean, r_std = compute_gap_ratio_logic(evals_cpu)
            return (n_total, sparsity, r_mean, r_std)
            
    except Exception as e:
        print(f"GPU Worker Error: {e}")
        return (params[0], params[1], np.nan, np.nan)

def run(args):
    if args.sparsity is not None:
        SPARSITIES = [args.sparsity]
    else:
        SPARSITIES = [1.0, 0.4, 0.1, 0.05]
    
    if args.gpu_ids is None:
        args.gpu_ids = [0]
        
    print("\n" + "="*60)
    print(f"FIGURE 5: GPU GAP RATIO SCALING")
    if args.gpu_ids is None:
        args.gpu_ids = [0]
        
    print("\n" + "="*60)
    print(f"FIGURE 5: GPU GAP RATIO SCALING")
    print(f"GPUs: {args.gpu_ids} | N_Values: {args.n_values} | Ensemble: {args.ensemble}")
    print("="*60)
    
    mp.set_start_method('spawn', force=True)

    # Queue
    tasks = []
    job_idx = 0
    for n_total in args.n_values:
        for sp in SPARSITIES:
            for k in range(args.ensemble):
                seed = 42 + n_total*1000 + k
                gpu_id = args.gpu_ids[job_idx % len(args.gpu_ids)]
                tasks.append((n_total, sp, seed, gpu_id))
                job_idx += 1
    
    print(f"Running {len(tasks)} tasks...")
    
    start_time = time.time()
    with mp.Pool(processes=len(args.gpu_ids)) as pool:
        raw_results = pool.map(run_spectral_job, tasks)
    print(f"Total Runtime: {time.time() - start_time:.2f}s")

    # Aggregate and Plot
    data_agg = {}
    for res in raw_results:
        n, sp, r_mean, _ = res
        if not np.isnan(r_mean):
            if (n, sp) not in data_agg: data_agg[(n, sp)] = []
            data_agg[(n, sp)].append(r_mean)

    plt.figure(figsize=(10, 7))
    colors = ['#D32F2F', '#1976D2', '#388E3C', '#FBC02D']
    
    for idx, sp in enumerate(SPARSITIES):
        n_list = []
        r_list = []
        r_err_list = []
        
        for n in args.n_values:
            if (n, sp) in data_agg:
                vals = data_agg[(n, sp)]
                avg = np.mean(vals)
                err = np.std(vals) / np.sqrt(len(vals))
                n_list.append(n)
                r_list.append(avg)
                r_err_list.append(err)
        
        if n_list:
            plt.errorbar(n_list, r_list, yerr=r_err_list, marker='o', 
                         linewidth=2, capsize=4, color=colors[idx], label=f'Sparsity {int(sp*100)}%')

    plt.axhline(0.5996, color='blue', linestyle='--', alpha=0.5, label='GUE (0.60)')
    plt.axhline(0.5359, color='green', linestyle='--', alpha=0.5, label='GOE (0.54)')
    plt.axhline(0.3863, color='red', linestyle=':', label='Poisson (0.39)')
    
    plt.xlabel('Total Majoranas N')
    plt.ylabel('Gap Ratio <r>')
    plt.title(f'Chaos Stability vs. System Size')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.ylim(0.35, 0.65)
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/figure5_gap_scaling.png", dpi=300)
    print("Saved figures/figure5_gap_scaling.png")