import numpy as np
from scipy import sparse
from src.config import COUPLING_J
from src.physics.algebra import get_majorana
from itertools import combinations

def generate_syk_couplings(n_majoranas, seed, sparsity=1.0, linear_connectivity=None):
    """
    Generates and prunes SYK couplings with corrected variance scaling.
    linear_connectivity: If int, restricts interactions to nearest neighbors range.
    """
    rng = np.random.RandomState(seed)
    
    # Scale variance by 1/p to keep energy constant
    var_base = 6 * (COUPLING_J**2) / (n_majoranas**3)
    var_scaled = var_base / sparsity
    std_dev = np.sqrt(var_scaled)
    
    terms = []
    
    # Loop strategy: Full or Linear
    if linear_connectivity:
        for i in range(n_majoranas):
            window_end = min(i + linear_connectivity + 1, n_majoranas)
            for j, k, l in combinations(range(i + 1, window_end), 3):
                J = rng.normal(0, std_dev)
                terms.append((np.abs(J), J, i, j, k, l))
    else:
        for i, j, k, l in combinations(range(n_majoranas), 4):
            J = rng.normal(0, std_dev)
            terms.append((np.abs(J), J, i, j, k, l))

    # Pruning
    terms.sort(key=lambda x: x[0], reverse=True)
    n_keep = max(1, int(len(terms) * sparsity))
    return terms[:n_keep], len(terms)

def build_hamiltonian(n_qubits, terms, n_majoranas):
    """Constructs the sparse matrix from terms."""
    H = sparse.csc_matrix((2**n_qubits, 2**n_qubits), dtype=complex)
    chis = [get_majorana(idx, n_qubits) for idx in range(n_majoranas)]
    
    for _, J, i, j, k, l in terms:
        term_op = chis[i] @ chis[j] @ chis[k] @ chis[l]
        H += J * term_op
    return H