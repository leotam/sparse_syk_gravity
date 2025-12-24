import numpy as np
from scipy import sparse

def get_pauli(char, dtype=complex, format='csr'):
    if char == 'I': return sparse.eye(2, dtype=dtype, format=format)
    if char == 'X': return sparse.csr_matrix([[0, 1], [1, 0]], dtype=dtype)
    if char == 'Y': return sparse.csr_matrix([[0, -1j], [1j, 0]], dtype=dtype)
    if char == 'Z': return sparse.csr_matrix([[1, 0], [0, -1]], dtype=dtype)
    raise ValueError(f"Unknown Pauli: {char}")

def get_majorana(idx, n_qubits):
    k = idx // 2
    is_odd = idx % 2
    ops = [get_pauli('Z')] * k
    ops.append(get_pauli('Y') if is_odd else get_pauli('X'))
    ops.extend([get_pauli('I')] * (n_qubits - k - 1))
    
    mat = ops[0]
    for i in range(1, len(ops)):
        mat = sparse.kron(mat, ops[i], format='csr')
    return mat