import numpy as np

def get_thermal_state(evals, evecs, beta):
    """
    Constructs the Thermofield Double (TFD) state vector.
    Returns: Normalized TFD state vector (dim = dim_local^2).
    """
    probs = np.exp(-beta * evals / 2)
    probs /= np.linalg.norm(probs)
    
    dim = len(evals)
    # TFD = sum_i sqrt(p_i) |i>_L |i*>_R
    tfd = np.zeros(dim * dim, dtype=np.complex128)
    for i, prob in enumerate(probs):
        term = np.kron(evecs[:, i], np.conj(evecs[:, i]))
        tfd += prob * term
    return tfd

def evolve_state_vector(psi, evals_H, evecs_H, t):
    """
    Evolves a state vector using pre-computed diagonalization.
    psi(t) = U e^{-iHt} U^dagger psi(0)
    """
    # 1. Rotate to Energy Basis: phi = U^dagger @ psi
    phi = evecs_H.conj().T @ psi
    
    # 2. Evolve phases: phi(t) = exp(-iEt) * phi
    phi_t = np.exp(-1j * evals_H * t) * phi
    
    # 3. Rotate back: psi(t) = U @ phi(t)
    return evecs_H @ phi_t

def evolve_operator_matrix(op_matrix, evals, evecs, t):
    """
    Evolves a dense operator matrix in the Heisenberg picture.
    O(t) = U^dagger(t) O U(t)
    """
    phases = np.exp(-1j * evals * t)
    # Construct unitary U(t)
    U_t = evecs @ (phases[:, None] * evecs.T)
    
    return U_t.conj().T @ op_matrix @ U_t

def apply_coupling_matrix_op(psi_matrix, chis_local, strength):
    """
    Applies the SYK wormhole coupling in the Matrix (Operator-State) representation.
    Used in Figure 3 (Linear Geometry).
    
    The coupling is V = exp(i * strength * sum(chi_L chi_R)).
    In the matrix representation |Rho>, this acts as:
    Rho -> exp(i*g*chi_L*chi_R) |Rho>
         ~= cos(g)*Rho + i*sin(g)*chi*Rho*chi^T
         
    Args:
        psi_matrix: The state as a (dim x dim) matrix.
        chis_local: List of Majorana matrices (dim x dim).
        strength: The coupling strength (mu).
    """
    res = psi_matrix.copy()
    cos_g = np.cos(strength)
    sin_g = np.sin(strength)
    
    # Apply terms sequentially (Trotter approximation)
    for chi in chis_local:
        # The operation chi_L chi_R on the TFD state corresponds to 
        # chi @ Rho @ chi.T in the matrix representation.
        term = chi @ res @ chi.T
        res = cos_g * res + 1j * sin_g * term
        
    return res