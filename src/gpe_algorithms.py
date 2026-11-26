import numpy as np
from numpy.linalg import norm

from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.optimize import minimize_scalar

import logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

def calculate_energy(v, L, V, beta, h_inv_sq):
    """
    Calculate the discrete Gross-Pitaevskii energy.

    Discretization of the energy functional from [Hen24, Eq. (2)]
        E(u) = ½ ∫ ( ½ |∇u|² + V u² + β/2 u⁴ ) dx
             = ¼ ∫ |∇u|² + ½ ∫ V u² + β/4 ∫ u⁴,
    which belongs to the GPE  -½Δu + Vu + β u³ = λu.

    The vector v is Euclidean-normalized (Σ v² = 1), which corresponds to
    the L²-normalized wavefunction u = v / h (cf. [JKM12, Sec. 5.1]).
    With integrals replaced by h² Σ this gives
        E(v) = ¼ vᵀ L v + ½ Σ V v² + β / (4 h²) Σ v⁴,
    where L is the discrete (positive) Laplacian -Δ_h.

    Parameters
    ----------
    v : ndarray
        Current vector (flattened 2D grid values).
    L : scipy.sparse.spmatrix
        Discrete Laplacian -Δ_h (see gpe_problem.get_L).
    V : ndarray
        Potential values on the grid (flattened like v).
    beta : float
        Repulsion constant (nonlinearity parameter).
    h_inv_sq : float
        Inverse squared grid spacing (1/h²).

    Returns
    -------
    float
        Computed energy of the system.
    """
    v_sq = v * v

    kinetic = 0.25 * (v @ (L @ v))
    potential = 0.5 * np.sum(V * v_sq)
    interaction = 0.25 * beta * h_inv_sq * np.sum(v_sq * v_sq)

    return kinetic + potential + interaction

def inverse_iteration(A, calc_E, dim, max_steps, tol):
    """
    Run the general inverse iteration algorithm.

    Parameters
    ----------
    A : callable
        Function that takes a vector v and returns a matrix/operator 
        to be applied in the inverse iteration step.
    calc_E : callable
        Function that accepts a vector and returns the corresponding energy.
    dim : int
        Dimension of the problem (length of vector space).
    max_steps : int
        Maximum number of iterations allowed.
    tol : float
        Convergence tolerance for stopping criterion (based on ||uⁿ - uⁿ⁻¹||).

    Returns
    -------
    success : bool
        True if convergence was achieved within the tolerance, False otherwise.
    iterations : int
        Number of iterations needed to converge, or np.inf if not successful.
    solution : ndarray
        The computed solution vector.
    residuum : ndarray
        Array of residuum values over iterations.
    energy : ndarray
        Array of energy values over iterations.
    diffs : ndarray
        Array of ||uⁿ - uⁿ⁻¹|| differences for each iteration.
    """
    start_vec = np.ones(dim)
    start_vec /= norm(start_vec)
    u_last = start_vec
    
    iterated_values = np.zeros((max_steps, dim))
    energy = np.zeros(max_steps)
    
    for i in range(0, max_steps):
        A_u = A(u_last)
        u_cur = spsolve(A_u, u_last)
        u_cur /= norm(u_cur)

        iterated_values[i] = u_cur
        energy[i] = calc_E(u_cur)
        
        if norm(u_cur - u_last) < tol:
            residuum = norm(iterated_values[:i] - u_cur, axis=1)
            diffs = norm(
                iterated_values[:i] - np.insert(iterated_values, 0, start_vec, axis=0)[:i],
                axis=1
            )
            return (True, i, u_cur, residuum, energy[:i], diffs)

        u_last = u_cur

    diffs = norm(
        iterated_values - np.insert(iterated_values, 0, start_vec, axis=0)[:-1],
        axis=1
    )
    return (False, np.inf, u_cur, np.array([np.inf]), energy, diffs)

def compute_line(tau, u_cur, u_last):
    """
    Compute a normalized linear combination between two iteration vectors.

    This function interpolates between `u_last` and `u_cur`, 
    applying a correction factor `gamma` to ensure stability.

    Parameters
    ----------
    tau : float
        Interpolation parameter (0 ≤ tau ≤ 1 typically).
    u_cur : ndarray
        Current iteration vector.
    u_last : ndarray
        Previous iteration vector.

    Returns
    -------
    ndarray
        Normalized interpolated vector.

    Raises
    ------
    Exception
        If the dot product between `u_cur` and `u_last` is too small 
        (to avoid numerical instability due to large `gamma`).
    """
    tmp = u_cur.T @ u_last

    if tmp < 1e-12:
        raise Exception("Do not want to calculate inverse: Gamma too large!")
    
    gamma = 1. / tmp
    res = (1. - tau) * u_last + tau * gamma * u_cur

    return res / norm(res)

def dampened_inverse_iteration(A, calc_E, dim, max_steps, tol):
    """
    Run the dampened inverse iteration algorithm with adaptive dampening.

    The algorithm adaptively applies damping during iteration and
    turns it off once ||uⁿ⁺¹ − uⁿ|| < 1e-3. After damping is disabled,
    standard inverse iteration proceeds until convergence.

    Parameters
    ----------
    A : callable
        Function that takes a vector v and returns a matrix/operator 
        to be applied in the inverse iteration step.
    calc_E : callable
        Function that accepts a vector and returns the corresponding energy.
    dim : int
        Dimension of the problem (length of vector space).
    max_steps : int
        Maximum number of iterations allowed.
    tol : float
        Convergence tolerance for stopping criterion (based on ||uⁿ - uⁿ⁻¹||).

    Returns
    -------
    success : bool
        True if convergence was achieved within the tolerance, False otherwise.
    iterations : int
        Number of iterations needed to converge, or np.inf if not successful.
    solution : ndarray
        The computed solution vector.
    residuum : ndarray
        Array of residuum values over iterations.
    energy : ndarray
        Array of energy values over iterations.
    diffs : ndarray
        Array of ||uⁿ - uⁿ⁻¹|| differences for each iteration.

    Raises
    ------
    Exception
        If the optimization of the damping parameter τ fails.
    """
    damping_stop = 1e-3
    start_vec = np.ones(dim)
    start_vec /= norm(start_vec)
    u_last = start_vec

    iterated_values = np.zeros((max_steps, dim))
    energy = np.zeros(max_steps)
    
    damping = True
    for i in range(0, max_steps):
        A_u = A(u_last)
        u_cur = spsolve(A_u, u_last)

        if damping:
            def objective(tau):
                u_tau = compute_line(tau, u_cur, u_last)
                return calc_E(u_tau)

            opt_tau = minimize_scalar(objective, bounds=(0, 2), method='bounded')
            if not opt_tau.success:
                raise Exception("Could not optimize for tau!")

            tau = opt_tau.x
            logging.debug(f"Dampening chose optimal tau={tau}.")

            # compute_line is already normalized
            u_cur = compute_line(tau, u_cur, u_last)
        else:
            u_cur /= norm(u_cur)

        iterated_values[i] = u_cur
        energy[i] = calc_E(u_cur)
        
        diff = norm(u_cur - u_last)
        if diff < tol:
            # At least one more run to make sure diff < tol isn't because of small damping steps
            if damping:
                logging.debug(f"Turned off damping after {i + 1} steps because of diff < tol!")
                damping = False
            else:
                residuum = norm(iterated_values[:i] - u_cur, axis=1)
                diffs = norm(
                    iterated_values[:i] - np.insert(iterated_values, 0, start_vec, axis=0)[:i],
                    axis=1
                )
                return (True, i, u_cur, residuum, energy[:i], diffs)

        if damping and diff < damping_stop:
            logging.debug(f"Turned off damping after {i + 1} steps!")
            damping = False
        
        u_last = u_cur

    diffs = norm(
        iterated_values - np.insert(iterated_values, 0, start_vec, axis=0)[:-1],
        axis=1
    )
    return (False, np.inf, u_cur, np.array([np.inf]), energy, diffs)

def shifted_inverse_iteration(A, calc_E, dim, max_steps, tol):
    """
    Run the inverse iteration algorithm with dynamic shifting.

    The algorithm applies a Rayleigh-quotient shift at each step 
    (based on the last iterate) to accelerate convergence.

    Parameters
    ----------
    A : callable
        Function that takes a vector v and returns a matrix/operator 
        to be applied in the inverse iteration step.
    calc_E : callable
        Function that accepts a vector and returns the corresponding energy.
    dim : int
        Dimension of the problem (length of vector space).
    max_steps : int
        Maximum number of iterations allowed.
    tol : float
        Convergence tolerance for stopping criterion (based on ||uⁿ - uⁿ⁻¹||).

    Returns
    -------
    success : bool
        True if convergence was achieved within the tolerance, False otherwise.
    iterations : int
        Number of iterations needed to converge, or np.inf if not successful.
    solution : ndarray
        The computed solution vector.
    residuum : ndarray
        Array of residuum values over iterations.
    energy : ndarray
        Array of energy values over iterations.
    diffs : ndarray
        Array of ||uⁿ - uⁿ⁻¹|| differences for each iteration.
    """
    ones = np.ones(dim)
    start_vec = ones / norm(ones)
    u_last = start_vec
    
    iterated_values = np.zeros((max_steps, dim))
    energy = np.zeros(max_steps)
    
    for i in range(0, max_steps):
        A_u = A(u_last)
        shift = u_last.T @ A_u @ u_last / (u_last.T @ u_last)
        
        u_cur = spsolve(A_u - diags(shift * ones), u_last)
        u_cur /= norm(u_cur)

        # The Rayleigh quotient is >= the smallest eigenvalue of A_u, so
        # A_u - shift*I has a negative eigenvalue along the ground state and the
        # solve flips the sign of the ground-state component. Since A(u) only
        # depends on u², u and -u are equivalent: align the sign with u_last so
        # that ||u_cur - u_last|| measures actual convergence.
        if u_cur @ u_last < 0:
            u_cur = -u_cur

        iterated_values[i] = u_cur
        energy[i] = calc_E(u_cur)
        
        if norm(u_cur - u_last) < tol:
            residuum = norm(iterated_values[:i] - u_cur, axis=1)
            diffs = norm(
                iterated_values[:i] - np.insert(iterated_values, 0, start_vec, axis=0)[:i],
                axis=1
            )
            return (True, i, u_cur, residuum, energy[:i], diffs)

        u_last = u_cur

    diffs = norm(
        iterated_values - np.insert(iterated_values, 0, start_vec, axis=0)[:-1],
        axis=1
    )
    return (False, np.inf, u_cur, np.array([np.inf]), energy, diffs)
