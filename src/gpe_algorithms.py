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

def get_start_vector(dim):
    """
    Return the normalized constant start vector u⁰ used by all methods.
    """
    start_vec = np.ones(dim)
    return start_vec / norm(start_vec)

def get_history(iterates, energy):
    """
    Compute the convergence history from the stored iterates.

    Parameters
    ----------
    iterates : ndarray
        Array of shape (N+1, dim) with the iterates u⁰, ..., uᴺ.
    energy : ndarray
        Array of length N+1 with the energies E(u⁰), ..., E(uᴺ).

    Returns
    -------
    residuum : ndarray
        ||uⁿ - uᴺ|| for n = 1, ..., N-1.
    energy : ndarray
        E(uⁿ) for n = 0, ..., N (passed through).
    diffs : ndarray
        ||uⁿ - uⁿ⁻¹|| for n = 1, ..., N.
    """
    residuum = norm(iterates[1:-1] - iterates[-1], axis=1)
    diffs = norm(np.diff(iterates, axis=0), axis=1)
    return residuum, energy, diffs

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
    n_steps : int
        Number of iteration steps N that were taken. The solution is the
        N-th iterate uᴺ (N = max_steps if not successful).
    solution : ndarray
        The last iterate uᴺ.
    residuum : ndarray
        Distance ||uⁿ - uᴺ|| of each iterate to the last iterate,
        for n = 1, ..., N-1 (length N-1).
    energy : ndarray
        Energy E(uⁿ) of each iterate including the start vector,
        for n = 0, ..., N (length N+1).
    diffs : ndarray
        Differences ||uⁿ - uⁿ⁻¹|| between consecutive iterates,
        for n = 1, ..., N (length N).
    """
    u_last = get_start_vector(dim)

    # Row n holds the n-th iterate uⁿ, row 0 is the start vector
    iterates = np.zeros((max_steps + 1, dim))
    energy = np.zeros(max_steps + 1)
    iterates[0] = u_last
    energy[0] = calc_E(u_last)

    success = False
    for n in range(1, max_steps + 1):
        A_u = A(u_last)
        u_cur = spsolve(A_u, u_last)
        u_cur /= norm(u_cur)

        iterates[n] = u_cur
        energy[n] = calc_E(u_cur)

        if norm(u_cur - u_last) < tol:
            success = True
            break

        u_last = u_cur

    return (success, n, u_cur, *get_history(iterates[:n + 1], energy[:n + 1]))

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
    n_steps : int
        Number of iteration steps N that were taken. The solution is the
        N-th iterate uᴺ (N = max_steps if not successful).
    solution : ndarray
        The last iterate uᴺ.
    residuum : ndarray
        Distance ||uⁿ - uᴺ|| of each iterate to the last iterate,
        for n = 1, ..., N-1 (length N-1).
    energy : ndarray
        Energy E(uⁿ) of each iterate including the start vector,
        for n = 0, ..., N (length N+1).
    diffs : ndarray
        Differences ||uⁿ - uⁿ⁻¹|| between consecutive iterates,
        for n = 1, ..., N (length N).

    Raises
    ------
    Exception
        If the optimization of the damping parameter τ fails.
    """
    damping_stop = 1e-3
    u_last = get_start_vector(dim)

    # Row n holds the n-th iterate uⁿ, row 0 is the start vector
    iterates = np.zeros((max_steps + 1, dim))
    energy = np.zeros(max_steps + 1)
    iterates[0] = u_last
    energy[0] = calc_E(u_last)

    damping = True
    success = False
    for n in range(1, max_steps + 1):
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

        iterates[n] = u_cur
        energy[n] = calc_E(u_cur)

        diff = norm(u_cur - u_last)
        if diff < tol:
            # At least one more run to make sure diff < tol isn't because of small damping steps
            if damping:
                logging.debug(f"Turned off damping after {n} steps because of diff < tol!")
                damping = False
            else:
                success = True
                break

        if damping and diff < damping_stop:
            logging.debug(f"Turned off damping after {n} steps!")
            damping = False

        u_last = u_cur

    return (success, n, u_cur, *get_history(iterates[:n + 1], energy[:n + 1]))

def shifted_inverse_iteration(A, calc_E, dim, max_steps, tol, sigma):
    """
    Run the inverse iteration algorithm with a fixed shift.

    Implements the shifted inverse iteration from [Hen24, Def. 5.1]:
        uⁿ⁺¹ = (A(uⁿ) - σI)⁻¹ uⁿ / ||(A(uⁿ) - σI)⁻¹ uⁿ||
    with a fixed shift σ ≠ λ. For linear problems, shifting σ towards the
    smallest eigenvalue λ accelerates convergence. For the GPE, [Hen24, Sec. 5]
    shows that the convergence rate contains the weighting function
    1 - 2β/(λ-σ)|u|², which blows up for σ → λ, so that shifting can slow
    down or destroy convergence, in particular for large β.

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
    sigma : float
        Fixed shift σ, typically 0 ≤ σ < λ (σ = 0 is the basic inverse iteration).

    Returns
    -------
    success : bool
        True if convergence was achieved within the tolerance, False otherwise.
    n_steps : int
        Number of iteration steps N that were taken. The solution is the
        N-th iterate uᴺ (N = max_steps if not successful).
    solution : ndarray
        The last iterate uᴺ.
    residuum : ndarray
        Distance ||uⁿ - uᴺ|| of each iterate to the last iterate,
        for n = 1, ..., N-1 (length N-1).
    energy : ndarray
        Energy E(uⁿ) of each iterate including the start vector,
        for n = 0, ..., N (length N+1).
    diffs : ndarray
        Differences ||uⁿ - uⁿ⁻¹|| between consecutive iterates,
        for n = 1, ..., N (length N).
    """
    shift = diags(np.full(dim, float(sigma)))
    u_last = get_start_vector(dim)

    # Row n holds the n-th iterate uⁿ, row 0 is the start vector
    iterates = np.zeros((max_steps + 1, dim))
    energy = np.zeros(max_steps + 1)
    iterates[0] = u_last
    energy[0] = calc_E(u_last)

    success = False
    for n in range(1, max_steps + 1):
        u_cur = spsolve(A(u_last) - shift, u_last)
        u_cur /= norm(u_cur)

        # If σ is larger than the smallest eigenvalue of A(uⁿ), the solve flips
        # the sign of the ground-state component ([JKM12, Prop. 2.2]). Since
        # A(u) only depends on u², u and -u are equivalent: align the sign with
        # u_last so that ||u_cur - u_last|| measures actual convergence.
        if u_cur @ u_last < 0:
            u_cur = -u_cur

        iterates[n] = u_cur
        energy[n] = calc_E(u_cur)

        if norm(u_cur - u_last) < tol:
            success = True
            break

        u_last = u_cur

    return (success, n, u_cur, *get_history(iterates[:n + 1], energy[:n + 1]))
