import numpy as np
from scipy.sparse import diags, kron

def get_D2(dim_sqrt, h_inv_sq):
    """
    Construct the 1D discrete Laplacian matrix.

    Builds a tridiagonal matrix of size (dim_sqrt, dim_sqrt) representing
    the 1D Laplace operator with second-order finite differences, scaled by 1/h².

    Parameters
    ----------
    dim_sqrt : int
        Number of grid points in one spatial dimension.
    h_inv_sq : float
        Inverse squared grid spacing (1/h²).

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse CSR matrix representing the 1D Laplace operator.
    """
    diagonals = [
        -np.ones(dim_sqrt - 1), 
        2 * np.ones(dim_sqrt), 
        -np.ones(dim_sqrt - 1)
    ]
    offsets = [-1, 0, 1]
    return 1.0 * h_inv_sq * diags(diagonals, offsets, format='csr')


def get_L(dim_sqrt, h_inv_sq):
    """
    Construct the 2D discrete Laplacian matrix.

    Builds a sparse matrix of size (dim_sqrt², dim_sqrt²) representing
    the 2D Laplace operator on a Cartesian grid, using the Kronecker sum
    of 1D Laplace matrices.

    Parameters
    ----------
    dim_sqrt : int
        Number of grid points in one spatial dimension.
    h_inv_sq : float
        Inverse squared grid spacing (1/h²).

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse CSR matrix representing the 2D Laplace operator.
    """
    D_2 = get_D2(dim_sqrt, h_inv_sq)
    I = diags(np.ones(dim_sqrt), 0, format='csr')
    return kron(D_2, I) + kron(I, D_2)

def get_M(bound, steps):
    """
    Construct the harmonic potential matrix on a 2D grid.

    The potential is defined as 
        V(x, y) = 0.5 * (x² + y²),
    discretized on the square domain [-bound, bound]² with a uniform grid.

    Parameters
    ----------
    bound : float
        Half-width of the square domain (grid spans [-bound, bound]).
    steps : int
        Number of grid points along each dimension.

    Returns
    -------
    scipy.sparse.dia_matrix
        Sparse diagonal matrix (size steps² × steps²) with the potential
        values on the diagonal.
    """
    x = np.linspace(-bound, bound, steps)
    y = np.linspace(-bound, bound, steps)
    X, Y = np.meshgrid(x, y)

    f = lambda x, y: 0.5 * (x * x + y * y)
    Z = f(X, Y)
    return diags(Z.flatten())

def get_L_M(bound, steps, h_inv_sq):
    """
    Construct the linear operator L_M for the Gross–Pitaevskii equation (GPE).

    The operator represents the constant part of the Hamiltonian:
        L_M = Δ + V,
    where Δ is the discrete Laplacian and V is the external potential
    (here taken as the harmonic trap).

    Parameters
    ----------
    bound : float
        Half-width of the spatial domain (grid spans [-bound, bound]).
    steps : int
        Number of grid points along one dimension (total size = steps²).
    h_inv_sq : float
        Inverse squared grid spacing (1/h²).

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse CSR matrix representing L_M = Laplacian + Potential.
    """
    # Steps = dim_sqrt since we discretize in a (steps, steps) field
    L_N = get_L(steps, h_inv_sq)
    M_V = get_M(bound, steps)

    return L_N + M_V

def get_A_squiddle(v, beta, h_inv_sq):
    """
    Construct the nonlinear interaction term for the Gross–Pitaevskii equation.

    The operator corresponds to the diagonal matrix
        Ã = (β / h²) * diag(|v|²),
    where v is the current wavefunction approximation.  
    This models the mean-field particle repulsion (nonlinearity).

    Parameters
    ----------
    v : ndarray
        Current state vector (wavefunction values on the grid).
    beta : float
        Nonlinearity parameter (interaction strength).
    h_inv_sq : float
        Inverse squared grid spacing (1/h²).

    Returns
    -------
    scipy.sparse.dia_matrix
        Sparse diagonal matrix representing the nonlinear interaction term.
    """
    D = diags(np.multiply(v, v))
    return h_inv_sq * beta * D

def A_v(L_M, v, beta, h_inv_sq):
    """
    Construct the full operator A(v) for the Gross–Pitaevskii equation.

    The operator is given by
        A(v) = L_M + (β / h²) * diag(|v|²),
    where L_M is the precomputed linear part (Laplace + potential),
    and the second term represents the nonlinear particle interaction.

    Parameters
    ----------
    L_M : scipy.sparse.spmatrix
        Precomputed linear operator (Laplace + potential).
    v : ndarray
        Current state vector (wavefunction values on the grid).
    beta : float
        Nonlinearity parameter (interaction strength).
    h_inv_sq : float
        Inverse squared grid spacing (1/h²).

    Returns
    -------
    scipy.sparse.spmatrix
        Sparse matrix representing the full operator A(v).
    """
    return L_M + get_A_squiddle(v, beta, h_inv_sq)
