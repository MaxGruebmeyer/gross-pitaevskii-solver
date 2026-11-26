# Gross-Pitaevskii Equation (GPE) Solver

This project implements and analyzes numerical methods for finding the ground
state solution of the two-dimensional **Gross-Pitaevskii Equation (GPE)**.
The GPE is a nonlinear partial differential equation (PDE) used to model the behavior
of Bose-Einstein condensates (BECs). This project reformulates the PDE into
a nonlinear eigenvalue problem, which is then solved using various iterative
algorithms which have been presented in [JKM12](https://arxiv.org/abs/1212.0417)
and further discussed in [Hen24](https://arxiv.org/abs/2202.07593).

## Motivation

The ground state of the GPE corresponds to the solution with the minimum energy.
Finding this solution is crucial for understanding the properties of BECs,
such as their density profile and stability. Direct iterative methods can be
unstable or slow to converge, especially for strong inter-particle interactions
(high nonlinearity). This project explores and compares three common inverse
iteration approaches to find a robust and efficient solution method.

## Numerical approach

As discussed in [Hen24](https://arxiv.org/abs/2202.07593) the continuous PDE
can be reformulated as an infinite-dimensional nonlinear eigenvalue problem
of the form:

$$
A(u)u = \lambda u
$$

where

- $u$ is the solution vector representing the wavefunction on the grid
- $A(u)$ is a nonlinear operator that depends on the solution itself
- $\lambda$ is the chemical potential (eigenvalue)

The eigenvalues $\lambda$ are all positive and the eigenfunction $u$ with the
minimal $\lambda$ corresponds to the solution with the minimum energy.

By discretizing this problem on a Cartesian grid using the **finite-difference
method** we can solve it using various iterative algorithms:

- **Inverse Iteration (A-Method)**: The generalized version of the inverse iteration
algorithm used in numerical linear algebra to find the smallest eigenvector
of an operator. It linearizes the problem in each step by just plugging in
$u_n$ into the function $A$.
- **Dampened Inverse Iteration**:
An enhanced method based on an adaptive dampening parameter $\tau_n$ which is
determined in each step via line search.
This ensures that the energy of the system
decreases with every iteration, guaranteeing convergence to a stable ground solution
provided a non-negative start vector and $\tau \in (0, 2)$.
Once the value is deemed to be in a neighbourhood of the optimal solution $\hat{u}$
$\tau_n$ is set to $1$ to achieve the asymptotically fastest convergence rate.
For $\tau_n = 1$ this method is the same as the inverse iteration algorithm.
- **Shifted Inverse Iteration**: Instead of using the Rayleigh qoutient it sues
Hen24's Def 5.1 with a fixed shift.

## Results and Analysis

The dampened and normal inverse iteration omit similar behaviour, with the dampened
version outperforming the standard iteration in all test cases. For the line
search of the dampened version I used `scipy.optimize.minimize_scalar` instead
of a custom line search algorithm. There might be some gains left on the table
here still.
The shifted iteration is a mixed bag. For $\beta \leq 1$ (small nonlinearity)
the shifted version converges in less than half of the iterations than the other
versions. For other values it does not work as well, which mathces Hen24's result.

## Project structure

- `gpe_algorithms.py`: Contains the implementations of the inverse iteration,
dampened inverse iteration, and shifted inverse iteration algorithms.
Also includes the energy functional calculation.
- `gpe_problem.py`: Defines the problem setup, including the discretization of the
2D Laplacian, the harmonic potential, and the construction of the nonlinear
operator $A(u)$.
- `main.ipynb`: A Jupyter Notebook for running the simulations, generating plots
and analyzing the results.

## How to run

1. Ensure you have the required Python packages: `numpy`, `scipy` and `matplotlib`
2. Run the main notebook to execute the algorithms and generate the convergence
plots.

## Possible Improvements

- There is a better alternative to the A-Method (called the J-Method) which uses the derivative
of the function (instead of just plugging $u$ into $A$) to better linearize
$A$ in each step. This could be implemented and compared to the A-Method.
Furthermore shifting with the J-Method is said to be highly effective (as opposed
to the A-Methods abysmal reaction to shifts). If you ever encounter such a problem
in the wild: Use the J-Method, not this. It's vastly superior.
- Replace `minimize_scalar` with custom line search algorithm to improve
performance of the dampened inverse iteration.
- Currently the program always uses a normalized $1$-Vector
as a starting point, which could be drawn from a distribution instead,
analyzing how the methods react to different starting vectors.
