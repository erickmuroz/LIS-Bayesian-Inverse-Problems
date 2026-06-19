# LIS for Bayesian Inverse Problems

Implementation of the **Likelihood-Informed Subspace (LIS)** method for linear
Bayesian inverse problems, following Spantini et al. (2015), with **Gaussian-process
surrogate modeling** for expensive forward operators (Villani et al., 2024).
Developed as part of my Bachelor's thesis at the Engineering Risk Analysis Group,
Technical University of Munich.

## Motivation

In high-dimensional Bayesian inverse problems the data typically inform only a
low-dimensional subspace of the parameter space. LIS identifies this subspace through
a generalized eigenvalue problem between the prior-preconditioned likelihood Hessian and
the prior precision, yielding optimal low-rank approximations of the posterior. Combining
LIS with a GP surrogate replaces costly forward evaluations and makes posterior inference
tractable.

## Examples

The same LIS machinery is validated across three forward operators:

| Folder | Forward model | Status |
|---|---|---|
| `01_linear_gaussian/` | Linear-Gaussian toy problem | done |
| `02_black_scholes/` | Black-Scholes local-volatility calibration (parabolic PDE) | active |
| `03_fem_bar/` | 1D bar FEM (stiffness-matrix formulation) | in progress |

## Method (`01_linear_gaussian/lis_by_hand.m`)

Self-contained implementation of the linear-Gaussian framework:

- Generalized eigenvalue problem (likelihood Hessian vs. prior precision)
- LIS basis construction
- Posterior covariance approximation (Theorem 2.3)
- Optimal oblique projector (Corollary 3.2) and projector verification
- Förstner / Frobenius error sweeps over the retained rank
- Eigenvalue spectrum with truncation threshold (`semilogy`)

## Black-Scholes example (`02_black_scholes/`)

Recovers a latent local-volatility field from noisy European option prices. The forward
operator maps a discretized volatility surface to model prices through a finite-difference
solver of the Black-Scholes PDE; the Jacobian is used to form the linearized Gaussian
posterior that feeds the LIS machinery.

Main script: `bs_lis_main.m`. Supporting: `bs_pricer.m`, `vega.m`, `fd_solver.m`,
`build_sigma.m`, `build_G.m`, `build_prior.m`.

## Requirements

MATLAB (R2022a or later). No external toolboxes required beyond base + standard
linear-algebra routines.

## References

- A. Spantini, A. Solonen, T. Cui, J. Martin, L. Tenorio, Y. Marzouk,
  *Optimal Low-Rank Approximations of Bayesian Linear Inverse Problems*,
  SIAM J. Sci. Comput., 2015.
- N. Villani et al., *Adaptive Gaussian Process Regression for Bayesian Inverse Problems*, 2024.

## Author

Erick Muro Zaldívar — TU Munich.
Supervised by Jakob Scheffels (Engineering Risk Analysis Group).
