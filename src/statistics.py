import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

from src.ising_kernel import IsingKernel


@dataclass(frozen=True)
class HessianFisherResult:
    """Container for observed-information diagnostics."""
    hessian: np.ndarray          # Observed information: -∇² log L
    fisher: np.ndarray           # Empirical Fisher: Σ s_t s_t^T
    eigvals_hessian: np.ndarray
    eigvals_fisher: np.ndarray
    effective_rank_hessian: int
    effective_rank_fisher: int
    condition_hessian: float
    condition_fisher: float


def _loglik_trajectory(kernel: IsingKernel, trajectory: np.ndarray, J: float, h: float, epsilon: float) -> float:
    """
    Exact trajectory log-likelihood under P_epsilon using enumerated 2-component kernel.
    """
    if trajectory.ndim != 2 or trajectory.shape[1] != 2:
        raise ValueError(f"trajectory must have shape (T+1,2). got {trajectory.shape}")
    if trajectory.shape[0] < 2:
        raise ValueError("trajectory must contain at least two states.")

    ll = 0.0
    for t in range(trajectory.shape[0] - 1):
        I = (int(trajectory[t, 0]), int(trajectory[t, 1]))
        Ip = (int(trajectory[t + 1, 0]), int(trajectory[t + 1, 1]))
        p = kernel.compute_Peps(Ip, I, J=J, h=h, epsilon=epsilon)
        ll += np.log(max(p, 1e-300))
    return float(ll)


def _step_loglik(kernel: IsingKernel, I: Tuple[int, int], Ip: Tuple[int, int], J: float, h: float, epsilon: float) -> float:
    """Single-step log P_epsilon(Ip|I)."""
    p = kernel.compute_Peps(Ip, I, J=J, h=h, epsilon=epsilon)
    return float(np.log(max(p, 1e-300)))


def _finite_diff_gradient(f, x: np.ndarray, step: np.ndarray) -> np.ndarray:
    """Central-difference gradient."""
    g = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        dx = np.zeros_like(x)
        dx[i] = step[i]
        g[i] = (f(x + dx) - f(x - dx)) / (2.0 * step[i])
    return g


def _finite_diff_hessian(f, x: np.ndarray, step: np.ndarray) -> np.ndarray:
    """
    Central-difference Hessian (symmetric).
    Uses mixed partials formula for off-diagonals.
    """
    d = len(x)
    H = np.zeros((d, d), dtype=float)

    f0 = f(x)

    # Diagonal second derivatives
    for i in range(d):
        dx = np.zeros_like(x)
        dx[i] = step[i]
        H[i, i] = (f(x + dx) - 2.0 * f0 + f(x - dx)) / (step[i] ** 2)

    # Off-diagonals mixed partials
    for i in range(d):
        for j in range(i + 1, d):
            dxi = np.zeros_like(x)
            dxj = np.zeros_like(x)
            dxi[i] = step[i]
            dxj[j] = step[j]

            f_pp = f(x + dxi + dxj)
            f_pm = f(x + dxi - dxj)
            f_mp = f(x - dxi + dxj)
            f_mm = f(x - dxi - dxj)

            H_ij = (f_pp - f_pm - f_mp + f_mm) / (4.0 * step[i] * step[j])
            H[i, j] = H_ij
            H[j, i] = H_ij

    return H


def effective_rank_from_eigs(eigs: np.ndarray, tau_rel: float = 1e-8) -> int:
    """
    Effective rank: #{i: lambda_i / lambda_max > tau_rel}.
    Assumes eigs are real (symmetric matrices).
    """
    lam = np.sort(np.real(eigs))[::-1]
    if lam.size == 0:
        return 0
    lam_max = lam[0]
    if lam_max <= 0:
        return 0
    return int(np.sum((lam / lam_max) > tau_rel))


def condition_number_from_eigs(eigs: np.ndarray, tau_rel: float = 1e-8) -> float:
    """
    Condition number computed using smallest eigenvalue above tolerance.
    """
    lam = np.sort(np.real(eigs))[::-1]
    lam = lam[lam > 0]
    if lam.size == 0:
        return float("inf")
    lam_max = lam[0]
    lam_valid = lam[(lam / lam_max) > tau_rel]
    if lam_valid.size == 0:
        return float("inf")
    lam_min = lam_valid[-1]
    return float(lam_max / lam_min)


def matrix_rank_symmetric(M: np.ndarray, tau_rel: float = 1e-8) -> int:
    """Rank via eigenvalue thresholding relative to max eigenvalue magnitude."""
    eigs = np.linalg.eigvalsh(0.5 * (M + M.T))
    lam = np.sort(np.abs(eigs))[::-1]
    if lam.size == 0:
        return 0
    lam_max = lam[0]
    if lam_max == 0:
        return 0
    return int(np.sum((lam / lam_max) > tau_rel))


def covariance_from_hessian(H_obs: np.ndarray, ridge: float = 1e-8) -> np.ndarray:
    """
    Approximate covariance via inverse observed information (regularized).
    """
    Hs = 0.5 * (H_obs + H_obs.T)
    Hreg = Hs + ridge * np.eye(Hs.shape[0])
    try:
        return np.linalg.inv(Hreg)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(Hreg)


def correlation_matrix_from_cov(Cov: np.ndarray) -> np.ndarray:
    """Convert covariance to correlation matrix safely."""
    d = Cov.shape[0]
    Corr = np.zeros((d, d), dtype=float)
    diag = np.diag(Cov)
    for i in range(d):
        for j in range(d):
            denom = np.sqrt(max(diag[i], 0.0) * max(diag[j], 0.0))
            Corr[i, j] = Cov[i, j] / denom if denom > 0 else 0.0
    return Corr


def compute_hessian_fisher(
    trajectory: np.ndarray,
    J: float,
    h: float,
    epsilon: float,
    step: Optional[np.ndarray] = None,
    tau_rel: float = 1e-8,
) -> HessianFisherResult:
    """
    Compute observed Hessian (information) and empirical Fisher for psi=(J,h,epsilon).

    - Hessian: H_obs = -∇² log L
    - Fisher:  J_emp = Σ_t s_t s_t^T, where s_t = ∇ log P_eps(I_{t+1}|I_t)

    Finite differences are used for scores and curvature.
    """
    kernel = IsingKernel()
    x0 = np.array([J, h, epsilon], dtype=float)

    if step is None:
        # Parameter-scaled steps (safe defaults for Phase I)
        step = np.array([
            1e-4 * max(1.0, abs(J)),
            1e-4 * max(1.0, abs(h)),
            1e-4 * max(1.0, abs(epsilon) + 0.1),
        ], dtype=float)

    # Total log-likelihood function
    def f_total(x: np.ndarray) -> float:
        return _loglik_trajectory(kernel, trajectory, J=float(x[0]), h=float(x[1]), epsilon=float(x[2]))

    # Observed information: - Hessian(logL)
    H_logL = _finite_diff_hessian(f_total, x0, step)
    H_obs = -0.5 * (H_logL + H_logL.T)

    # Empirical Fisher: sum over per-step scores
    T = trajectory.shape[0] - 1
    scores = np.zeros((T, 3), dtype=float)

    for t in range(T):
        I = (int(trajectory[t, 0]), int(trajectory[t, 1]))
        Ip = (int(trajectory[t + 1, 0]), int(trajectory[t + 1, 1]))

        def f_step(x: np.ndarray) -> float:
            return _step_loglik(kernel, I, Ip, J=float(x[0]), h=float(x[1]), epsilon=float(x[2]))

        scores[t, :] = _finite_diff_gradient(f_step, x0, step)

    fisher = scores.T @ scores
    fisher = 0.5 * (fisher + fisher.T)

    eig_h = np.linalg.eigvalsh(H_obs)
    eig_f = np.linalg.eigvalsh(fisher)

    r_h = effective_rank_from_eigs(eig_h, tau_rel=tau_rel)
    r_f = effective_rank_from_eigs(eig_f, tau_rel=tau_rel)

    k_h = condition_number_from_eigs(eig_h, tau_rel=tau_rel)
    k_f = condition_number_from_eigs(eig_f, tau_rel=tau_rel)

    return HessianFisherResult(
        hessian=H_obs,
        fisher=fisher,
        eigvals_hessian=eig_h,
        eigvals_fisher=eig_f,
        effective_rank_hessian=r_h,
        effective_rank_fisher=r_f,
        condition_hessian=k_h,
        condition_fisher=k_f,
    )