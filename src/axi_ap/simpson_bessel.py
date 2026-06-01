import numpy as np
from scipy.special import jv


def _mixed_bessel_I(
    k_rho: np.ndarray,
    u0n: np.ndarray,
    b: float,
    tol_rel: float = 1e-10,
    tol_abs: float = 1e-12,
) -> np.ndarray:
    """
    Analytic mixed-Bessel integral for J0–J0 basis functions with robust
    handling near the removable singularity kρ b ≈ u_n.

    Computes:
        I00(kρ, n) = ∫₀ᵇ r J₀(kρ r) J₀(u₀ₙ r/b) dr
                   = -(u_n b^2 J1(u_n) J0(kρ b)) / ( (kρ b)^2 - u_n^2 )
    with the limit at (kρ b) -> u_n:
        I00 -> (b^2 / 2) [J1(u_n)]^2
    """
    k_rho = np.atleast_1d(k_rho).astype(float)  # (K,)
    u0n = np.atleast_1d(u0n).astype(float)  # (N,)

    beta = (k_rho * b)[:, None]  # (K,1)
    u = u0n[None, :]  # (1,N)

    j1_u = jv(1, u0n)[None, :]  # (1,N)
    j0_b = jv(0, (k_rho * b))[:, None]  # (K,1)

    denom = beta**2 - u**2  # (K,N)
    numer = -(u * b**2 * j1_u) * j0_b  # (K,N)

    I = numer / denom

    # Robust removable singularity handling near beta ~= u
    delta = np.abs(beta - u)  # (K,N)
    scale = np.abs(beta) + np.abs(u)  # (K,N)
    thresh = np.maximum(tol_abs, tol_rel * scale)  # (K,N)
    mask = delta <= thresh

    if np.any(mask):
        # Limit value depends only on column (n), broadcast rows
        limit_vals = (b**2 / 2.0) * (j1_u**2)  # (1,N)
        I[mask] = np.broadcast_to(limit_vals, I.shape)[mask]

    return I


def _simpson_weights_uniform(n: int, a: float, b: float) -> np.ndarray:
    """
    Compute uniform-grid Simpson weights for numerical integration.

    Args:
        n (int): Number of grid points (will be made odd if even).
        a (float): Start of interval.
        b (float): End of interval.

    Returns:
        np.ndarray: Array of Simpson weights for the interval.
    """
    n = _ensure_odd(n)
    h = (b - a) / (n - 1)
    w = np.ones(n, float)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    w *= h / 3.0
    return w


def _ensure_odd(n: int) -> int:
    return n if (n % 2 == 1) else (n + 1)
