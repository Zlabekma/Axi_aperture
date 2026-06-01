import numpy as np

class BaseAperture:
    """
    Shared base class for axisymmetric aperture basis expansions.
    Provides common initialization and grid generation logic.
    """
    def __init__(self, params, func=None, num_nodes=None):
        self.params = params
        self.func = func
        self.b = float(params.b)
        self.w0 = float(params.w0)
        if num_nodes is not None:
            self.N = int(num_nodes)

    @property
    def num_modes(self) -> int:
        return getattr(self, "N", 0)

    def _generate_k_rho_grid(self, k: float, N_default: int = 1025, include_evanescent: bool = False, u_max: float = None, N_u: int = 0) -> np.ndarray:
        if k <= 0.0:
            raise ValueError("k must be set (>0) to auto-generate k_rho.")
        Kp = int(N_default)
        k_rho_p = np.linspace(0.0, k, Kp)
        k_rho_list = [k_rho_p]
        if include_evanescent:
            if u_max is None:
                u_max = 8.0
            Nu = int(N_u) if N_u and int(N_u) > 0 else int(max(201, Kp // 2))
            u = np.linspace(0.0, float(u_max), Nu)
            k_rho_list.append(k * np.cosh(u))
        return np.concatenate(k_rho_list)
