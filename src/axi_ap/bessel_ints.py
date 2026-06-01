import numpy as np
from scipy.special import j0, j1, jn_zeros
from scipy.integrate import simpson
from .paraxial_gauss import gauss_inc
from .base_aperture import BaseAperture


class bessel_aperture(BaseAperture):
    def __init__(
        self, params, func=gauss_inc, num_nodes: int = 150, num_rho: int = 2000
    ):
        super().__init__(params, func, num_nodes)

        self.alpha_n = jn_zeros(0, self.N)
        self.gamma_n = self.alpha_n / self.b

        num_rho = int(num_rho) if int(num_rho) % 2 != 0 else int(num_rho) + 1
        self.rho = np.linspace(0.0, self.b, num_rho)
        self.w_rho = np.ones_like(self.rho)

        self.basis = j0(np.outer(self.gamma_n, self.rho))

        f_rho = func(self.rho, self.w0)

        norm_n = (self.b**2 / 2.0) * j1(self.alpha_n) ** 2

        integrand = f_rho[None, :] * self.basis * self.rho[None, :]
        self.coeffs = simpson(integrand, x=self.rho, axis=1) / norm_n
        self.coeffs = self.coeffs.astype(np.complex128)

    def get_intensity_matrix(self) -> np.ndarray:
        """
        Analytical intensity integral matrix for the Bessel basis:
        I_nm = 2pi * ∫_0^b J0(γ_n ρ) J0(γ_m ρ) ρ dρ
             = 2pi * δ_nm * (b^2 / 2) * [J1(α_n)]^2
        """
        norm_n = (self.b**2 / 2.0) * j1(self.alpha_n) ** 2
        return np.diag(2.0 * np.pi * norm_n)

    def basis_transform(self, k_rho=None, **kwargs) -> np.ndarray:
        if k_rho is None:
            k = float(getattr(self.params, "k", 0.0))
            k_rho = self._generate_k_rho_grid(
                k=k,
                N_default=kwargs.get("N_default", 1025),
                include_evanescent=kwargs.get("include_evanescent", False),
                u_max=kwargs.get("u_max"),
                N_u=kwargs.get("N_u", 0),
            )

        k_rho = np.asarray(k_rho, float)
        K = len(k_rho)
        I = np.zeros((K, self.N), dtype=np.float64)

        beta = k_rho[:, None]
        gamma = self.gamma_n[None, :]

        denom = gamma**2 - beta**2
        num = self.alpha_n[None, :] * j0(beta * self.b) * j1(self.alpha_n)[None, :]

        with np.errstate(divide="ignore", invalid="ignore"):
            I = num / denom

        for n in range(self.N):
            match_idx = np.isclose(k_rho, self.gamma_n[n])
            if np.any(match_idx):
                I[match_idx, n] = (self.b**2 / 2.0) * j1(self.alpha_n[n]) ** 2

        return I

    def get_total_energy(self, ops, include_e: bool = True) -> dict:
        """
        Calculates the exact total electromagnetic energy W = WE + WH over the entire
        transverse plane using the continuous spectral Parseval-Plancherel theorem.

        Parameters
        ----------
        ops : Operators
            The initialized Operators object to provide the exact spectral grids.
        include_e : bool
            Whether to include the reactive evanescent energy in the calculation.

        Returns
        -------
        dict
            Dictionary containing 'W_prop', 'W_evan', and 'W_total'.
        """
        params = self.params
        k = params.k
        eps0 = params.epsilon_0

        # Fetch the exact grids from the operator
        spec, _, _, _, _, _, _, _ = ops._prepare_spectral_data(
            0.0, 0.0, 0.0, params=params, include_e=include_e
        )

        # ---------------------------------------------------------
        # 1. Propagating Energy (0 to pi/2)
        # ---------------------------------------------------------
        theta = spec["theta"]
        sin_th = spec["sin_th"]
        cos_th = np.maximum(spec["cos_th"], 1e-12)  # Cap the grazing singularity

        S_p = spec["I_p"] @ self.coeffs
        abs_S_p_sq = np.abs(S_p) ** 2

        # For propagating plane waves, WE and WH are exactly equal.
        # W_E_density = W_H_density = eps0 * pi * |S|^2 * ((1 + cos^2(theta)) / cos^2(theta))
        W_E_p_dens = eps0 * np.pi * abs_S_p_sq * ((1 + cos_th**2) / cos_th**2)

        # W_total = 2 * WE. Multiply by measure k^2 sin(th) cos(th)
        measure_p = k**2 * sin_th * cos_th
        W_prop_total = simpson(2 * W_E_p_dens * measure_p, x=theta)

        # ---------------------------------------------------------
        # 2. Evanescent Reactive Energy (0 to u_max)
        # ---------------------------------------------------------
        W_evan_total = 0.0
        if include_e and "u" in spec:
            u = spec["u"]
            ch = spec["ch"]
            sh = np.maximum(spec["sh"], 1e-12)

            S_e = spec["I_e"] @ self.coeffs
            abs_S_e_sq = np.abs(S_e) ** 2

            # Evanescent waves are reactive: WE != WH
            W_E_e_dens = eps0 * np.pi * abs_S_e_sq * ((1 + 3 * sh**2) / sh**2)
            W_H_e_dens = eps0 * np.pi * abs_S_e_sq * ((ch**2 + 2 * sh**4) / sh**2)

            measure_e = k**2 * sh * ch
            W_evan_total = simpson((W_E_e_dens + W_H_e_dens) * measure_e, x=u)

        return {
            "W_prop": W_prop_total,
            "W_evan": W_evan_total,
            "W_total": W_prop_total + W_evan_total,
        }
