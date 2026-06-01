from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional, List, Callable, Dict, Any
import numpy as np
from scipy.special import j0, j1
import warnings
from .simpson_bessel import _simpson_weights_uniform, _ensure_odd
from .parameters import params, merge_params

BasisTransform = Callable[[np.ndarray], np.ndarray]


@dataclass
class FieldOpsPack:
    Ex: np.ndarray
    Ey: np.ndarray
    Ez: np.ndarray
    Ex_x: np.ndarray
    Ex_y: np.ndarray
    Ex_z: np.ndarray
    Ey_x: np.ndarray
    Ey_y: np.ndarray
    Ey_z: np.ndarray
    Ez_x: np.ndarray
    Ez_y: np.ndarray
    Ez_z: np.ndarray
    Ex_xx: np.ndarray
    Ex_xy: np.ndarray
    Ex_xz: np.ndarray
    Ex_yy: np.ndarray
    Ex_yz: np.ndarray
    Ex_zz: np.ndarray
    Ey_xx: np.ndarray
    Ey_xy: np.ndarray
    Ey_xz: np.ndarray
    Ey_yy: np.ndarray
    Ey_yz: np.ndarray
    Ey_zz: np.ndarray
    Ez_xx: np.ndarray
    Ez_xy: np.ndarray
    Ez_xz: np.ndarray
    Ez_yy: np.ndarray
    Ez_yz: np.ndarray
    Ez_zz: np.ndarray

    @property
    def N(self) -> int:
        return int(self.Ex.shape[0])

    @property
    def shape(self) -> Tuple[()]:
        return ()


class _CachedSpectrum:
    def __init__(self):
        self.spec_key: Optional[Tuple] = None
        self.spec_data: Dict[str, Any] = {}
        self.j_key: Optional[Tuple] = None
        self.j_data: Dict[str, Any] = {}

    def ensure_spectrum(
        self,
        *,
        k: float,
        include_e: bool,
        N_theta: int,
        N_u: int,
        u_max: Optional[float],
        basis_transform: BasisTransform,
    ):
        key = (
            float(k),
            bool(include_e),
            int(N_theta),
            int(N_u),
            None if u_max is None else float(u_max),
        )
        if key == self.spec_key:
            return

        theta = np.linspace(0.0, np.pi / 2, N_theta)
        w_th = _simpson_weights_uniform(N_theta, theta[0], theta[-1])
        sin_th = np.sin(theta)
        cos_th = np.cos(theta)
        k_rho_p = k * sin_th
        kz_p = k * cos_th
        I_p = basis_transform(k_rho_p)

        spec = {
            "theta": theta,
            "w_th": w_th,
            "sin_th": sin_th,
            "cos_th": cos_th,
            "k_rho_p": k_rho_p,
            "kz_p": kz_p,
            "I_p": I_p,
        }

        if include_e:
            if u_max is None:
                raise ValueError("u_max must be provided to cache evanescent sector.")
            u = np.linspace(0.0, u_max, N_u)
            w_u = _simpson_weights_uniform(N_u, u[0], u[-1])
            sh = np.sinh(u)
            ch = np.cosh(u)
            k_rho_e = k * ch
            I_e = basis_transform(k_rho_e)
            spec.update(
                {"u": u, "w_u": w_u, "sh": sh, "ch": ch, "k_rho_e": k_rho_e, "I_e": I_e}
            )

        self.spec_key = key
        self.spec_data = spec
        self.j_key = None
        self.j_data = {}

    def ensure_J_rho(self, rho_flat: np.ndarray):
        rho_key = (self.spec_key, rho_flat.shape, float(np.sum(rho_flat)))
        if rho_key == self.j_key:
            return
        spec = self.spec_data
        k_rho_p = spec["k_rho_p"]

        J0_p = j0(np.outer(rho_flat, k_rho_p))
        J1_p = j1(np.outer(rho_flat, k_rho_p))
        j_data = {"J0_p": J0_p, "J1_p": J1_p}

        if "k_rho_e" in spec:
            k_rho_e = spec["k_rho_e"]
            J0_e = j0(np.outer(rho_flat, k_rho_e))
            J1_e = j1(np.outer(rho_flat, k_rho_e))
            j_data.update({"J0_e": J0_e, "J1_e": J1_e})

        self.j_key = rho_key
        self.j_data = j_data


class Operators:
    """
    Vector diffraction operators for axisymmetric apertures.

    Computes electromagnetic field components and their derivatives using
    basis expansion (e.g., Bessel modes). Supports both single-aperture
    and multi-aperture configurations.

    Usage:
        aperture = bessel_aperture(params, num_nodes=150)
        ops = Operators(aperture)  # From aperture object

        pack = ops.build(x=0, y=0, z=z, params=params)
        F, H = ops.force_hessian_operators(pack, alpha=alpha, eps0=eps0)
    """

    def __init__(self, basis_transform=None, b: float = None, k: float = None):
        # Support aperture object shorthand: Operators(aperture)
        if (
            basis_transform is not None
            and not callable(basis_transform)
            and b is None
            and k is None
        ):
            aperture = basis_transform
            if (
                not hasattr(aperture, "basis_transform")
                or not hasattr(aperture, "b")
                or not hasattr(aperture, "params")
            ):
                raise TypeError(
                    "If passing a single argument, it must be an aperture object with "
                    "basis_transform, b, and params attributes."
                )
            self.basis_transform = aperture.basis_transform
            self.b = float(aperture.b)
            self.k = float(aperture.params.k)
        else:
            # Traditional explicit arguments
            if not callable(basis_transform):
                raise TypeError(
                    "Operators requires a callable basis_transform(k_rho) -> I(k_rho, :)."
                )
            self.basis_transform = basis_transform
            self.b = float(b)
            self.k = float(k)
        self._cache = _CachedSpectrum()

    def _resolve_params(
        self,
        x,
        y,
        z,
        *,
        params: Optional[params],
        N_theta: Optional[int],
        N_u: Optional[int],
        u_max: Optional[float],
        include_e: Optional[bool],
        phase_sign: Optional[str],
    ):
        call_params = merge_params(
            params,
            k=self.k,
            N_theta=N_theta,
            N_u=N_u,
            u_max=u_max,
            include_e=include_e,
            phase_sign=phase_sign,
        )
        N_theta = call_params.N_theta
        N_u = call_params.N_u
        include_e = call_params.include_e
        phase_sign = call_params.phase_sign
        N_theta_max = getattr(call_params, "N_theta_max", 20001)
        N_theta_max_val = int(N_theta_max) if N_theta_max is not None else 20001
        u_max = getattr(call_params, "u_max", None)

        Xb, Yb, Zb = np.broadcast_arrays(
            np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)
        )
        x_flat = Xb.ravel()
        y_flat = Yb.ravel()
        z_flat = Zb.ravel()
        rho_flat = np.hypot(x_flat, y_flat)
        cos_phi = np.divide(
            x_flat, rho_flat, out=np.zeros_like(x_flat), where=rho_flat != 0.0
        )[:, None]

        z_max = float(np.max(np.abs(z_flat))) if z_flat.size else 0.0
        if z_max > 0:
            N_theta_target = int(6.0 * self.k * z_max + 50)
            if N_theta < N_theta_target:
                N_theta = _ensure_odd(min(N_theta_target, N_theta_max_val))

        return (
            N_theta,
            N_u,
            include_e,
            phase_sign,
            u_max,
            x_flat,
            y_flat,
            z_flat,
            rho_flat,
            cos_phi,
        )

    def _prepare_spectral_data(
        self,
        x,
        y,
        z,
        *,
        params: Optional[params] = None,
        N_theta: Optional[int] = None,
        N_u: Optional[int] = None,
        u_max: Optional[float] = None,
        include_e: Optional[bool] = None,
        phase_sign: Optional[str] = None,
    ):
        (
            N_theta,
            N_u,
            include_e,
            phase_sign,
            u_max,
            x_flat,
            y_flat,
            z_flat,
            rho_flat,
            cos_phi,
        ) = self._resolve_params(
            x,
            y,
            z,
            params=params,
            N_theta=N_theta,
            N_u=N_u,
            u_max=u_max,
            include_e=include_e,
            phase_sign=phase_sign,
        )
        sigma = +1 if phase_sign == "plus" else -1
        k = self.k

        if include_e and (u_max is None):
            tol_e = 1e-12
            z_abs = np.abs(z_flat)
            z_pos = z_abs[z_abs > 0]
            if z_pos.size > 0:
                z_min_pos = float(np.min(z_pos))
                target = np.log(1.0 / tol_e) / (k * z_min_pos)
                u_max = float(np.arcsinh(target) + 0.5)
                u_max = max(u_max, 4.0)
            else:
                u_max = 8.0

        self._cache.ensure_spectrum(
            k=k,
            include_e=bool(include_e),
            N_theta=int(N_theta),
            N_u=int(N_u),
            u_max=None if not include_e else float(u_max),
            basis_transform=self.basis_transform,
        )
        self._cache.ensure_J_rho(rho_flat)

        return (
            self._cache.spec_data,
            self._cache.j_data,
            sigma,
            x_flat,
            y_flat,
            z_flat,
            rho_flat,
            cos_phi,
        )

    def E_op(
        self,
        x,
        y,
        z,
        *,
        params: Optional[params] = None,
        N_theta: Optional[int] = None,
        N_u: Optional[int] = None,
        u_max: Optional[float] = None,
        include_e: Optional[bool] = None,
        phase_sign: Optional[str] = None,
    ) -> np.ndarray:
        spec, J, sigma, _, _, z_flat, _, cos_phi = self._prepare_spectral_data(
            x,
            y,
            z,
            params=params,
            N_theta=N_theta,
            N_u=N_u,
            u_max=u_max,
            include_e=include_e,
            phase_sign=phase_sign,
        )
        k = self.k
        include_e = bool("sh" in spec)

        theta = spec["theta"]
        w_th = spec["w_th"]
        sin_th = spec["sin_th"]
        cos_th = spec["cos_th"]
        kz_p = spec["kz_p"]
        I_p = spec["I_p"]
        J0_p = J["J0_p"]
        J1_p = J["J1_p"]

        phase_p = np.exp(1j * sigma * (z_flat[:, None] * kz_p[None, :]))

        W_ex_p = (k**2) * sin_th * cos_th * w_th
        Ex_op = (J0_p * (phase_p * W_ex_p[None, :])) @ I_p

        W_ez_p = (-1j * sigma) * (k**2) * (sin_th**2) * w_th
        Ez_op = (J1_p * (phase_p * W_ez_p[None, :])) @ I_p
 
        if include_e:
            sh = spec["sh"]
            ch = spec["ch"]
            w_u = spec["w_u"]
            I_e = spec["I_e"]
            J0_e = J["J0_e"]
            J1_e = J["J1_e"]
            phase_e = np.exp(-k * (z_flat[:, None] * sh[None, :]))
            W_ex_e = (-1j) * (k**2) * sh * ch * w_u
            Ex_op += (J0_e * (phase_e * W_ex_e[None, :])) @ I_e
            W_ez_e = (1j) * (k**2) * (ch**2) * w_u
            Ez_op += (J1_e * (phase_e * W_ez_e[None, :])) @ I_e

        Ey_op = np.zeros_like(Ex_op)
        Ez_op = cos_phi * Ez_op

        return np.stack([Ex_op, Ey_op, Ez_op], axis=0)

    def get_E_total_op(
        self,
        Xf: np.ndarray,
        Yf: np.ndarray,
        Zf: np.ndarray,
        positions: List[np.ndarray],
        polarizations: Optional[List[np.ndarray]],
        op_kw: Optional[dict] = None,
    ) -> np.ndarray:
        Xf = Xf.ravel()
        Yf = Yf.ravel()
        Zf = Zf.ravel()

        if op_kw is None:
            op_kw = {}
        apertures = []

        use_default_pol = polarizations is None
        if use_default_pol:
            warnings.warn(
                "No 'polarizations' provided. Defaulting to an orientation "
                "where the local x-axis is derived from the global y-axis.",
                UserWarning,
            )

        for i, pos in enumerate(positions):
            t_i = pos
            t_i_norm = np.linalg.norm(t_i)
            if np.isclose(t_i_norm, 0.0):
                raise ValueError(
                    f"Aperture {i} position cannot be at the origin (0,0,0)."
                )
            z_prime = -t_i / t_i_norm

            if use_default_pol:
                ref_vec = np.array([0.0, 1.0, 0.0])
                if np.abs(np.dot(z_prime, ref_vec)) > 0.999:
                    ref_vec = np.array([1.0, 0.0, 0.0])
                x_prime = np.cross(ref_vec, z_prime)
                x_prime = x_prime / np.linalg.norm(x_prime)
            else:
                pol_vec = polarizations[i]
                pol_mag = np.linalg.norm(pol_vec)
                if np.isclose(pol_mag, 0.0):
                    raise ValueError(
                        f"Aperture {i} polarization vector cannot be zero."
                    )
                dot_prod = np.dot(z_prime, pol_vec)
                x_prime = pol_vec - dot_prod * z_prime
                x_prime_mag = np.linalg.norm(x_prime)
                if not np.isclose(dot_prod, 0.0):
                    warnings.warn(
                        f"Aperture {i} polarization was not orthogonal to its normal "
                        f"(dot product = {dot_prod:.2e}). "
                        "Projecting to the closest orthogonal vector.",
                        UserWarning,
                    )
                if np.isclose(x_prime_mag, 0.0):
                    raise ValueError(
                        f"Aperture {i} polarization is parallel to its normal."
                    )
                x_prime = x_prime / x_prime_mag

            y_prime = np.cross(z_prime, x_prime)
            R_i_matrix = np.stack([x_prime, y_prime, z_prime], axis=-1)
            R_i_inv_matrix = R_i_matrix.T
            apertures.append({"t_i": t_i, "R_i": R_i_matrix, "R_i_inv": R_i_inv_matrix})

        R_global = np.stack([Xf, Yf, Zf], axis=-1)

        E_global_list = []
        for ap in apertures:
            R_relative = R_global - ap["t_i"][None, :]
            R_local = ap["R_i_inv"] @ R_relative.T
            X_local = R_local[0]
            Y_local = R_local[1]
            Z_local = R_local[2]
            E_local_op = self.E_op(X_local, Y_local, Z_local, **op_kw)
            E_global_op = np.einsum("ij,jrn->irn", ap["R_i"], E_local_op)
            E_global_list.append(E_global_op)

        A_total = np.concatenate(E_global_list, axis=2)
        return A_total

    def _E_point(self, x: float, y: float, z: float, **kwargs) -> np.ndarray:
        A = self.E_op(x, y, z, **kwargs)
        return A[:, 0, :]

    def build(
        self,
        x: float,
        y: float,
        z: float,
        *,
        params: Optional[params] = None,
        positions: Optional[List[np.ndarray]] = None,
        polarizations: Optional[List[np.ndarray]] = None,
        N_theta: Optional[int] = None,
        N_u: Optional[int] = None,
        u_max: Optional[float] = None,
        include_e: Optional[bool] = None,
        phase_sign: Optional[str] = None,
        hx: Optional[float] = 1e-5,
        hy: Optional[float] = 1e-5,
        hz: Optional[float] = 1e-5,
    ) -> FieldOpsPack:
        call_params = merge_params(
            params,
            k=self.k,
            N_theta=N_theta,
            N_u=N_u,
            u_max=u_max,
            include_e=include_e,
            phase_sign=phase_sign,
        )
        N_theta = call_params.N_theta
        N_u = call_params.N_u
        include_e = call_params.include_e
        phase_sign = call_params.phase_sign
        u_max = getattr(call_params, "u_max", None)

        def _u_max_shared(
            z0: float,
            k_val: float,
            hz_local: float,
            tol: float = 1e-12,
            pad: float = 0.5,
        ) -> float:
            z_candidates = [
                abs(z0 + s)
                for s in (0.0, hz_local, -hz_local, 2 * hz_local, -2 * hz_local)
            ]
            z_eff = max(max(z_candidates), 1e-9 / max(k_val, 1e-12))
            target = np.log(1.0 / tol) / (self.k * z_eff)
            return float(np.arcsinh(target) + pad)

        if include_e and (u_max is None):
            u_max_shared = _u_max_shared(z, self.k, hz)
        else:
            u_max_shared = u_max

        op_kw = dict(
            N_theta=N_theta,
            N_u=N_u,
            u_max=u_max_shared,
            include_e=include_e,
            phase_sign=phase_sign,
        )

        x_off = np.array([-2 * hx, -hx, 0.0, hx, 2 * hx], dtype=float)
        y_off = np.array([-2 * hy, -hy, 0.0, hy, 2 * hy], dtype=float)
        z_off = np.array([-2 * hz, -hz, 0.0, hz, 2 * hz], dtype=float)
        X_xy = np.broadcast_to(x + x_off[:, None], (5, 5))
        Y_xy = np.broadcast_to(y + y_off[None, :], (5, 5))
        Z_xy = np.full((5, 5), z, dtype=float)
        X_xz = np.broadcast_to(x + x_off[:, None], (5, 5))
        Y_xz = np.full((5, 5), y, dtype=float)
        Z_xz = np.broadcast_to(z + z_off[None, :], (5, 5))
        X_yz = np.full((5, 5), x, dtype=float)
        Y_yz = np.broadcast_to(y + y_off[:, None], (5, 5))
        Z_yz = np.broadcast_to(z + z_off[:, None], (5, 5))
        Xf = np.concatenate([X_xy.ravel(), X_xz.ravel(), X_yz.ravel()])
        Yf = np.concatenate([Y_xy.ravel(), Y_xz.ravel(), Y_yz.ravel()])
        Zf = np.concatenate([Z_xy.ravel(), Z_xz.ravel(), Z_yz.ravel()])
        assert Xf.size == 75 and Yf.size == 75 and Zf.size == 75

        if positions is None:
            if polarizations is not None:
                warnings.warn(
                    "'polarizations' is ignored when 'positions' is not set.",
                    UserWarning,
                )
            A = self.E_op(Xf, Yf, Zf, **op_kw)
        else:
            if polarizations is not None and (len(positions) != len(polarizations)):
                raise ValueError(
                    "'positions' and 'polarizations' must all have the same length."
                )
            A = self.get_E_total_op(Xf, Yf, Zf, positions, polarizations, op_kw)

        _, _, N_total = A.shape
        A_xy = A[:, 0:25, :].copy().reshape(3, 5, 5, N_total)
        A_xz = A[:, 25:50, :].copy().reshape(3, 5, 5, N_total)
        A_yz = A[:, 50:75, :].copy().reshape(3, 5, 5, N_total)

        cx = cy = cz = 2

        w1x = np.array([1, -8, 0, 8, -1], dtype=float) / (12.0 * hx)
        w1y = np.array([1, -8, 0, 8, -1], dtype=float) / (12.0 * hy)
        w1z = np.array([1, -8, 0, 8, -1], dtype=float) / (12.0 * hz)
        w2x = np.array([-1, 16, -30, 16, -1], dtype=float) / (12.0 * hx * hx)
        w2y = np.array([-1, 16, -30, 16, -1], dtype=float) / (12.0 * hy * hy)
        w2z = np.array([-1, 16, -30, 16, -1], dtype=float) / (12.0 * hz * hz)

        def d1x(Axy):
            return np.einsum("i,in->n", w1x, Axy[:, cy, :])

        def d1y(Axy):
            return np.einsum("j,jn->n", w1y, Axy[cx, :, :])

        def d1z(Axz):
            return np.einsum("k,kn->n", w1z, Axz[cx, :, :])

        def d2x(Axy):
            return np.einsum("i,in->n", w2x, Axy[:, cy, :])

        def d2y(Axy):
            return np.einsum("j,jn->n", w2y, Axy[cx, :, :])

        def d2z(Axz):
            return np.einsum("k,kn->n", w2z, Axz[cx, :, :])

        def dxy(Axy):
            Dy = np.einsum("j,ijn->in", w1y, Axy)
            return np.einsum("i,in->n", w1x, Dy)

        def dxz(Axz):
            Dz = np.einsum("k,ikn->in", w1z, Axz)
            return np.einsum("i,in->n", w1x, Dz)

        def dyz(Ayz):
            Dz = np.einsum("k,jkn->jn", w1z, Ayz)
            return np.einsum("j,jn->n", w1y, Dz)

        def comp_all(i):
            f_xy = A_xy[i, cx, cy, :]
            f_xz = A_xz[i, cx, cz, :]
            f_yz = A_yz[i, cy, cz, :]
            f = (f_xy + f_xz + f_yz) / 3.0
            fx = d1x(A_xy[i])
            fy = d1y(A_xy[i])
            fz = d1z(A_xz[i])
            fxx = d2x(A_xy[i])
            fyy = d2y(A_xy[i])
            fzz = d2z(A_xz[i])
            fxy = dxy(A_xy[i])
            fxz = dxz(A_xz[i])
            fyz = dyz(A_yz[i])
            return f, fx, fy, fz, fxx, fyy, fzz, fxy, fxz, fyz

        Ex, Ex_x, Ex_y, Ex_z, Ex_xx, Ex_yy, Ex_zz, Ex_xy, Ex_xz, Ex_yz = comp_all(0)
        Ey, Ey_x, Ey_y, Ey_z, Ey_xx, Ey_yy, Ey_zz, Ey_xy, Ey_xz, Ey_yz = comp_all(1)
        Ez, Ez_x, Ez_y, Ez_z, Ez_xx, Ez_yy, Ez_zz, Ez_xy, Ez_xz, Ez_yz = comp_all(2)

        return FieldOpsPack(
            Ex=Ex,
            Ey=Ey,
            Ez=Ez,
            Ex_x=Ex_x,
            Ex_y=Ex_y,
            Ex_z=Ex_z,
            Ey_x=Ey_x,
            Ey_y=Ey_y,
            Ey_z=Ey_z,
            Ez_x=Ez_x,
            Ez_y=Ez_y,
            Ez_z=Ez_z,
            Ex_xx=Ex_xx,
            Ex_xy=Ex_xy,
            Ex_xz=Ex_xz,
            Ex_yy=Ex_yy,
            Ex_yz=Ex_yz,
            Ex_zz=Ex_zz,
            Ey_xx=Ey_xx,
            Ey_xy=Ey_xy,
            Ey_xz=Ey_xz,
            Ey_yy=Ey_yy,
            Ey_yz=Ey_yz,
            Ey_zz=Ey_zz,
            Ez_xx=Ez_xx,
            Ez_xy=Ez_xy,
            Ez_xz=Ez_xz,
            Ez_yy=Ez_yy,
            Ez_yz=Ez_yz,
            Ez_zz=Ez_zz,
        )

    def force_hessian_operators(
        self,
        pack: FieldOpsPack,
        *,
        alpha: complex,
        eps0: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute force and stiffness (Hessian) operators for optical trapping.

        The force operator F[dir, i, j] represents the force on mode i due to
        mode j interaction. Stiffness H[p, q, i, j] represents the restoring
        force (spring constant) along directions p, q.

        Formula:
            F_dir = (eps0/2) * (α* ⟨E|∇_dir E⟩ + α ⟨∇_dir E|E⟩) / 2
            H_pq = (eps0/2) * (α* (⟨∇_q E|∇_p E⟩ + ⟨∇_pq E|E⟩) + h.c.) / 2

        Parameters
        ----------
        pack : FieldOpsPack
            Field and derivative operators at (x, y, z).
        alpha : complex
            Dynamic polarizability of the particle. Must be complex to support
            lossy particles. Real part drives radiation pressure; imaginary
            part drives thermal effects.
        eps0 : float
            Permittivity of free space (units must match basis_transform).

        Returns
        -------
        F : ndarray, shape (3, N, N)
            Force operators [Fx, Fy, Fz] for N basis modes.
        H : ndarray, shape (3, 3, N, N)
            Stiffness operators H[p, q, ...] for directions p, q ∈ {x, y, z}.
        """
        if not isinstance(alpha, (int, float, complex)):
            raise TypeError(f"alpha must be numeric, got {type(alpha)}")

        E = np.stack([pack.Ex, pack.Ey, pack.Ez], axis=0)
        dE = np.stack(
            [
                [pack.Ex_x, pack.Ex_y, pack.Ex_z],
                [pack.Ey_x, pack.Ey_y, pack.Ey_z],
                [pack.Ez_x, pack.Ez_y, pack.Ez_z],
            ],
            axis=0,
        )
        d2E = np.stack(
            [
                [
                    [pack.Ex_xx, pack.Ex_xy, pack.Ex_xz],
                    [pack.Ex_xy, pack.Ex_yy, pack.Ex_yz],
                    [pack.Ex_xz, pack.Ex_yz, pack.Ex_zz],
                ],
                [
                    [pack.Ey_xx, pack.Ey_xy, pack.Ey_xz],
                    [pack.Ey_xy, pack.Ey_yy, pack.Ey_yz],
                    [pack.Ey_xz, pack.Ey_yz, pack.Ey_zz],
                ],
                [
                    [pack.Ez_xx, pack.Ez_xy, pack.Ez_xz],
                    [pack.Ez_xy, pack.Ez_yy, pack.Ez_yz],
                    [pack.Ez_xz, pack.Ez_yz, pack.Ez_zz],
                ],
            ],
            axis=0,
        )

        prefac = eps0 / 2.0
        a, a_star = alpha, np.conjugate(alpha)

        Qdir_all = np.einsum("ci, cdj -> dij", np.conjugate(E), dE)
        F = (
            prefac
            * (a_star * Qdir_all + a * np.conj(np.swapaxes(Qdir_all, 1, 2)))
            / 2.0
        )

        term1 = np.einsum("cqi, cpj -> pqij", np.conjugate(dE), dE)
        term2 = np.einsum("cpqi, cj -> pqij", np.conjugate(d2E), E)
        Spq_all = term1 + term2
        H = prefac * (a_star * Spq_all + a * np.conj(np.swapaxes(Spq_all, 2, 3))) / 2.0

        return F.copy(), H.copy()

    def H_op(
        self,
        x,
        y,
        z,
        *,
        params: Optional[params] = None,
        N_theta: Optional[int] = None,
        N_u: Optional[int] = None,
        u_max: Optional[float] = None,
        include_e: Optional[bool] = None,
        phase_sign: Optional[str] = None,
    ) -> np.ndarray:
        """Exact analytical magnetic field reconstruction."""
        from scipy.special import jv

        spec, J, sigma, _, y_flat, z_flat, rho_flat, cos_phi = (
            self._prepare_spectral_data(
                x,
                y,
                z,
                params=params,
                N_theta=N_theta,
                N_u=N_u,
                u_max=u_max,
                include_e=include_e,
                phase_sign=phase_sign,
            )
        )
        k = self.k
        Z0 = getattr(params, "Z0", 376.730313668)
        prefactor = 1.0 / (k * Z0)
        include_e = bool("sh" in spec)

        sin_phi = np.divide(
            y_flat, rho_flat, out=np.zeros_like(y_flat), where=rho_flat != 0.0
        )[:, None]
        cos_2phi = cos_phi**2 - sin_phi**2
        sin_2phi = 2 * cos_phi * sin_phi

        theta, w_th, sin_th = spec["theta"], spec["w_th"], spec["sin_th"]
        kz_p, k_rho_p, I_p = spec["kz_p"], spec["k_rho_p"], spec["I_p"]
        J0_p, J1_p = J["J0_p"], J["J1_p"]
        J2_p = jv(2, np.outer(rho_flat, k_rho_p))

        phase_p = np.exp(1j * sigma * (z_flat[:, None] * kz_p[None, :]))
        measure_p = k**2 * sin_th * spec["cos_th"] * w_th
        measure_over_kz_p = k * sin_th * w_th  # Analytically cancels kz=0 singularity

        # Hx
        W_hx_p = prefactor * 0.5 * k_rho_p**2 * measure_over_kz_p
        Hx_op = sin_2phi * (J2_p * (phase_p * W_hx_p[None, :])) @ I_p

        # Hy
        W_hy_j0_p = prefactor * (k**2 - 0.5 * k_rho_p**2) * measure_over_kz_p
        W_hy_j2_p = -prefactor * 0.5 * k_rho_p**2 * measure_over_kz_p
        Hy_op = (J0_p * (phase_p * W_hy_j0_p[None, :])) @ I_p + cos_2phi * (
            J2_p * (phase_p * W_hy_j2_p[None, :])
        ) @ I_p

        # Hz
        W_hz_p = prefactor * (-1j * sigma * k_rho_p) * measure_p
        Hz_op = sin_phi * (J1_p * (phase_p * W_hz_p[None, :])) @ I_p

        if include_e:
            sh, ch, w_u = spec["sh"], spec["ch"], spec["w_u"]
            k_rho_e, I_e = spec["k_rho_e"], spec["I_e"]
            J0_e, J1_e = J["J0_e"], J["J1_e"]
            J2_e = jv(2, np.outer(rho_flat, k_rho_e))

            phase_e = np.exp(-k * (z_flat[:, None] * sh[None, :]))
            measure_e = k**2 * sh * ch * w_u
            measure_over_kz_e = 1j * k * ch * w_u

            W_hx_e = prefactor * 0.5 * k_rho_e**2 * measure_over_kz_e
            Hx_op += sin_2phi * (J2_e * (phase_e * W_hx_e[None, :])) @ I_e

            W_hy_j0_e = prefactor * (k**2 - 0.5 * k_rho_e**2) * measure_over_kz_e
            W_hy_j2_e = -prefactor * 0.5 * k_rho_e**2 * measure_over_kz_e
            Hy_op += (J0_e * (phase_e * W_hy_j0_e[None, :])) @ I_e + cos_2phi * (
                J2_e * (phase_e * W_hy_j2_e[None, :])
            ) @ I_e

            W_hz_e = prefactor * (-1j * k_rho_e) * measure_e
            Hz_op += sin_phi * (J1_e * (phase_e * W_hz_e[None, :])) @ I_e

        return np.stack([Hx_op, Hy_op, Hz_op], axis=0)

    def get_energy_matrix(self, params=None, include_e=True):
        """
        Constructs the (N, N) operator matrix for total electromagnetic energy.
        Evaluate in your script via: W = np.real(np.vdot(coeffs, W_mat @ coeffs))
        """
        if params is None:
            raise ValueError("params object containing epsilon_0 must be provided.")

        eps0 = params.epsilon_0
        k = self.k

        spec, _, _, _, _, _, _, _ = self._prepare_spectral_data(
            0.0, 0.0, 0.0, params=params, include_e=include_e
        )

        # ---------------------------------------------------------
        # 1. Propagating Operator Matrix
        # ---------------------------------------------------------
        w_th = spec["w_th"]
        sin_th = spec["sin_th"]
        cos_th = np.maximum(spec["cos_th"], 1e-12)
        I_p = spec["I_p"]

        # Diagonal weighting array
        D_p = (
            2
            * np.pi
            * eps0
            * ((1 + cos_th**2) / cos_th**2)
            * (k**2 * sin_th * cos_th)
            * w_th
        )
        D_p[spec["cos_th"] <= 1e-10] = 0.0
        # D_p = 0.0 * D_p
        W_mat = (np.conjugate(I_p).T * D_p) @ I_p

        # ---------------------------------------------------------
        # 2. Add Evanescent Operator Matrix (if requested)
        # ---------------------------------------------------------
        if include_e and "u" in spec:
            w_u = spec["w_u"]
            ch = spec["ch"]
            sh = np.maximum(spec["sh"], 1e-12)
            I_e = spec["I_e"]

            W_E_factor = (1 + 3 * sh**2) / sh**2
            W_H_factor = (ch**2 + 2 * sh**4) / sh**2

            D_e = np.pi * eps0 * (W_E_factor + W_H_factor) * (k**2 * sh * ch) * w_u
            D_e[spec["sh"] <= 1e-10] = 0.0
            W_mat += (np.conjugate(I_e).T * D_e) @ I_e

        return W_mat
