"""
Refactored parameters class using dataclasses.
Provides physical constants and integration settings for the axisymmetric trap library.
"""

from dataclasses import dataclass, field, replace
import math
import numpy as np
from typing import Any, Dict, Optional, Tuple
from scipy import constants


@dataclass
class params:
    """
    Parameter container for physical and integration settings.

    - units: "SI" for standard units, "CGS" for normalized (1.0) values.
    - k: wavenumber
    - b: aperture radius
    - n: number of modes/coefficients
    - w0: beam waist
    - alpha: dynamic polarizability
    - N_theta, N_u: integration grid sizes
    - include_e: whether to include evanescent waves (basis dependent)
    - phase_sign: "plus" or "minus" for propagator phase
    """

    units: str = "SI"
    k: Optional[float] = None
    b: Optional[float] = None
    n: Optional[int] = None
    w0: Optional[float] = None
    alpha: Optional[complex] = None
    N_theta: int = 7001
    N_u: int = 3001
    include_e: bool = True
    phase_sign: str = "plus"
    u_max: Optional[float] = None
    N_theta_max: Optional[int] = None

    # Physical constants (calculated in __post_init__)
    c: float = field(init=False)
    epsilon_0: float = field(init=False)
    mu_0: float = field(init=False)
    Z0: float = field(init=False)

    def __post_init__(self):
        if self.units == "SI":
            self.c = constants.c
            self.epsilon_0 = constants.epsilon_0
            self.mu_0 = constants.mu_0
        elif self.units == "CGS":
            self.c = 1.0
            self.epsilon_0 = 1.0
            self.mu_0 = 1.0
        else:
            raise ValueError("units must be 'SI' or 'CGS'")
        self.Z0 = float(np.sqrt(self.mu_0 / self.epsilon_0))

    def _need(self, *names: str) -> None:
        for n in names:
            if getattr(self, n) is None:
                raise ValueError(f"params.{n} must be set first.")

    @property
    def zR(self) -> float:
        self._need("k", "w0")
        return 0.5 * self.k * self.w0**2

    @property
    def z_over_zR(self) -> Tuple[complex, complex]:
        self._need("k", "w0", "alpha")
        zr = self.zR
        real_alpha = float(np.real(self.alpha))
        imag_alpha = float(np.imag(self.alpha))
        if imag_alpha == 0:
            raise ZeroDivisionError(
                "alpha imaginary part is zero; cannot compute z_over_zR"
            )

        A = real_alpha / (self.k * zr * imag_alpha)
        B = math.sqrt(max(0.0, A**2 - 4 * (1 - 1 / (self.k * zr))))
        return (-A + B) / 2, (-A - B) / 2

    @property
    def trapping_condition(self) -> bool:
        self._need("k", "w0", "alpha")
        zr = self.zR
        lhs = abs(np.real(self.alpha) / np.imag(self.alpha))
        rhs = math.sqrt(4 * (self.k * zr) ** 2 * (1 - 1 / (self.k * zr)))
        return lhs < rhs

    def copy_with(self, **overrides: Any) -> "params":
        """
        Return a NEW params instance with provided overrides.
        """
        # Filter out constants if they were passed
        valid_overrides = {
            k: v
            for k, v in overrides.items()
            if k not in ("c", "epsilon_0", "mu_0", "Z0")
        }
        return replace(self, **valid_overrides)


def merge_params(base: Optional[params] = None, **overrides: Any) -> params:
    """
    Helper used inside operator functions.
    Ensures the base object is not mutated.
    """
    filtered = {k: v for k, v in overrides.items() if v is not None}
    if base is None:
        return params(**filtered)
    return base.copy_with(**filtered)
