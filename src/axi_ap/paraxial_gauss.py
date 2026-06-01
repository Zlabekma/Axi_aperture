import numpy as np
from scipy.integrate import quad

gauss_inc = lambda rho, omega: np.exp(-(rho**2) / omega**2)

gauss_inc.__doc__ = """
Gaussian incident field profile.

Parameters
----------
rho : float or ndarray
  Radial coordinate(s).
omega : float
  Beam waist parameter.

Returns
-------
ndarray
  Field amplitude at each rho.
"""


spectrum = lambda k_rho_arr, omega: (
    0.5 * np.exp(-0.25 * k_rho_arr**2 * omega**2) * omega**2
)

spectrum.__doc__ = """
Spectrum of the Gaussian beam in k_rho.

Parameters
----------
k_rho_arr : float or ndarray
  Radial spatial frequency(ies).
omega : float
  Beam waist parameter.

note that, k_\rho^2 = k_x^2 + k_y^2

Returns
-------
ndarray
  Spectral amplitude at each k_rho.
"""


def full_field(x, y, z, k, w0):
    """
    Returns Ex at (x, y, z) and definite integral of Ez over z0 from 0 to z.
    Accepts x, y, z as floats or ndarrays (must be broadcastable to the same shape).

    Parameters
    ----------
    x, y, z : float or ndarray
        Cartesian coordinates (broadcastable).
    k : float
        Wavenumber.
    w0 : float
        Beam waist parameter.

    Returns
    -------
    Ex : complex ndarray
        Transverse (x) electric field at (x, y, z).
    Ez_integrated : complex ndarray
        Definite integral of Ez(x, y, z0) over z0 from 0 to z.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    # Broadcast all arrays to the same shape
    x, y, z = np.broadcast_arrays(x, y, z)

    denom = k * w0**2 + 2j * z
    Ex = np.exp(k * (-(x**2 + y**2) / denom + 1j * z)) * k * w0**2 / denom

    def Ez_integral_element(xi, yi, zi):
        def ez_integrand(z0):
            denom0 = k * w0**2 + 2j * z0
            num = np.exp(k * (-(xi**2 + yi**2) / denom0 + 1j * z0)) * k**2 * w0**2 * xi
            return -2 * num / denom0**2

        real_part, _ = quad(lambda z0: ez_integrand(z0).real, 0, zi, limit=200)
        imag_part, _ = quad(lambda z0: ez_integrand(z0).imag, 0, zi, limit=200)
        return real_part + 1j * imag_part

    Ez_integrated = -np.vectorize(Ez_integral_element, otypes=[np.complex128])(x, y, z)

    return Ex, Ez_integrated


def field_derivatives(x, y, z, k, w0, h=1e-6):
    """
    Compute field derivatives using finite differences on full_field().
    Guarantees consistency with full_field() definition.
    
    Parameters
    ----------
    x, y, z : float or ndarray
        Cartesian coordinates.
    k : float
        Wavenumber.
    w0 : float
        Beam waist parameter.
    h : float
        Step size for finite differences.
    
    Returns
    -------
    dict with Ex, Ey, Ez and all 1st/2nd derivatives
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    
    Ex, Ez = full_field(x, y, z, k, w0)
    
    Ex_xp, _ = full_field(x + h, y, z, k, w0)
    Ex_xm, _ = full_field(x - h, y, z, k, w0)
    Ex_yp, _ = full_field(x, y + h, z, k, w0)
    Ex_ym, _ = full_field(x, y - h, z, k, w0)
    Ex_zp, Ez_zp = full_field(x, y, z + h, k, w0)
    Ex_zm, Ez_zm = full_field(x, y, z - h, k, w0)
    
    Ex_x = (Ex_xp - Ex_xm) / (2 * h)
    Ex_y = (Ex_yp - Ex_ym) / (2 * h)
    Ex_z = (Ex_zp - Ex_zm) / (2 * h)
    
    Ex_xx = (Ex_xp - 2 * Ex + Ex_xm) / h**2
    Ex_yy = (Ex_yp - 2 * Ex + Ex_ym) / h**2
    Ex_zz = (Ex_zp - 2 * Ex + Ex_zm) / h**2
    
    Ez_z = (Ez_zp - Ez_zm) / (2 * h)
    Ez_zz = (Ez_zp - 2 * Ez + Ez_zm) / h**2
    
    zero = np.zeros_like(Ex)
    
    return {
        "Ex": Ex,
        "Ey": zero,
        "Ez": Ez,
        "Ex_x": Ex_x,
        "Ex_y": Ex_y,
        "Ex_z": Ex_z,
        "Ey_x": zero,
        "Ey_y": zero,
        "Ey_z": zero,
        "Ez_x": zero,
        "Ez_y": zero,
        "Ez_z": Ez_z,
        "Ex_xx": Ex_xx,
        "Ex_xy": zero,
        "Ex_xz": zero,
        "Ex_yy": Ex_yy,
        "Ex_yz": zero,
        "Ex_zz": Ex_zz,
        "Ey_xx": zero,
        "Ey_xy": zero,
        "Ey_xz": zero,
        "Ey_yy": zero,
        "Ey_yz": zero,
        "Ey_zz": zero,
        "Ez_xx": zero,
        "Ez_xy": zero,
        "Ez_xz": zero,
        "Ez_yy": zero,
        "Ez_yz": zero,
        "Ez_zz": Ez_zz,
    }


def force_from_fields(E_dict, alpha, eps0):
    """
    Compute force vector from field and derivative dict.
    
    Parameters
    ----------
    E_dict : dict
        Dict with Ex, Ey, Ez, Ex_x, Ex_y, Ex_z, Ey_x, Ey_y, Ey_z, Ez_x, Ez_y, Ez_z
    alpha : complex
        Polarizability.
    eps0 : float
        Permittivity of free space.
    
    Returns
    -------
    F : ndarray of shape (3,)
        Force components [Fx, Fy, Fz]
    """
    Ex = E_dict["Ex"]
    Ey = E_dict["Ey"]
    Ez = E_dict["Ez"]
    
    Ex_x = E_dict["Ex_x"]
    Ex_y = E_dict["Ex_y"]
    Ex_z = E_dict["Ex_z"]
    Ey_x = E_dict["Ey_x"]
    Ey_y = E_dict["Ey_y"]
    Ey_z = E_dict["Ey_z"]
    Ez_x = E_dict["Ez_x"]
    Ez_y = E_dict["Ez_y"]
    Ez_z = E_dict["Ez_z"]
    
    prefac = eps0 / 2.0
    a_star = np.conjugate(alpha)
    
    Fx = prefac * (a_star * (np.conj(Ex) * Ex_x + np.conj(Ey) * Ey_x + np.conj(Ez) * Ez_x) + 
                   alpha * (Ex * np.conj(Ex_x) + Ey * np.conj(Ey_x) + Ez * np.conj(Ez_x))) / 2.0
    Fy = prefac * (a_star * (np.conj(Ex) * Ex_y + np.conj(Ey) * Ey_y + np.conj(Ez) * Ez_y) + 
                   alpha * (Ex * np.conj(Ex_y) + Ey * np.conj(Ey_y) + Ez * np.conj(Ez_y))) / 2.0
    Fz = prefac * (a_star * (np.conj(Ex) * Ex_z + np.conj(Ey) * Ey_z + np.conj(Ez) * Ez_z) + 
                   alpha * (Ex * np.conj(Ex_z) + Ey * np.conj(Ey_z) + Ez * np.conj(Ez_z))) / 2.0
    
    return np.array([Fx, Fy, Fz])


# def field_derivatives_at_point(
#    x,
#    y,
#    z,
#    k,
#    omega,
#    field_func=full_field_cartesian,
#    hx=None,
#    hy=None,
#    hz=None,
#    edge_safety=True,
#    **field_kwargs,
# ):
#    """
#    Derivatives of (Ex, Ey, Ez) at a single point (x, y, z) using np.gradient
#    on a local 3x3x3 stencil.
#
#    Computes, for each component F ∈ {Ex, Ey, Ez}:
#      First-order:  F_x, F_y, F_z
#      Second-order: F_xx, F_yy, F_zz, F_xy, F_xz, F_yz
#
#    Args:
#      x, y, z, k, omega: floats
#      field_func: callable like full_field_cartesian(X, Y, Z, k, omega, **kwargs)
#      hx, hy, hz: optional step sizes; if None, chosen automatically
#      edge_safety: if True, shrink steps so (coord - h) > 0 (useful if your model
#                   has removable singularities at zero)
#      **field_kwargs: forwarded to field_func (e.g., E0=1.0)
#
#    Returns:
#      dict with field values and all first/second partials at (x,y,z).
#    """
#    x = float(x)
#    y = float(y)
#    z = float(z)
#
#    def pick_h(v, h):
#        if h is not None:
#            return float(h)
#        h_auto = 1e-6 * (1.0 + abs(v))
#        if edge_safety and (v - h_auto) <= 0.0:
#            h_auto = max(0.5 * max(v, 1e-6), 1e-9)
#        return h_auto
#
#    hx = pick_h(x, hx)
#    hy = pick_h(y, hy)
#    hz = pick_h(z, hz)
#
#    # Local 3-point coordinates along each axis
#    x_loc = np.array([x - hx, x, x + hx], dtype=float)
#    y_loc = np.array([y - hy, y, y + hy], dtype=float)
#    z_loc = np.array([z - hz, z, z + hz], dtype=float)
#
#    # Build tiny grid (axis order: x, y, z)
#    X, Y, Z = np.meshgrid(x_loc, y_loc, z_loc, indexing="ij")  # shape (3,3,3)
#
#    # Evaluate fields on the stencil
#    Ex, Ey, Ez = field_func(X, Y, Z, k, omega, **field_kwargs)
#
#    # First derivatives for each component
#    Ex_x, Ex_y, Ex_z = np.gradient(Ex, x_loc, y_loc, z_loc, edge_order=2)
#    Ey_x, Ey_y, Ey_z = np.gradient(Ey, x_loc, y_loc, z_loc, edge_order=2)
#    Ez_x, Ez_y, Ez_z = np.gradient(Ez, x_loc, y_loc, z_loc, edge_order=2)
#
#    # Second derivatives (pure)
#    Ex_xx = np.gradient(Ex_x, x_loc, axis=0, edge_order=2)
#    Ex_yy = np.gradient(Ex_y, y_loc, axis=1, edge_order=2)
#    Ex_zz = np.gradient(Ex_z, z_loc, axis=2, edge_order=2)
#
#    Ey_xx = np.gradient(Ey_x, x_loc, axis=0, edge_order=2)
#    Ey_yy = np.gradient(Ey_y, y_loc, axis=1, edge_order=2)
#    Ey_zz = np.gradient(Ey_z, z_loc, axis=2, edge_order=2)
#
#    Ez_xx = np.gradient(Ez_x, x_loc, axis=0, edge_order=2)
#    Ez_yy = np.gradient(Ez_y, y_loc, axis=1, edge_order=2)
#    Ez_zz = np.gradient(Ez_z, z_loc, axis=2, edge_order=2)
#
#    # Mixed second derivatives via sequential gradients
#    Ex_xy = np.gradient(Ex_x, y_loc, axis=1, edge_order=2)
#    Ex_xz = np.gradient(Ex_x, z_loc, axis=2, edge_order=2)
#    Ex_yz = np.gradient(Ex_y, z_loc, axis=2, edge_order=2)
#
#    Ey_xy = np.gradient(Ey_x, y_loc, axis=1, edge_order=2)
#    Ey_xz = np.gradient(Ey_x, z_loc, axis=2, edge_order=2)
#    Ey_yz = np.gradient(Ey_y, z_loc, axis=2, edge_order=2)
#
#    Ez_xy = np.gradient(Ez_x, y_loc, axis=1, edge_order=2)
#    Ez_xz = np.gradient(Ez_x, z_loc, axis=2, edge_order=2)
#    Ez_yz = np.gradient(Ez_y, z_loc, axis=2, edge_order=2)
#
#    # Extract center (x, y, z) = index (1,1,1)
#    C = (1, 1, 1)
#    out = {
#        # Field values at center
#        "Ex": Ex[C],
#        "Ey": Ey[C],
#        "Ez": Ez[C],
#        # First-order partials
#        "Ex_x": Ex_x[C],
#        "Ex_y": Ex_y[C],
#        "Ex_z": Ex_z[C],
#        "Ey_x": Ey_x[C],
#        "Ey_y": Ey_y[C],
#        "Ey_z": Ey_z[C],
#        "Ez_x": Ez_x[C],
#        "Ez_y": Ez_y[C],
#        "Ez_z": Ez_z[C],
#        # Second-order partials (pure)
#        "Ex_xx": Ex_xx[C],
#        "Ex_yy": Ex_yy[C],
#        "Ex_zz": Ex_zz[C],
#        "Ey_xx": Ey_xx[C],
#        "Ey_yy": Ey_yy[C],
#        "Ey_zz": Ey_zz[C],
#        "Ez_xx": Ez_xx[C],
#        "Ez_yy": Ez_yy[C],
#        "Ez_zz": Ez_zz[C],
#        # Second-order partials (mixed)
#        "Ex_xy": Ex_xy[C],
#        "Ex_xz": Ex_xz[C],
#        "Ex_yz": Ex_yz[C],
#        "Ey_xy": Ey_xy[C],
#        "Ey_xz": Ey_xz[C],
#        "Ey_yz": Ey_yz[C],
#        "Ez_xy": Ez_xy[C],
#        "Ez_xz": Ez_xz[C],
#        "Ez_yz": Ez_yz[C],
#        # Steps used
#        # "hx": hx,
#        # "hy": hy,
#        # "hz": hz,
#    }
#    return out
#
