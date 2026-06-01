# %%
import numpy as np
import matplotlib.pyplot as plt
import axi_ap

# %%

params = axi_ap.params(k=1.0, b=35.0, w0=15.0, units="CGS")
params.phase_sign = "plus"



#params = axi_ap.params(k=1, b=80, w0=20, units="CGS")
#params.n = 100

ap = axi_ap.bessel_aperture(params=params, num_nodes=150)

ops = axi_ap.Operators(ap)


# %%
z_range = np.linspace(0.02, 100, 100)

Full_data = {
    key: np.zeros(len(z_range), dtype=np.complex128)
    for key in [
        "Ex",
        "Ey",
        "Ez",
        "Ex_x",
        "Ex_y",
        "Ex_z",
        "Ey_x",
        "Ey_y",
        "Ey_z",
        "Ez_x",
        "Ez_y",
        "Ez_z",
        "Ex_xx",
        "Ex_xy",
        "Ex_xz",
        "Ex_yy",
        "Ex_yz",
        "Ex_zz",
        "Ey_xx",
        "Ey_xy",
        "Ey_xz",
        "Ey_yy",
        "Ey_yz",
        "Ey_zz",
        "Ez_xx",
        "Ez_xy",
        "Ez_xz",
        "Ez_yy",
        "Ez_yz",
        "Ez_zz",
    ]
}
Full_F = np.zeros((len(z_range), 3), dtype=np.complex128)

Paraxial_data = {
    key: np.zeros(len(z_range), dtype=np.complex128)
    for key in [
        "Ex",
        "Ey",
        "Ez",
        "Ex_x",
        "Ex_y",
        "Ex_z",
        "Ey_x",
        "Ey_y",
        "Ey_z",
        "Ez_x",
        "Ez_y",
        "Ez_z",
        "Ex_xx",
        "Ex_xy",
        "Ex_xz",
        "Ex_yy",
        "Ex_yz",
        "Ex_zz",
        "Ey_xx",
        "Ey_xy",
        "Ey_xz",
        "Ey_yy",
        "Ey_yz",
        "Ey_zz",
        "Ez_xx",
        "Ez_xy",
        "Ez_xz",
        "Ez_yy",
        "Ez_yz",
        "Ez_zz",
    ]
}
Paraxial_F = np.zeros((len(z_range), 3), dtype=np.complex128)
Full_Fz_explicit = np.zeros(len(z_range), dtype=np.complex128)

alpha = 1 + 1j * 1 / 2
eps0 = params.epsilon_0

for i, z in enumerate(z_range):
    pack = ops.build(x=0, y=0, z=z, params=params)

    for key in Full_data.keys():
        Full_data[key][i] = getattr(pack, key) @ ap.coeffs

    F_full, H_full = ops.force_hessian_operators(pack, alpha=alpha, eps0=eps0)
    Full_F[i] = [ap.coeffs @ F_full[d] @ ap.coeffs for d in range(3)]

    par_deriv = axi_ap.paraxial_gauss.field_derivatives(
        x=0, y=0, z=z, k=params.k, w0=params.w0
    )
    for key in Paraxial_data.keys():
        Paraxial_data[key][i] = par_deriv[key]

    Paraxial_F[i] = axi_ap.paraxial_gauss.force_from_fields(
        par_deriv, alpha=alpha, eps0=eps0
    )

    Ex = Full_data["Ex"][i]
    Ey = Full_data["Ey"][i]
    Ez = Full_data["Ez"][i]
    Ex_z = Full_data["Ex_z"][i]
    Ey_z = Full_data["Ey_z"][i]
    Ez_z = Full_data["Ez_z"][i]

    Fz_explicit = (
        (eps0 / 2.0)
        * (
            np.conj(alpha)
            * (np.conj(Ex) * Ex_z + np.conj(Ey) * Ey_z + np.conj(Ez) * Ez_z)
            + alpha * (Ex * np.conj(Ex_z) + Ey * np.conj(Ey_z) + Ez * np.conj(Ez_z))
        )
        / 2.0
    )
    Full_Fz_explicit[i] = Fz_explicit
    print(i)
# %%
rho_vec = ap.rho
basis = ap.basis
E_gauss_r = basis.T @ ap.coeffs

plt.plot(rho_vec, np.real(E_gauss_r), label="Gaussian")



# %%
plt.figure(figsize=(10, 5))
plt.plot(
    z_range, np.real(Full_data["Ex"]), label="Full Ex", linewidth=2, linestyle="-."
)
plt.plot(
    z_range,
    np.real(Paraxial_data["Ex"]),
    label="Paraxial Ex",
    linewidth=2,
    linestyle="--",
)
plt.xlabel("z")
plt.ylabel("Ex")
plt.legend()
plt.grid()
plt.show()

# %%
plt.figure(figsize=(10, 5))
plt.plot(
    z_range, np.real(Full_data["Ex_z"]), label="Full Ex_z", linewidth=2, linestyle="-."
)
plt.plot(
    z_range,
    np.real(Paraxial_data["Ex_z"]),
    label="Paraxial Ex_z",
    linewidth=2,
    linestyle="--",
)
plt.xlabel("z")
plt.ylabel("Ex_z")
plt.legend()
plt.grid()
plt.show()

# %%
plt.figure(figsize=(10, 5))
plt.plot(
    z_range,
    np.real(Full_data["Ex_zz"]),
    label="Full Ex_zz",
    linewidth=2,
    linestyle="-.",
)
plt.plot(
    z_range,
    np.real(Paraxial_data["Ex_zz"]),
    label="Paraxial Ex_zz",
    linewidth=2,
    linestyle="--",
)
plt.xlabel("z")
plt.ylabel("Ex_zz")
plt.legend()
plt.grid()
plt.show()

# %%
plt.figure(figsize=(10, 5))
plt.plot(
    z_range,
    np.real(Full_F[:, 2]),
    label="Full Fz (operators)",
    linewidth=2,
    linestyle="-.",
)
plt.plot(
    z_range,
    np.real(Full_Fz_explicit),
    label="Full Fz (explicit)",
    linewidth=2,
    linestyle="--",
)
plt.plot(
    z_range, np.real(Paraxial_F[:, 2]), label="Paraxial Fz", linewidth=2, linestyle="-."
)
plt.xlabel("z")
plt.ylabel("Fz")
plt.legend()
plt.grid()
plt.show()

# %%

# %% Plotting Absolute Value and Argument

Ex_full = Full_data["Ex"]
Ex_par = Paraxial_data["Ex"]

abs_full = np.real(Ex_full)
arg_full = np.imag((Ex_full))

abs_par = np.real(Ex_par)
arg_par = np.imag(Ex_par)

z_R = 0.5 * params.k * params.w0**2
z_norm = z_range / z_R

plt.figure(figsize=(8, 4.94))

plt.plot(z_norm, abs_par, color='gray', linewidth=4, alpha=0.4, label=r'Analytical $|E_x|$')
plt.plot(z_norm, abs_full, marker='s', markersize=5, markerfacecolor='white', markeredgecolor='black', linestyle='None', label=r'Numerical $|E_x|$')

plt.plot(z_norm, arg_par, color='#b61516', linewidth=1.5, label=r'Analytical $\arg(E_x)$')
plt.plot(z_norm, arg_full, marker='o', markersize=4, markerfacecolor='white', markeredgecolor='black', linestyle='None', label=r'Numerical $\arg(E_x)$')

plt.xlabel(r'$z / z_\mathrm{R}$')
plt.ylabel(r'$E_x(0, z)$ Amplitude & Phase')
plt.grid(True, linestyle='--', color='gray', alpha=0.4)
plt.legend(loc='upper right', framealpha=0.9, edgecolor='black')

plt.tight_layout()
plt.show()