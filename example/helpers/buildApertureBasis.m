function basis_ap = buildApertureBasis(P_ap, A1_ap)

    [V_p, D_p] = eig(P_ap, 'vector');

    tol_p = max(D_p) * 1e-4;
    keep_p = D_p > tol_p;

    V_red = V_p(:, keep_p);
    D_red = D_p(keep_p);

    power_basis = V_red * diag(1 ./ sqrt(D_red));
    A1_power = A1_ap * power_basis;

    [~, S_ap, V_ap] = svd(A1_power, 'econ');
    s_ap = diag(S_ap);
    keep_s = s_ap > max(s_ap) * 1e-4;

    basis_ap = power_basis * V_ap(:, keep_s);
end
