function basis_vsw = buildVswBasis(P_vsw, A1_vsw)
    [V_p, D_p] = eig(P_vsw, 'vector');
    tol_p = max(D_p) * 1e-4;
    keep_p = D_p > tol_p;
    V_red = V_p(:, keep_p);
    D_red = D_p(keep_p);
    power_basis = V_red * diag(1 ./ sqrt(D_red));
    A1_power = A1_vsw * power_basis;
    [~, S_vsw, V_vsw] = svd(A1_power, 'econ');
    s_vsw = diag(S_vsw);
    keep_s = s_vsw > max(s_vsw) * 1e-4;
    basis_vsw = power_basis * V_vsw(:, keep_s);
end
% function basis_vsw = buildVswBasis(P_vsw, A1_vsw)
% 
%     Pchol_vsw = chol(P_vsw);
% 
%     [~, S_vsw, V_vsw] = svd(A1_vsw / Pchol_vsw, 'econ');
%     s_vsw = diag(S_vsw);
%     keep_s = s_vsw > max(s_vsw) * 1e-4;
% 
%     basis_vsw = Pchol_vsw \ V_vsw(:, keep_s);
% end
