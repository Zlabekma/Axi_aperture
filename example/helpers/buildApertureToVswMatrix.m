function T_ap_to_vsw = buildApertureToVswMatrix( ...
    ops, positions_list, polarizations_list, focal_length, lmax, k, np)

    ka = k * focal_length / 2;
    rObs = models.utilities.thirdParty.getLebedevSphere(400) * ka;

    xObs_py = np.array(rObs(:, 1));
    yObs_py = np.array(rObs(:, 2));
    zObs_py = np.array(rObs(:, 3));

    E_sph_py = ops.get_E_total_op( ...
        xObs_py, ...
        yObs_py, ...
        zObs_py, ...
        pyargs( ...
            'positions', positions_list, ...
            'polarizations', polarizations_list));

    E_sph = double(E_sph_py);

    Ex_sph = squeeze(E_sph(1, :, :));
    Ey_sph = squeeze(E_sph(2, :, :));
    Ez_sph = squeeze(E_sph(3, :, :));

    indexMatrix = models.utilities.matrixOperators.MoM2D.SMatrix.indexMatrix(lmax);
    nSW = size(indexMatrix, 2);
    nBasis = size(E_sph, 3);

    T_ap_to_vsw = zeros(nSW, nBasis);

    for i = 1:nBasis
        E_basis = [Ex_sph(:, i), Ey_sph(:, i), Ez_sph(:, i)];
        T_ap_to_vsw(:, i) = utilities.projectEToSWReal(lmax, k, rObs, E_basis, 1);
    end
end
