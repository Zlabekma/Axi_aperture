function P_vsw = buildPvsw(lmax, ka, k)

    indexMatrix = models.utilities.matrixOperators.MoM2D.SMatrix.indexMatrix(lmax);

    lidx = nan(size(indexMatrix, 2), 1);
    tauidx = nan(size(indexMatrix, 2), 1);

    lidx(indexMatrix(5, :), 1) = indexMatrix(1, :);
    tauidx(indexMatrix(5, :), 1) = indexMatrix(4, :);

    [R1, R2, R3, ~] = models.utilities.matrixOperators.MoM2D.SMatrix.functionR(lidx, ka, 1);

    Rsquared = R1.^2;
    Rsquared(tauidx == 2) = R2(tauidx == 2).^2 + R3(tauidx == 2).^2;

    P_vsw = 2 * k * ka^2 * diag(Rsquared);
end
