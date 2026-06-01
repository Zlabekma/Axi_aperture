function [E, H] = EderivativesSVWE(lmax, k, rVec, reqDer)
% Central difference

if nargin<4
    reqDer = [0,1,2];
end

p = 1;

need1 = any(reqDer > 0);
need2 = any(reqDer > 1);

if need2
    nPts = 19;
elseif need1
    nPts = 7;
else
    nPts = 1;
end

dPos = nan(nPts,3);
dPos(1,:) = rVec;

h = 1e-5/k;
hx = h; hy = h; hz = h;

if need1
    dPos(2,:) = rVec + [ hx 0  0];
    dPos(3,:) = rVec + [-hx 0  0];
    dPos(4,:) = rVec + [ 0  hy 0];
    dPos(5,:) = rVec + [ 0 -hy 0];
    dPos(6,:) = rVec + [ 0  0  hz];
    dPos(7,:) = rVec + [ 0  0 -hz];
end
if need2
    dPos(8,:)  = rVec + [ hx  hy 0];
    dPos(9,:)  = rVec + [ hx -hy 0];
    dPos(10,:) = rVec + [-hx  hy 0];
    dPos(11,:) = rVec + [-hx -hy 0];
    dPos(12,:) = rVec + [ hx 0  hz];
    dPos(13,:) = rVec + [ hx 0 -hz];
    dPos(14,:) = rVec + [-hx 0  hz];
    dPos(15,:) = rVec + [-hx 0 -hz];
    dPos(16,:) = rVec + [ 0  hy  hz];
    dPos(17,:) = rVec + [ 0  hy -hz];
    dPos(18,:) = rVec + [ 0 -hy  hz];
    dPos(19,:) = rVec + [ 0 -hy -hz];
end

dR     = nan(nPts,1);
dTheta = nan(nPts,1);
dPhi   = nan(nPts,1);
for i = 1:nPts
    [dR(i), dTheta(i), dPhi(i)] = models.utilities.converter.cart2sph(...
        dPos(i,1), dPos(i,2), dPos(i,3));
end

indexMatrix = models.utilities.matrixOperators.MoM2D.SMatrix.indexMatrix(lmax);
E.NSW = size(indexMatrix,2);
E.o   = nan(3,E.NSW);

if need1
    E.dx = E.o; E.dy = E.o; E.dz = E.o;
end
if need2
    E.dxdx = E.o; E.dxdy = E.o; E.dxdz = E.o;
    E.dydy = E.o; E.dydz = E.o; E.dzdz = E.o;
end
H = E;

u2E = @(u) k*sqrt(models.utilities.constants.Z0)*u;
u2H = @(u) 1i*k/sqrt(models.utilities.constants.Z0)*u;

idx.base = 1;
if need1
    idx.xp = 2; idx.xm = 3;
    idx.yp = 4; idx.ym = 5;
    idx.zp = 6; idx.zm = 7;
end
if need2
    idx.xpyp = 8;  idx.xpym = 9;  idx.xmyp = 10; idx.xmym = 11;
    idx.xpzp = 12; idx.xpzm = 13; idx.xmzp = 14; idx.xmzm = 15;
    idx.ypzp = 16; idx.ypzm = 17; idx.ymzp = 18; idx.ymzm = 19;
end

for iCol = 1:E.NSW
    L     = indexMatrix(1,iCol);
    M     = indexMatrix(2,iCol);
    sigma = indexMatrix(3,iCol);
    tau   = indexMatrix(4,iCol);
    alpha = indexMatrix(5,iCol);

    [u12, u11, u22, u21] = models.utilities.matrixOperators.MoM2D.SMatrix.functionU(...
        L, M, dTheta, dPhi, k*dR, p);

    if tau == 1
        if sigma == 2
            uE = u12; 
            uH = u22; 
        else
            uE = u11; 
            uH = u21; 
        end
    else
        if sigma == 2
            uE = u22; 
            uH = u12; 
        else
            uE = u21; 
            uH = u11; 
        end
    end

    uE = reshape(uE,[nPts,3]);
    uH = reshape(uH,[nPts,3]);

    [uE(:,1),uE(:,2),uE(:,3)] = models.utilities.converter.vecSph2Cart(...
        uE(:,1),uE(:,2),uE(:,3), dTheta,dPhi);
    [uH(:,1),uH(:,2),uH(:,3)] = models.utilities.converter.vecSph2Cart(...
        uH(:,1),uH(:,2),uH(:,3), dTheta,dPhi);

    fE0 = uE(idx.base,:); fH0 = uH(idx.base,:);

    E.o(:,alpha) = u2E(fE0.');
    H.o(:,alpha) = u2H(fH0.');

    if need1
        Ex_p = uE(idx.xp,:); Ex_m = uE(idx.xm,:);
        Ey_p = uE(idx.yp,:); Ey_m = uE(idx.ym,:);
        Ez_p = uE(idx.zp,:); Ez_m = uE(idx.zm,:);

        Hx_p = uH(idx.xp,:); Hx_m = uH(idx.xm,:);
        Hy_p = uH(idx.yp,:); Hy_m = uH(idx.ym,:);
        Hz_p = uH(idx.zp,:); Hz_m = uH(idx.zm,:);

        E.dx(:,alpha) = u2E(((Ex_p - Ex_m)/(2*hx)).');
        E.dy(:,alpha) = u2E(((Ey_p - Ey_m)/(2*hy)).');
        E.dz(:,alpha) = u2E(((Ez_p - Ez_m)/(2*hz)).');

        H.dx(:,alpha) = u2H(((Hx_p - Hx_m)/(2*hx)).');
        H.dy(:,alpha) = u2H(((Hy_p - Hy_m)/(2*hy)).');
        H.dz(:,alpha) = u2H(((Hz_p - Hz_m)/(2*hz)).');
    end

    if need2
        E.dxdx(:,alpha) = u2E(((uE(idx.xp,:) - 2*fE0 + uE(idx.xm,:))/hx^2).');
        E.dydy(:,alpha) = u2E(((uE(idx.yp,:) - 2*fE0 + uE(idx.ym,:))/hy^2).');
        E.dzdz(:,alpha) = u2E(((uE(idx.zp,:) - 2*fE0 + uE(idx.zm,:))/hz^2).');

        H.dxdx(:,alpha) = u2H(((uH(idx.xp,:) - 2*fH0 + uH(idx.xm,:))/hx^2).');
        H.dydy(:,alpha) = u2H(((uH(idx.yp,:) - 2*fH0 + uH(idx.ym,:))/hy^2).');
        H.dzdz(:,alpha) = u2H(((uH(idx.zp,:) - 2*fH0 + uH(idx.zm,:))/hz^2).');

        E.dxdy(:,alpha) = u2E(((uE(idx.xpyp,:) - uE(idx.xpym,:) - uE(idx.xmyp,:) + uE(idx.xmym,:))/(4*hx*hy)).');
        E.dxdz(:,alpha) = u2E(((uE(idx.xpzp,:) - uE(idx.xpzm,:) - uE(idx.xmzp,:) + uE(idx.xmzm,:))/(4*hx*hz)).');
        E.dydz(:,alpha) = u2E(((uE(idx.ypzp,:) - uE(idx.ypzm,:) - uE(idx.ymzp,:) + uE(idx.ymzm,:))/(4*hy*hz)).');

        H.dxdy(:,alpha) = u2H(((uH(idx.xpyp,:) - uH(idx.xpym,:) - uH(idx.xmyp,:) + uH(idx.xmym,:))/(4*hx*hy)).');
        H.dxdz(:,alpha) = u2H(((uH(idx.xpzp,:) - uH(idx.xpzm,:) - uH(idx.xmzp,:) + uH(idx.xmzm,:))/(4*hx*hz)).');
        H.dydz(:,alpha) = u2H(((uH(idx.ypzp,:) - uH(idx.ypzm,:) - uH(idx.ymzp,:) + uH(idx.ymzm,:))/(4*hy*hz)).');
    end
end

end