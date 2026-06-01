function OPpot = getOPpotential(...
    lmax, k, rVec, alphaE, alphaM)
[E, H] = EderivativesSVWE(lmax, k, rVec);

% %% reduced basis
% A1 = [E.o; [E.dx; E.dy; E.dz]./k;...
%     [E.dxdx; E.dxdy; E.dxdz;...
%     E.dydy; E.dydz; E.dzdz]./k^2];
% Pchol = chol(Pmat);
% A2 = A1/Pchol;
% 
% [U,S,V] = svd(A2, 'econ');
% s = diag(S);
% 
% t = 1e-4;
% figure;
% semilogy(diag(S), '*');

%% basis reduction

% Tmat = Pchol\V(:,s>max(s)*t);

%% potential operators
% if rdc
%     Pmat = eye(size(Tmat,2));
%     potential_fun = @(M) Tmat'*((alpha*M)' + alpha*M)*Tmat./4;
% 
%     OPpotential.Efield.o = E.o*Tmat;
% 
%     OPpotential.Efield.dx = E.dx*Tmat;
%     OPpotential.Efield.dy = E.dy*Tmat;
%     OPpotential.Efield.dz = E.dz*Tmat;
% 
%     OPpotential.Efield.dxdx = E.dxdx*Tmat;
%     OPpotential.Efield.dxdy = E.dxdy*Tmat;
%     OPpotential.Efield.dxdz = E.dxdz*Tmat;
%     OPpotential.Efield.dydy = E.dydy*Tmat;
%     OPpotential.Efield.dydz = E.dydz*Tmat;
%     OPpotential.Efield.dzdz = E.dzdz*Tmat;
% else
%     
% end

potential_fun = @(M, alpha) ((alpha*M)' + alpha*M)./4;

%% E-field
potential_Efun = @(M) potential_fun(M, alphaE);

OPpot.Efield = E;

OPpot.EE = 0.5*potential_Efun(E.o'*E.o);

OPpot.EdxE = potential_Efun(E.dx'*E.o);
OPpot.EdyE = potential_Efun(E.dy'*E.o);
OPpot.EdzE = potential_Efun(E.dz'*E.o);

OPpot.EdxEdx = potential_Efun(E.dx'*E.dx + E.dxdx'*E.o);
OPpot.EdxEdy = potential_Efun(E.dx'*E.dy + E.dxdy'*E.o);
OPpot.EdxEdz = potential_Efun(E.dx'*E.dz + E.dxdz'*E.o);
OPpot.EdyEdy = potential_Efun(E.dy'*E.dy + E.dydy'*E.o);
OPpot.EdyEdz = potential_Efun(E.dy'*E.dz + E.dydz'*E.o);
OPpot.EdzEdz = potential_Efun(E.dz'*E.dz + E.dzdz'*E.o);

OPpot.NSW = E.NSW;

%% H-field
potential_Hfun = @(M) potential_fun(M,alphaM);

OPpot.Hfield = H;

OPpot.HH = 0.5*potential_Hfun(H.o'*H.o);

OPpot.HdxH = potential_Hfun(H.dx'*H.o);
OPpot.HdyH = potential_Hfun(H.dy'*H.o);
OPpot.HdzH = potential_Hfun(H.dz'*H.o);

OPpot.HdxHdx = potential_Hfun(H.dx'*H.dx + H.dxdx'*H.o);
OPpot.HdxHdy = potential_Hfun(H.dx'*H.dy + H.dxdy'*H.o);
OPpot.HdxHdz = potential_Hfun(H.dx'*H.dz + H.dxdz'*H.o);
OPpot.HdyHdy = potential_Hfun(H.dy'*H.dy + H.dydy'*H.o);
OPpot.HdyHdz = potential_Hfun(H.dy'*H.dz + H.dydz'*H.o);
OPpot.HdzHdz = potential_Hfun(H.dz'*H.dz + H.dzdz'*H.o);

%% p corss m^H
potential_EHfun = @(M) ((alphaE*alphaM'*M)' +...
    alphaE*alphaM'*M)./4;

OPpot.ExH = potential_EHfun(H.o(3,:)'*E.o(2,:) - H.o(2,:)'*E.o(3,:));
OPpot.EyH = potential_EHfun(H.o(1,:)'*E.o(3,:) - H.o(3,:)'*E.o(1,:));
OPpot.EzH = potential_EHfun(H.o(2,:)'*E.o(1,:) - H.o(1,:)'*E.o(2,:));

OPpot.dxExH = potential_EHfun(H.dx(3,:)'*E.o(2,:) - H.dx(2,:)'*E.o(3,:) + ...
    H.o(3,:)'*E.dx(2,:) - H.o(2,:)'*E.dx(3,:));
OPpot.dyExH = potential_EHfun(H.dy(3,:)'*E.o(2,:) - H.dy(2,:)'*E.o(3,:) + ...
    H.o(3,:)'*E.dy(2,:) - H.o(2,:)'*E.dy(3,:));
OPpot.dzExH = potential_EHfun(H.dz(3,:)'*E.o(2,:) - H.dz(2,:)'*E.o(3,:) + ...
    H.o(3,:)'*E.dz(2,:) - H.o(2,:)'*E.dz(3,:));

OPpot.dxEyH = potential_EHfun(H.dx(1,:)'*E.o(3,:) - H.dx(3,:)'*E.o(1,:) + ...
    H.o(1,:)'*E.dx(3,:) - H.o(3,:)'*E.dx(1,:));
OPpot.dyEyH = potential_EHfun(H.dy(1,:)'*E.o(3,:) - H.dy(3,:)'*E.o(1,:) + ...
    H.o(1,:)'*E.dy(3,:) - H.o(3,:)'*E.dy(1,:));
OPpot.dzEyH = potential_EHfun(H.dz(1,:)'*E.o(3,:) - H.dz(3,:)'*E.o(1,:) + ...
    H.o(1,:)'*E.dz(3,:) - H.o(3,:)'*E.dz(1,:));

OPpot.dxEzH = potential_EHfun(H.dx(2,:)'*E.o(1,:) - H.dx(1,:)'*E.o(2,:) + ...
    H.o(2,:)'*E.dx(1,:) - H.o(1,:)'*E.dx(2,:));
OPpot.dyEzH = potential_EHfun(H.dy(2,:)'*E.o(1,:) - H.dy(1,:)'*E.o(2,:) + ...
    H.o(2,:)'*E.dy(1,:) - H.o(1,:)'*E.dy(2,:));
OPpot.dzEzH = potential_EHfun(H.dz(2,:)'*E.o(1,:) - H.dz(1,:)'*E.o(2,:) + ...
    H.o(2,:)'*E.dz(1,:) - H.o(1,:)'*E.dz(2,:));

end
