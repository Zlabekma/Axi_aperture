%% Python setup
% Start a clean Python session and import the modules used by the
% aperture-based field representation.

%! add "helpers" folder to you matlab path
% run setup_enviroment.m to setup the python package correctly

terminate(pyenv)
pyenv(ExecutionMode="OutOfProcess");
clc
clear
close all

axi_ap = py.importlib.import_module('axi_ap');
scipy_special = py.importlib.import_module('scipy.special');
np = py.importlib.import_module('numpy');

%% Geometry and numerical settings
% Define the physical and numerical parameters used throughout the
% aperture and VSW calculations.
k = 1.0;                 % wavevector magnitude
b = 39.0;                % aperture radius
w0 = 3.0;                % Gaussian-beam reference parameter
focal_length = 10.0;     % distance from aperture to trapping point
hstep = 5e-4 / k;        % finite-difference step for spatial derivatives
num_rho = int32(500);    % number of radial quadrature points
lmax_ap = 21;            % truncation order for aperture-to-VSW projection
verbosity = 0;           % solver verbosity level
energy_constraint = 10;   % quadratic energy normalization

%% Python parameters and aperture object
% Initialize the aperture model and choose a sufficient number of
% Fourier-Bessel modes to represent both propagating and evanescent content.
params_obj = axi_ap.params(pyargs( ...
    'k', k, ...
    'b', b, ...
    'w0', w0, ...
    'units', 'CGS'));
params_obj.phase_sign = 'minus';

estimated_roots_needed = int32(ceil(k * b / pi) + 10);
alpha_n_py = scipy_special.jn_zeros(int32(0), estimated_roots_needed);
alpha_n = double(alpha_n_py);
max_propagating_n = sum(alpha_n <= (k * b));
params_obj.n = int32(max_propagating_n + 20);

ap = axi_ap.bessel_aperture(pyargs( ...
    'params', params_obj, ...
    'num_nodes', params_obj.n, ...
    'num_rho', num_rho));
ops = axi_ap.Operators(ap);

%% Aperture geometry
% The multi-aperture illumination is defined by the aperture positions and
% by the polarization assigned to each channel. Different configurations
% can be explored by moving the apertures or changing these polarizations.
f = focal_length;

positions = [
     f,  0,  0;
     f,  0,  0;
    -f,  0,  0;
    -f,  0,  0;
     0,  f,  0;
     0,  f,  0;
     0, -f,  0;
     0, -f,  0;
     0,  0,  f;
     0,  0,  f;
     0,  0, -f;
     0,  0, -f
];

polarizations = [
     0, 0, 1;
     0, 1, 0;
     0, 0, 1;
     0, 1, 0;
     1, 0, 0;
     0, 0, 1;
     1, 0, 0;
     0, 0, 1;
     1, 0, 0;
     0, 1, 0;
     1, 0, 0;
     0, 1, 0
];

% Example: four-aperture configuration with two polarizations per side.
% % positions = [
% %      f,  0,  0;
% %      f,  0,  0;
% %     -f,  0,  0;
% %     -f,  0,  0
% % ];
% %
% % polarizations = [
% %      0, 0, 1;
% %      0, 1, 0;
% %      0, 0, 1;
% %      0, 1, 0
% % ];

% Example: four-aperture test configuration.
% positions = [
%      f, 0, 0;
%      f, 0, 0;
%     -f, 0, 0;
%     -f, 0, 0
% ];
%
% polarizations = [
%      0, 1, 0;
%      0, 1, 1;
%      0, 1, 0;
%      0, 0, 1
% ];

positions_list = py.list(num2cell(positions, 2)');
polarizations_list = py.list(num2cell(polarizations, 2)');

%% Geometry-dependent operators
% Build the local field map at the trapping point, construct the
% aperture-to-VSW transformation, and define the quadratic energy metric.
pack = ops.build( ...
    0.0, 0.0, 0.0, ...
    pyargs( ...
        'positions', positions_list, ...
        'polarizations', polarizations_list, ...
        'hx', hstep, ...
        'hy', hstep, ...
        'hz', hstep, ...
        'include_e', true));

A1_ap = buildA1Aperture(pack, k);

T_ap_to_vsw = buildApertureToVswMatrix( ...
    ops, positions_list, polarizations_list, focal_length, lmax_ap, k, np);

ka = k * focal_length / 2;
P_vsw_ap = buildPvsw(lmax_ap, ka, k);
P_ap = T_ap_to_vsw' * P_vsw_ap * T_ap_to_vsw;
P_ap = (P_ap + P_ap') / 2;

basis_ap = buildApertureBasis(P_ap, A1_ap);

%% Particle parameters
% Define the particle polarizability and construct the corresponding
% aperture-side force and stiffness operators.
zeta = 1.0;                                      % normalized static volume
phi = pi/6;                                       % fixed phase

alpha0 = zeta * (6 * pi / k^3) * exp(1i * phi);
alpha_val = alpha0 / (1 - 1i * (k^3 / (6 * pi)) * alpha0);

FH = ops.force_hessian_operators(pack, pyargs( ...
    'alpha', alpha_val, ...
    'eps0', double(params_obj.epsilon_0)));

F_ap = double(FH{1});
H_ap = double(FH{2});

Fx_ap = squeeze(F_ap(1, :, :));
Fy_ap = squeeze(F_ap(2, :, :));
Fz_ap = squeeze(F_ap(3, :, :));

Hxx_ap = squeeze(H_ap(1, 1, :, :));
Hyy_ap = squeeze(H_ap(2, 2, :, :));
Hzz_ap = squeeze(H_ap(3, 3, :, :));
Hxy_ap = squeeze(H_ap(1, 2, :, :));
Hxz_ap = squeeze(H_ap(1, 3, :, :));
Hyz_ap = squeeze(H_ap(2, 3, :, :));

%% Aperture optimization sweep
% Sweep over trial force-balance directions, reduce the search space by the
% corresponding nullspace projection, and solve the QCQP for each case.
best_trace_ap = 0;
best_power_ap = NaN;
best_direction_ap = [];
best_coeffs_ap = [];
best_dg_ap = NaN;

angles_ap = linspace(0, pi, 10);
trace_values_ap = nan(numel(angles_ap), 1);
power_values_ap = nan(numel(angles_ap), 1);
dg_values_ap = nan(numel(angles_ap), 1);
null_dims_ap = nan(numel(angles_ap), 1);

objective_ap = Hxx_ap + Hyy_ap + Hzz_ap;

for ii = 1:numel(angles_ap)
    angle = angles_ap(ii);

    direction = [sin(angle) * cos(pi/4), ...
                 sin(angle) * sin(pi/4), ...
                 cos(angle)];
    direction = direction / norm(direction);

    null_basis_ap = null((direction * A1_ap(1:3, :)) * basis_ap);
    opt_basis_ap = basis_ap * null_basis_ap;
    null_dims_ap(ii) = size(null_basis_ap, 2);

    if isempty(opt_basis_ap)
        continue
    end

    V0_ap = zeros(size(opt_basis_ap, 1), 1);
    lambda0_ap = zeros(10, 1);

    trip_ap = controllableRegion.transformQuadFormAff( ...
        zeros(size(opt_basis_ap, 1), 1), opt_basis_ap, ...
        objective_ap, V0_ap, 0, ...
        P_ap, V0_ap, -energy_constraint, ...
        Fx_ap, V0_ap, 0, ...
        Fy_ap, V0_ap, 0, ...
        Fz_ap, V0_ap, 0, ...
        Hxy_ap, V0_ap, 0, ...
        Hxz_ap, V0_ap, 0, ...
        Hyz_ap, V0_ap, 0, ...
        Hxx_ap - Hyy_ap, V0_ap, 0, ...
        Hyy_ap - Hzz_ap, V0_ap, 0, ...
        Hxx_ap - Hzz_ap, V0_ap, 0);

    [~, Tn_ap, ~, trip_ap] = solvers.centralize(trip_ap{:});
    [~, trip_ap] = solvers.normalizeTriplet(trip_ap{:});

    [primal_ap, dual_ap, a_ap, lambda_ap, dg_ap] = ...
        solvers.QCQP.minAstBn(trip_ap{1}, lambda0_ap, verbosity, trip_ap{4:end});

    coeffs_ap = opt_basis_ap * (Tn_ap * a_ap);
    trace_ap = real(coeffs_ap' * objective_ap * coeffs_ap);
    power_ap = real(coeffs_ap' * P_ap * coeffs_ap);

    trace_values_ap(ii) = trace_ap;
    power_values_ap(ii) = power_ap;
    dg_values_ap(ii) = dg_ap;

    if dg_ap == 0 && trace_ap < best_trace_ap
        best_trace_ap = trace_ap;
        best_power_ap = power_ap;
        best_direction_ap = direction;
        best_coeffs_ap = coeffs_ap;
        best_dg_ap = dg_ap;
    end
end

%% VSW operators
% Build the corresponding VSW representation so that the aperture-based
% result can be compared with the spherical-wave formulation.
lmax_vsw = 7;                                  % VSW truncation order
P_vsw = buildPvsw(lmax_vsw, ka, k);
OP = getOPpotential(lmax_vsw, k, [0,0,0.1], alpha_val, 0);
A1_vsw = buildA1Vsw(OP, k);
basis_vsw = buildVswBasis(P_vsw, A1_vsw);

%% VSW optimization sweep
% Repeat the constrained QCQP sweep in the VSW basis using the same
% normalization and isotropy constraints.
best_trace_vsw = Inf;
best_power_vsw = NaN;
best_direction_vsw = [];
best_coeffs_vsw = [];
best_dg_vsw = NaN;

angles_vsw = linspace(0, 2*pi, 10);
trace_values_vsw = nan(numel(angles_vsw), 1);
power_values_vsw = nan(numel(angles_vsw), 1);
dg_values_vsw = nan(numel(angles_vsw), 1);
null_dims_vsw = nan(numel(angles_vsw), 1);

objective_vsw = OP.EdxEdx + OP.EdyEdy + OP.EdzEdz;
lambda_vsw = lambda_ap;

for ii = 1:numel(angles_vsw)
    angle = angles_vsw(ii);

    direction = [sin(angle) * cos(pi/4), ...
                 sin(angle) * sin(pi/4), ...
                 cos(angle)];
    direction = direction / norm(direction);

    null_basis_vsw = null((direction * A1_vsw(1:3, :)) * basis_vsw);
    opt_basis_vsw = basis_vsw * null_basis_vsw;
    null_dims_vsw(ii) = size(null_basis_vsw, 2);

    if isempty(opt_basis_vsw)
        continue
    end

    V0_vsw = zeros(size(opt_basis_vsw, 1), 1);

    trip_vsw = controllableRegion.transformQuadFormAff( ...
        zeros(size(opt_basis_vsw, 1), 1), opt_basis_vsw, ...
        objective_vsw, V0_vsw, 0, ...
        P_vsw, V0_vsw, -energy_constraint, ...
        OP.EdxE, V0_vsw, 0, ...
        OP.EdyE, V0_vsw, 0, ...
        OP.EdzE, V0_vsw, 0, ...
        OP.EdxEdy, V0_vsw, 0, ...
        OP.EdxEdz, V0_vsw, 0, ...
        OP.EdyEdz, V0_vsw, 0, ...
        OP.EdxEdx - OP.EdyEdy, V0_vsw, 0, ...
        OP.EdyEdy - OP.EdzEdz, V0_vsw, 0, ...
        OP.EdxEdx - OP.EdzEdz, V0_vsw, 0);

    [~, Tn_vsw, ~, trip_vsw] = solvers.centralize(trip_vsw{:});
    [~, trip_vsw] = solvers.normalizeTriplet(trip_vsw{:});

    [primal_vsw, dual_vsw, a_vsw, lambda_vsw, dg_vsw] = ...
        solvers.QCQP.minAstBn(trip_vsw{1}, lambda_vsw, verbosity, trip_vsw{4:end});
    
    coeffs_vsw = opt_basis_vsw * (Tn_vsw * a_vsw);
    trace_vsw = real(coeffs_vsw' * objective_vsw * coeffs_vsw);
    power_vsw = real(coeffs_vsw' * P_vsw * coeffs_vsw);

    trace_values_vsw(ii) = trace_vsw;
    power_values_vsw(ii) = power_vsw;
    dg_values_vsw(ii) = dg_vsw;

    if dg_vsw == 0 && trace_vsw < best_trace_vsw
        best_trace_vsw = trace_vsw;
        best_power_vsw = power_vsw;
        best_direction_vsw = direction;
        best_coeffs_vsw = coeffs_vsw;
        best_dg_vsw = dg_vsw;
    end
end

%% Results
% Display the best aperture and VSW solutions together with the data
% collected during the directional sweeps.
disp(' ')
disp('=== Aperture result ===')
fprintf('best trace_ap = %.12e\n', best_trace_ap);
fprintf('best power_ap = %.12e\n', best_power_ap);
fprintf('best dg_ap    = %d\n', best_dg_ap);
disp('best direction_ap = ')
disp(best_direction_ap)

disp(' ')
disp('=== VSW result ===')
fprintf('best trace_vsw = %.12e\n', best_trace_vsw);
fprintf('best power_vsw = %.12e\n', best_power_vsw);
fprintf('best dg_vsw    = %d\n', best_dg_vsw);
disp('best direction_vsw = ')
disp(best_direction_vsw)

%%
disp(' ')
disp('=== Aperture sweep table ===')
disp(table( ...
    angles_ap(:), ...
    null_dims_ap, ...
    dg_values_ap, ...
    trace_values_ap, ...
    power_values_ap, ...
    'VariableNames', {'angle','nullDim','dg','trace','power'}))

disp(' ')
disp('=== VSW sweep table ===')
disp(table( ...
    angles_vsw(:), ...
    null_dims_vsw, ...
    dg_values_vsw, ...
    trace_values_vsw, ...
    power_values_vsw, ...
    'VariableNames', {'angle','nullDim','dg','trace','power'}))