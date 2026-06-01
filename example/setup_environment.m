function setup_environment()
%SETUP_ENVIRONMENT Configure and verify MATLAB, Python, axi_ap, and AToM.
%
% This helper:
%   1) Adds likely local repository folders to the MATLAB path.
%   2) Tries to discover nearby AToM / optimization-toolbox folders.
%   3) Reports the active Python interpreter used by MATLAB.
%   4) Tries to install missing Python packages needed by the examples.
%   5) Verifies that required MATLAB / AToM functions are available.
%
% Recommended usage:
%   >> cd example
%   >> setup_environment

fprintf('Checking MATLAB/Python/AToM environment...\n\n');

%% Locate repository root from this file
this_file = mfilename('fullpath');
this_dir = fileparts(this_file);
repo_root = fileparts(this_dir);

fprintf('Detected example folder: %s\n', this_dir);
fprintf('Detected repository root: %s\n\n', repo_root);

%% Optional manual AToM path override
manual_atom_root = '';
% Example:
% manual_atom_root = 'C:\Users\marti\Documents\GIT\AToM';

if ~isempty(manual_atom_root)
    if isfolder(manual_atom_root)
        addpath(genpath(manual_atom_root));
        fprintf('[OK] Added manual AToM path: %s\n\n', manual_atom_root);
    else
        fprintf('[WARN] manual_atom_root does not exist: %s\n\n', manual_atom_root);
    end
end

%% Add likely local MATLAB source folders
candidate_paths = {
    fullfile(repo_root, 'src'), ...
    fullfile(repo_root, 'matlab'), ...
    fullfile(repo_root, 'example')
    };

fprintf('Adding local repository paths when available...\n');
for i = 1:numel(candidate_paths)
    p = candidate_paths{i};
    if isfolder(p)
        addpath(p);
        fprintf('[OK] Added to MATLAB path: %s\n', p);
    else
        fprintf('[INFO] Not found, skipped: %s\n', p);
    end
end
fprintf('\n');

%% Try to discover likely nearby AToM folders
atom_candidates = {
    fullfile(repo_root, '..', 'AToM'), ...
    fullfile(repo_root, '..', '..', 'AToM'), ...
    fullfile(repo_root, '..', 'fundamental_bounds'), ...
    fullfile(repo_root, '..', '..', 'fundamental_bounds')
    };

fprintf('Searching for likely AToM / optimization-toolbox folders...\n');
for i = 1:numel(atom_candidates)
    p = atom_candidates{i};
    if isfolder(p)
        addpath(genpath(p));
        fprintf('[OK] Added recursive path: %s\n', p);
    else
        fprintf('[INFO] Not found, skipped: %s\n', p);
    end
end
fprintf('\n');

%% MATLAB <-> Python
try
    pe = pyenv;
    fprintf('[OK] MATLAB Python interface is available.\n');
    fprintf('     Version: %s\n', pe.Version);
    fprintf('     Executable: %s\n', pe.Executable);
    fprintf('     Status: %s\n', pe.Status);
    fprintf('     ExecutionMode: %s\n\n', pe.ExecutionMode);
catch ME
    error(['[FAIL] MATLAB could not access Python. ' ...
           'Configure a supported CPython interpreter with pyenv first. ' ...
           'Original error: %s'], ME.message);
end

python_exe = string(pe.Executable);

if strlength(python_exe) == 0
    error(['[FAIL] MATLAB did not report a Python executable.' newline ...
           'Set it explicitly, for example:' newline ...
           '    pyenv("Version","C:\Path\To\python.exe","ExecutionMode","OutOfProcess")']);
end

%% Helpers for Python package installation
fprintf('Using Python executable:\n    %s\n\n', python_exe);

install_python_package(python_exe, 'pip', ...
    'python -m ensurepip --upgrade', true);

required_modules = {
    'numpy', 'numpy';
    'scipy.special', 'scipy';
    'axi_ap', '.'
    };

missing_python = {};

for i = 1:size(required_modules, 1)
    module_name = required_modules{i, 1};
    install_target = required_modules{i, 2};

    if ~python_module_available(module_name)
        fprintf('[WARN] Python module "%s" is missing.\n', module_name);

        if strcmp(install_target, '.')
            fprintf('       Attempting editable install from repository root...\n');
            cmd = sprintf('"%s" -m pip install -e "%s"', python_exe, repo_root);
        else
            fprintf('       Attempting installation of package "%s"...\n', install_target);
            cmd = sprintf('"%s" -m pip install %s', python_exe, install_target);
        end

        [status, out] = system(cmd);
        if status == 0
            fprintf('[OK] Installation command succeeded for "%s".\n', module_name);
            fprintf('%s\n', out);
        else
            fprintf('[FAIL] Installation command failed for "%s".\n', module_name);
            fprintf('%s\n', out);
            missing_python{end+1} = module_name; %#ok<AGROW>
            continue
        end

        if python_module_available(module_name)
            fprintf('[OK] Python module "%s" is now available.\n', module_name);
        else
            fprintf('[FAIL] Python module "%s" is still not importable after installation.\n', module_name);
            missing_python{end+1} = module_name; %#ok<AGROW>
        end
    else
        fprintf('[OK] Python module "%s" found.\n', module_name);
    end
end
fprintf('\n');

%% Required MATLAB / AToM functions
matlab_checks = {
    'buildA1Aperture', 'Local aperture operator builder';
    'buildApertureToVswMatrix', 'Aperture-to-VSW projection';
    'buildPvsw', 'VSW normalization matrix';
    'buildApertureBasis', 'Aperture basis reduction';
    'buildA1Vsw', 'VSW local operator builder';
    'buildVswBasis', 'VSW basis reduction';
    'getOPpotential', 'VSW potential operators';
    'controllableRegion.transformQuadFormAff', 'AToM quadratic-form transform';
    'solvers.centralize', 'AToM centralization';
    'solvers.normalizeTriplet', 'AToM triplet normalization';
    'solvers.QCQP.minAstBn', 'AToM QCQP solver';
    'solvers.QCQP.minAstBn_BFGS', 'AToM QCQP BFGS solver'
    };

missing_matlab = {};

for i = 1:size(matlab_checks, 1)
    fname = matlab_checks{i, 1};
    fdesc = matlab_checks{i, 2};

    resolved = which(fname);
    ok_exist = exist(fname, 'file') || exist(fname, 'class');

    if ~isempty(resolved) || ok_exist
        fprintf('[OK] %s found (%s).\n', fname, fdesc);
        if ~isempty(resolved)
            fprintf('     -> %s\n', resolved);
        end
    else
        fprintf('[FAIL] %s missing (%s).\n', fname, fdesc);
        missing_matlab{end+1} = fname; %#ok<AGROW>
    end
end
fprintf('\n');

%% Summary
if isempty(missing_python) && isempty(missing_matlab)
    fprintf('Environment check passed. You can now run the example scripts.\n');
else
    fprintf('Environment setup finished with unresolved items.\n');

    if ~isempty(missing_python)
        fprintf('\nMissing Python modules:\n');
        for i = 1:numel(missing_python)
            fprintf('  - %s\n', missing_python{i});
        end
        fprintf('\nTry these commands in the same Python environment used by MATLAB:\n');
        fprintf('  "%s" -m pip install numpy scipy\n', python_exe);
        fprintf('  "%s" -m pip install -e "%s"\n', python_exe, repo_root);
    end

    if ~isempty(missing_matlab)
        fprintf('\nMissing MATLAB / AToM functions:\n');
        for i = 1:numel(missing_matlab)
            fprintf('  - %s\n', missing_matlab{i});
        end
        fprintf('\nAToM appears not to be on the active MATLAB path.\n');
        fprintf('Add the AToM root folder manually, for example:\n');
        fprintf('  addpath(genpath(''C:\\path\\to\\AToM''))\n');
        fprintf('Then rerun setup_environment.\n');
    end
end

end

function tf = python_module_available(module_name)
    tf = true;
    try
        py.importlib.import_module(module_name);
    catch
        tf = false;
    end
end

function install_python_package(python_exe, label, cmd_tail, quiet_ok)
    if nargin < 4
        quiet_ok = false;
    end
    if strcmp(label, 'pip')
        cmd = sprintf('"%s" -m pip --version', python_exe);
        [status, ~] = system(cmd);
        if status == 0
            if ~quiet_ok
                fprintf('[OK] pip is available.\n');
            end
            return;
        end

        fprintf('[WARN] pip is not available. Attempting to bootstrap pip...\n');
        cmd = sprintf('"%s" %s', python_exe, cmd_tail);
        [status, out] = system(cmd);
        if status == 0
            fprintf('[OK] pip bootstrap succeeded.\n');
            fprintf('%s\n', out);
        else
            fprintf('[FAIL] pip bootstrap failed.\n');
            fprintf('%s\n', out);
        end
    end
end