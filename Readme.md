# Axi_aperture

Axi_aperture is a research code repository for axisymmetric aperture field modeling and optimization, with a workflow that combines Python-based aperture representations and MATLAB-based optimization scripts.

The repository is intended to support numerical experiments involving aperture-generated electromagnetic fields, aperture-to-VSW projections, local field operators, and constrained optimization problems for force and stiffness objectives. In practice, the example scripts use Python for the axisymmetric aperture model and MATLAB for the optimization and post-processing pipeline.

## What this repository contains

- `src/`: source code for the Python package.
- `example/`: MATLAB example scripts and environment setup helpers.
- `tests/`: tests for validating package functionality.
- `docs/`: supporting documentation and project notes.
- `pyproject.toml`: Python packaging configuration.

## What the code does

The code supports a workflow in which axisymmetric aperture fields are represented in Python and then coupled into MATLAB for electromagnetic optimization studies. A typical example script:

1. configures a Python environment from MATLAB,
2. loads the `axi_ap` package and its numerical dependencies,
3. constructs aperture geometries and polarization channels,
4. builds local field, force, stiffness, and aperture-to-VSW operators,
5. solves constrained optimization problems for aperture and VSW representations,
6. compares optimized results across formulations.

This makes the repository useful for reproducible studies of aperture-based field synthesis and optimization-driven electromagnetic design.

## Installation

### Python package

From the repository root, install the package in editable mode:

```bash
pip install -e .
```

If needed, also install the numerical dependencies explicitly:

```bash
pip install numpy scipy
```

### MATLAB dependencies

The MATLAB examples also require the external AToM toolbox to be available on the MATLAB path.

A helper script is expected in:

```matlab
example/setup_environment.m
```

This script can be used to check whether Python, `axi_ap`, and the required MATLAB-side dependencies are available before running the examples.

## Typical usage

From MATLAB:

```matlab
cd example
setup_environment
```

Then run the desired example script from the same folder.

## Notes

This GitHub repository was created in support of the publication associated with placeholder DOI: `PLACEHOLDER_DOI`.
