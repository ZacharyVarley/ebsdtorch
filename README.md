# EBSDTorch

[![PyPI version](https://img.shields.io/pypi/v/ebsdtorch)](https://pypi.org/project/ebsdtorch/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ebsdtorch)](https://pypi.org/project/ebsdtorch/)
[![GitHub - License](https://img.shields.io/github/license/ZacharyVarley/ebsdtorch)](https://github.com/ZacharyVarley/ebsdtorch/blob/main/LICENSE)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/ebsdtorch)](https://pypi.org/project/ebsdtorch/)

PyTorch library for electron backscatter diffraction (EBSD)

**Warning: This library is in early development and is not yet stable.**

## Installation

First install PyTorch, then install via pip:

```bash
pip install ebsdtorch
```

## Documentation

Documentation is coming soon...

## Features (and TODOs)
- :white_check_mark: wide GPU support via PyTorch
- :white_check_mark: Uniform sampling on S2 & SO(3)
- :white_check_mark: Laue symmetry operations on S2 & SO(3)
- :white_check_mark: Rosca-Lambert square-circle equal area bijection
- :white_check_mark: dictionary indexing & PCA dictionary indexing
- :white_check_mark: Dynamic Quantization on CPU
- :white_check_mark: Wigner d matrix computation on the GPU
- :white_check_mark: Spherical Harmonics and SO(3) Cross Correlation
- :white_check_mark: Global FFT Pattern Center Search
- :white_check_mark: Monte Carlo BSE Simulation (currently slow)

- :white_large_square: Spherical indexing
- :white_large_square: Orientation coloring schemes
- :white_large_square: Misorientation coloring schemes
- :white_large_square: Quantization on GPU (PyTorch will soon support it)
- :white_large_square: Individual pattern centers
- :white_large_square: 6 DOF EBSD geometry fitting
- :white_large_square: Support for generic crystal unit cells
- :white_large_square: Dynamical scattering simulation