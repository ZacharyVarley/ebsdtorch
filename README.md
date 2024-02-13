# EBSDTorch

[![PyPI version](https://img.shields.io/pypi/v/ebsdtorch)](https://pypi.org/project/ebsdtorch/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ebsdtorch)](https://pypi.org/project/ebsdtorch/)
[![GitHub - License](https://img.shields.io/github/license/ZacharyVarley/ebsdtorch)](https://github.com/ZacharyVarley/ebsdtorch/blob/main/LICENSE)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/ebsdtorch)](https://pypi.org/project/ebsdtorch/)

PyTorch-only library for electron backscatter diffraction (EBSD)

## Installation

To install EBSDTorch, first install PyTorch, then run this command in your terminal:

```bash
pip install ebsdtorch
```

## Features (and TODOs)

- :white_check_mark: wide GPU support via PyTorch device abstraction & backends

- :white_check_mark: Uniform sampling on sphere / SO(3)
- :white_check_mark: Laue symmetry operations on sphere / SO(3)
- :white_check_mark: Modified square Lambert projection and inverse

- :white_check_mark: dictionary indexing (conventional pixel space)
- :white_check_mark: dictionary indexing (covariance matrix PCA)
- :white_large_square: dictionary indexing (Halko randomized PCA)

- :white_check_mark: 8-bit Quantization on CPU for fast indexing
- :white_large_square: 8-bit Quantization on GPU for (very) fast indexing
- :white_large_square: Further reduced bit depth quantization (CPU or GPU)
- :white_check_mark: EBSD master pattern direct space convolution with detector annulus

- :white_check_mark: Spherical covariance matrix calculation
- :white_large_square: Spherical covariance matrix interpolation onto detector

- :white_check_mark: pattern projection with average projection center
- :white_check_mark: pattern projection with individual projection centers
- :white_large_square: pattern projection with single camera matrix

- :white_large_square: pattern center fitting (conventional)
- :white_large_square: geometry fitting (single camera matrix)

- :white_check_mark: Wigner D matrices
- :white_large_square: spherical harmonics
- :white_large_square: SO3 FFT for cross correlation / convolution
- :white_large_square: EBSD master pattern blur via SO3 FFT (for BSE image simulation)

- :white_large_square: Support for generic crystal unit cells
- :white_large_square: Monte Carlo backscatter electron simulation
- :white_large_square: Dynamical scattering simulation
