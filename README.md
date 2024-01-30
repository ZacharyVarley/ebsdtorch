# ebsdtorch

PyTorch-only Python library for analyzing electron backscatter diffraction (EBSD)
patterns. It is designed to be fast and easy to use.

## Installation

To install ebsdtorch, first install PyTorch, then run this command in your
terminal:

```bash pip install ebsdtorch ```

## Features (and TODOs)

- [:heavy_check_mark:] Uniform orientation sampling on the sphere and SO(3)
- [:heavy_check_mark:] Laue group operations on the sphere and SO(3)
- [:heavy_check_mark:] Modified square Lambert projection and inverse

- [:heavy_check_mark:] EBSD dictionary indexing (conventional pixel space)
- [:heavy_check_mark:] EBSD dictionary indexing (covariance matrix PCA)
- [ ] EBSD dictionary indexing (Halko randomized PCA)

- [:heavy_check_mark:] 8-bit Quantization on CPU for fast indexing
- [ ] 8-bit Quantization on GPU for (very) fast indexing
- [ ] Further reduced bit depth quantization (CPU or GPU)
- [:heavy_check_mark:] EBSD master pattern direct space convolution with detector annulus

- [:heavy_check_mark:] Spherical covariance matrix calculation
- [ ] Spherical covariance matrix interpolation onto detector

- [:heavy_check_mark:] pattern projection with average projection center
- [:heavy_check_mark:] pattern projection with individual projection centers
- [ ] pattern projection with single camera matrix

- [ ] pattern center fitting (conventional)
- [ ] geometry fitting (single camera matrix)

- [:heavy_check_mark:] Wigner D matrices
- [ ] spherical harmonics
- [ ] SO3 FFT for cross correlation / convolution
- [ ] EBSD master pattern blur via SO3 FFT (for BSE image simulation)

- [ ] Support for generic crystal unit cells
- [ ] Monte Carlo backscatter electron simulation
- [ ] Dynamical scattering simulation
