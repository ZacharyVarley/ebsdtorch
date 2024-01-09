This repo contains PyTorch implementations of Laue point group symmetries, orientation conversion code,
and the preliminary code for EBSD PCA decomposition via Wilford's batched update algorithm for an 
online exact calculation of covariance matrices.

WARNING: Until the package is released, there could be severe mistakes in the code and in the logic.

My immediate goals for this packages:

1) Fast EBSD dictionary indexing via compression / quantization

2) Human-readable fast implementations of SO3 harmonics and spherical indexing
