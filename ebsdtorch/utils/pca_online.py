"""

Online PCA Methods
-----------------------------

The quadratic memory dependence of the full covariance matrix is not tolerable
for large dimensionality. Therefore, we resort to well known top-k component
online PCA methods. The batch Oja update rule with a decaying learning rate and
implicit matrix Krasulina update are two well-studied methods for this purpose.
If you search "Online Batch Incremental PCA" etc. you will find a plethora of
irrelevant suggestions that veer off from the core mathematical proofs and
derivations.

See the following papers for more information:

Allen-Zhu, Zeyuan, and Yuanzhi Li. “First Efficient Convergence for Streaming
K-PCA: A Global, Gap-Free, and Near-Optimal Rate.” 2017 IEEE 58th Annual
Symposium on Foundations of Computer Science (FOCS), IEEE, 2017, pp. 487–92.
DOI.org (Crossref), https://doi.org/10.1109/FOCS.2017.51.

Tang, Cheng. “Exponentially Convergent Stochastic K-PCA without Variance
Reduction.” Advances in Neural Information Processing Systems, vol. 32, 2019.

Amid, Ehsan, and Manfred K. Warmuth. “An Implicit Form of Krasulina’s k-PCA
Update without the Orthonormality Constraint.” Proceedings of the AAAI
Conference on Artificial Intelligence, vol. 34, no. 04, Apr. 2020, pp. 3179–86.
DOI.org (Crossref), https://doi.org/10.1609/aaai.v34i04.5715.


Oja tremendously benefits from an exponentially decaying learning rate. That
will be implemented in the future and used as the default method due to QR
decomposition's complexity vs matrix inversion for high dimensions.

"""

import torch
from torch import nn, Tensor


class KrasulinaPCA(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_components: int,
        eta: float = 0.5,
        reg: float = 1e-7,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_components = n_components
        self.eta = eta

        # Initialize parameters
        self.register_buffer("C", torch.randn(n_features, n_components, dtype=dtype))
        self.register_buffer(
            "C_cross", torch.linalg.inv_ex(self.C.T @ self.C)[0] @ self.C.T
        )
        self.register_buffer("reg_mat", reg * torch.eye(n_components, dtype=dtype))

    def forward(self, x: Tensor) -> None:
        """Update PCA with new batch of data"""
        # Project data
        latent = self.C_cross @ x.T
        residual = x.T - self.C @ latent
        alpha = self.eta / (1.0 + self.eta * torch.norm(latent, dim=0))
        self.C.add_((alpha[None, :] * residual) @ latent.T)
        self.C_cross.copy_(torch.linalg.inv_ex(self.C.T @ self.C)[0] @ self.C.T)

    def get_components(self) -> Tensor:
        U, S, V = torch.svd(self.C_cross.T)
        return U.flip(1)

    def transform(self, x: Tensor) -> Tensor:
        return self.C_cross @ x.T

    def inverse_transform(self, z: Tensor) -> Tensor:
        return self.C @ z


class OjaPCA(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_components: int,
        eta: float = 0.5,
        dtype: torch.dtype = torch.float32,
        use_oja_plus: bool = False,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_components = n_components
        self.eta = eta
        self.use_oja_plus = use_oja_plus

        # Initialize parameters
        self.register_buffer("Q", torch.randn(n_features, n_components, dtype=dtype))
        self.register_buffer("step", torch.zeros(1, dtype=torch.int64))

        # For Oja++, we initialize columns gradually
        if self.use_oja_plus:
            self.register_buffer(
                "initialized_cols", torch.zeros(n_components, dtype=torch.bool)
            )
            self.register_buffer("next_col_to_init", torch.tensor(0, dtype=torch.int64))

    def forward(self, x: Tensor) -> None:
        """Update PCA with new batch of data using Oja's algorithm"""
        # Update then Orthonormalize Q_t using QR decomposition
        self.Q.copy_(torch.linalg.qr(self.Q + self.eta * (x.T @ (x @ self.Q)))[0])

        # Update step
        self.step.add_(1)

        # For Oja++, gradually initialize columns
        if self.use_oja_plus and self.next_col_to_init < self.n_components:
            if self.step % (self.n_components // 2) == 0:
                self.Q[:, self.next_col_to_init] = torch.randn(
                    self.n_features, dtype=self.Q.dtype
                )
                self.initialized_cols[self.next_col_to_init] = True
                self.next_col_to_init.add_(1)

    def get_components(self) -> Tensor:
        """Get orthogonalized components sorted by variance"""
        return self.Q

    def transform(self, x: Tensor) -> Tensor:
        """Project data onto components"""
        return x @ self.Q

    def inverse_transform(self, x: Tensor) -> Tensor:
        """Reconstruct data from projections"""
        return x @ self.Q.T
