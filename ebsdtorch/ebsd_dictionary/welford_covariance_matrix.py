import torch
from torch import Tensor


@torch.jit.script
def update_covmat(
    current_covmat: Tensor,
    current_obs: Tensor,
    current_mean: Tensor,
    data_new: Tensor,
    delta_dtype: torch.dtype,
) -> None:
    """
    Update the covariance matrix and mean using Welford's online algorithm.

    Args:
        current_covmat: current covariance matrix
        current_obs: current number of observations
        current_mean: current mean
        data_new: new data to be included in the covariance matrix

    Returns:
        None
    """
    # compute the batch mean
    N = data_new.shape[0]
    batch_mean = torch.mean(data_new, dim=0, keepdim=True)

    # update the global mean
    new_mean = (current_mean * current_obs + batch_mean * N) / (current_obs + N)

    # compute the deltas
    delta = (data_new - current_mean).to(delta_dtype)
    delta_prime = (data_new - new_mean).to(delta_dtype)

    # update the running covariance matrix
    current_covmat += torch.einsum("ij,ik->jk", delta, delta_prime).to(
        current_covmat.dtype
    )

    # update the number of observations and mean
    current_obs += N
    current_mean.copy_(new_mean)


class OnlineCovMatrix(torch.nn.Module):
    """
    Online covariance matrix calculator
    """

    def __init__(
        self,
        n_features: int,
        covmat_dtype: torch.dtype = torch.float64,
        delta_dtype: torch.dtype = torch.float32,
    ):
        super(OnlineCovMatrix, self).__init__()
        self.n_features = n_features
        self.covmat_dtype = covmat_dtype
        self.delta_dtype = delta_dtype

        # Initialize the running mean
        self.register_buffer("mean", torch.zeros(1, n_features, dtype=covmat_dtype))

        # Initialize the running covariance matrix
        self.register_buffer(
            "covmat_aggregate",
            torch.zeros((n_features, n_features), dtype=covmat_dtype),
        )

        # Initialize the number of observations
        self.register_buffer("obs", torch.tensor([0.0], dtype=delta_dtype))

    def forward(self, x: Tensor):
        """
        Update the covariance matrix with new data

        Args:
            x: torch tensor of shape (B, n_features) containing the new data

        Returns:
            None
        """
        # update the covariance matrix
        update_covmat(self.covmat_aggregate, self.obs, self.mean, x, self.delta_dtype)

    def get_covmat(self):
        """
        Get the covariance matrix

        Returns:
            torch tensor of shape (n_features, n_features) containing the covariance matrix
        """
        return self.covmat_aggregate / (self.obs - 1)
