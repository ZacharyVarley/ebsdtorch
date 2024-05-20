import torch
from torch import Tensor
from typing import Optional

from ebsdtorch.ebsd.projection import EBSDBasicProjector, EBSDDictionaryChunked
from ebsdtorch.ebsd.geometry import PointEBSDGeometry
from ebsdtorch.ebsd.knn import ChunkedKNN
from ebsdtorch.ebsd.pca import OnlineCovMatrix
from ebsdtorch.ebsd_dictionary_indexing.utils_progress_bar import progressbar


class DictionaryIndexer:
    """

    Class for indexing a dictionary of EBSD patterns.

    Args:
        geometry (PointEBSDGeometry): The EBSD geometry object.
        projector (EBSDBasicProjector): The EBSD projector object.


    Notes:
        It is rare that you reuse the same dictionary because the
        geometry and projector are usually different for each scan.
        So the dictionary projection is done in the indexing call.

    """

    def __init__(
        self,
        geometry: PointEBSDGeometry,
        projector: EBSDBasicProjector,
    ):
        self.geometry = geometry
        self.projector = projector

    def index_patterns_static(
        self,
        exp_pats: Tensor,
        binning: float = 1.0,
        ori_fz_n_samples: int = 100000,
        ori_batch_size: int = 4096,
        signal_mask: Optional[Tensor] = None,
        zero_mean: bool = True,
        metric: str = "angular",
        topk: int = 1,
        quantized_if_x86: bool = True,
    ):
        """
        Index the patterns using the dictionary.

        Args:
            patterns (Tensor): The EBSD pattern tensor, shape (..., H, W).
            binning (float): The binning factor for the dictionary.
            ori_fz_n_samples (int): The number of FZ orientations to sample.
            ori_batch_size (int): The batch size for pattern projection.
            signal_mask Optional[Tensor]: The signal mask tensor.
            zero_mean (bool): Whether to zero mean the patterns.
            quantize_if_cpu (bool): Whether to quantize the patterns if on CPU.

        Returns:
            Tensor: The indexed orientations.

        Notes:
            Can easily run out of memory if the patterns are larger than 128x128.

        """
        signal_shape = exp_pats.shape[:-2]
        n_patterns = torch.prod(torch.tensor(signal_shape)).item()

        # static dictionary can be precomputed
        sim_pats, so3_fz = self.projector.project_dictionary(
            n_target_so3_samples=ori_fz_n_samples,
            batch_size=ori_batch_size,
            binning=binning,
        )

        # flatten the patterns
        exp_pats = exp_pats.view(n_patterns, -1)
        sim_pats = sim_pats.view(len(sim_pats), -1)

        # mask the patterns
        if signal_mask is not None:
            exp_pats = exp_pats[:, signal_mask.view(-1)]

        # background subtract the experimental patterns
        exp_pats = exp_pats - torch.mean(exp_pats, dim=0, keepdim=True)

        # zero mean the patterns
        if zero_mean:
            exp_pats = exp_pats - torch.mean(exp_pats, dim=1, keepdim=True)
            sim_pats = sim_pats - torch.mean(sim_pats, dim=1, keepdim=True)

        # index the patterns
        indexer = ChunkedKNN(
            data_size=len(sim_pats),
            query_size=len(exp_pats),
            topk=topk,
            match_device=exp_pats.device,
            distance_metric=metric,
            match_dtype=exp_pats.dtype,
            quantized_if_x86=quantized_if_x86,
        )

        indexer.set_data_chunk(sim_pats)
        indexer.query_all(exp_pats)
        knn_indices, knn_dists = indexer.retrieve_topk()

        if topk == 1:
            return so3_fz[knn_indices].view(signal_shape + (4,))
        else:
            return so3_fz[knn_indices].view(signal_shape + (topk, 4))

    def index_patterns_dynamic(
        self,
        exp_pats: Tensor,
        binning: float = 1.0,
        ori_fz_n_samples: int = 100000,
        ori_batch_size: int = 4096,
        signal_mask: Optional[Tensor] = None,
        zero_mean: bool = True,
        metric: str = "angular",
        topk: int = 1,
        quantized_if_x86: bool = True,
    ):
        """
        Index the patterns using the dictionary.

        Args:
            patterns (Tensor): The EBSD pattern tensor, shape (..., H, W).
            binning (float): The binning factor for the dictionary.
            ori_fz_n_samples (int): The number of FZ orientations to sample.
            ori_batch_size (int): The batch size for pattern projection.
            signal_mask Optional[Tensor]: The signal mask tensor.
            zero_mean (bool): Whether to zero mean the patterns.
            quantize_if_cpu (bool): Whether to quantize the patterns if on CPU.

        Returns:
            Tensor: The indexed orientations.

        Notes:
            Can easily run out of memory if the patterns are larger than 128x128.

        """
        signal_shape = exp_pats.shape[:-2]
        n_patterns = torch.prod(torch.tensor(signal_shape)).item()

        # flatten the patterns
        exp_pats = exp_pats.view(n_patterns, -1)

        # mask the patterns
        if signal_mask is not None:
            exp_pats = exp_pats[:, signal_mask.view(-1)]

        # background subtract the experimental patterns
        exp_pats = exp_pats - torch.mean(exp_pats, dim=0, keepdim=True)

        # zero mean the patterns
        if zero_mean:
            exp_pats = exp_pats - torch.mean(exp_pats, dim=1, keepdim=True)

        ebsd_dictionary = EBSDDictionaryChunked(
            projector=self.projector,
            n_target_so3_samples=ori_fz_n_samples,
            batch_size=ori_batch_size,
            binning=binning,
        )

        # index the patterns
        indexer = ChunkedKNN(
            data_size=ebsd_dictionary.get_n_so3_samples(),
            query_size=len(exp_pats),
            topk=topk,
            match_device=exp_pats.device,
            distance_metric=metric,
            match_dtype=exp_pats.dtype,
            quantized_if_x86=quantized_if_x86,
        )

        # set the sync function based on the available device
        if exp_pats.device.type == "cuda":
            sync = torch.cuda.synchronize
        elif exp_pats.device.type == "mps":
            sync = torch.mps.synchronize
        elif exp_pats.device.type == "xla":
            sync = torch.xpu.synchronize
        else:
            sync = lambda: None

        pb_index = progressbar(ebsd_dictionary, prefix="DI Indexing ")
        for sim_pats in pb_index:
            # flatten the patterns
            sim_pats = sim_pats.view(len(sim_pats), -1)

            # zero mean sim patterns batch
            if zero_mean:
                sim_pats = sim_pats - torch.mean(sim_pats, dim=1, keepdim=True)

            # update indexing results with this batch of new sim patterns
            indexer.set_data_chunk(sim_pats)
            indexer.query_all(exp_pats)

            # sync so that progress bar updates
            sync()

        knn_indices, knn_dists = indexer.retrieve_topk()

        if topk == 1:
            return ebsd_dictionary.get_so3_fz()[knn_indices].view(signal_shape + (4,))
        else:
            return ebsd_dictionary.get_so3_fz()[knn_indices].view(
                signal_shape + (topk, 4)
            )

    def index_patterns_pca_dynamic(
        self,
        exp_pats: Tensor,
        binning: float = 1.0,
        ori_fz_n_samples_pca: int = 100000,
        ori_fz_n_samples_dict: int = 300000,
        pca_n_max_components: int = 1000,
        ori_batch_size_pca: int = 4096,
        ori_batch_size_dict: int = 32768,
        signal_mask: Optional[Tensor] = None,
        zero_mean: bool = True,
        metric: str = "angular",
        topk: int = 1,
        quantized_if_x86: bool = True,
    ):
        """
        Index the patterns using the dictionary.

        Args:
            patterns (Tensor): The EBSD pattern tensor, shape (..., H, W).
            binning (float): The binning factor for the dictionary.
            ori_fz_n_samples (int): The number of FZ orientations to sample.
            ori_batch_size (int): The batch size for pattern projection.
            signal_mask Optional[Tensor]: The signal mask tensor.
            zero_mean (bool): Whether to zero mean the patterns.
            quantize_if_cpu (bool): Whether to quantize the patterns if on CPU.

        Returns:
            Tensor: The indexed orientations.

        Notes:
            Can easily run out of memory if the patterns are larger than 128x128.

        """
        signal_shape = exp_pats.shape[:-2]
        n_patterns = torch.prod(torch.tensor(signal_shape)).item()

        # flatten the patterns
        exp_pats = exp_pats.view(n_patterns, -1)

        # mask the patterns
        if signal_mask is not None:
            exp_pats = exp_pats[:, signal_mask.view(-1)]

        # background subtract the experimental patterns
        exp_pats = exp_pats - torch.mean(exp_pats, dim=0, keepdim=True)

        # zero mean the patterns
        if zero_mean:
            exp_pats = exp_pats - torch.mean(exp_pats, dim=1, keepdim=True)

        # begin PCA computation
        ebsd_dictionary = EBSDDictionaryChunked(
            projector=self.projector,
            n_target_so3_samples=ori_fz_n_samples_pca,
            batch_size=ori_batch_size_pca,
            binning=binning,
        )

        # initialize the covariance matrix
        onlinecovmat = OnlineCovMatrix(
            n_features=exp_pats.shape[1],
            covmat_dtype=exp_pats.dtype,
            delta_dtype=exp_pats.dtype,
        ).to(exp_pats.device)

        # set the sync function based on the available device
        if exp_pats.device.type == "cuda":
            sync = torch.cuda.synchronize
        elif exp_pats.device.type == "mps":
            sync = torch.mps.synchronize
        elif exp_pats.device.type == "xla":
            sync = torch.xpu.synchronize
        else:
            sync = lambda: None

        # loop over the dictionary patterns and update the covariance matrix
        pb_pca = progressbar(ebsd_dictionary, prefix="PCA Covariance Matrix")
        for sim_pats in pb_pca:
            # flatten the patterns
            sim_pats = sim_pats.view(len(sim_pats), -1)

            # zero mean sim patterns batch
            if zero_mean:
                sim_pats = sim_pats - torch.mean(sim_pats, dim=1, keepdim=True)

            # update the covariance matrix
            onlinecovmat(sim_pats)

            # sync so that progress bar updates
            sync()

        # compute the PCA
        covmat = onlinecovmat.get_covmat()

        # find the eigenvalues and eigenvectors of the symmetric covariance matrix
        print("Computing Eigen Decomposition")
        eigen_decomp = torch.linalg.eigh(covmat)
        eigvals, eigvecs = eigen_decomp
        pca_matrix = eigvecs[:, -pca_n_max_components:].to(exp_pats.dtype)

        # project the exp_pats
        exp_pats_pca = torch.matmul(exp_pats, pca_matrix)

        # begin PCA dictionary indexing
        ebsd_dictionary = EBSDDictionaryChunked(
            projector=self.projector,
            n_target_so3_samples=ori_fz_n_samples_dict,
            batch_size=ori_batch_size_dict,
            binning=binning,
        )

        # index the patterns
        indexer = ChunkedKNN(
            data_size=ebsd_dictionary.get_n_so3_samples(),
            query_size=len(exp_pats_pca),
            topk=topk,
            match_device=exp_pats.device,
            distance_metric=metric,
            match_dtype=exp_pats.dtype,
            quantized_if_x86=quantized_if_x86,
        )

        pb_index = progressbar(ebsd_dictionary, prefix="PCA DI Indexing ")
        for sim_pats in pb_index:
            # flatten the patterns
            sim_pats = sim_pats.view(len(sim_pats), -1)

            # zero mean sim patterns batch
            if zero_mean:
                sim_pats = sim_pats - torch.mean(sim_pats, dim=1, keepdim=True)

            # project the sim patterns
            sim_pats_pca = torch.matmul(sim_pats, pca_matrix)

            # update indexing results with this batch of new sim patterns
            indexer.set_data_chunk(sim_pats_pca)
            indexer.query_all(exp_pats_pca)

            # sync so that progress bar updates
            sync()

        knn_indices, knn_dists = indexer.retrieve_topk()

        if topk == 1:
            return ebsd_dictionary.get_so3_fz()[knn_indices].view(signal_shape + (4,))
        else:
            return ebsd_dictionary.get_so3_fz()[knn_indices].view(
                signal_shape + (topk, 4)
            )


# # Example Usage
# # get the master pattern
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# import kikuchipy as kp
# from orix import plot
# from orix.vector import Vector3d

# mp = kp.data.nickel_ebsd_master_pattern_small(projection="lambert", hemisphere="both")
# ni = mp.phase
# pg_m3m = ni.point_group.laue

# # Orientation colors
# ckey_m3m = plot.IPFColorKeyTSL(ni.point_group, direction=Vector3d.zvector())

# # to torch tensor
# mLPNH = torch.from_numpy(mp.data[0, :, :]).to(torch.float32)
# mLPSH = torch.from_numpy(mp.data[1, :, :]).to(torch.float32)

# # normalize each master pattern to 0 to 1
# mLPNH = 255 * (mLPNH - torch.min(mLPNH)) / (torch.max(mLPNH) - torch.min(mLPNH))
# mLPSH = 255 * (mLPSH - torch.min(mLPSH)) / (torch.max(mLPSH) - torch.min(mLPSH))

# # put into (2, H, W) shape tensor
# master_pattern = torch.stack((mLPNH, mLPSH), dim=0).to(device)

# # create the geometry
# detector_shape = (60, 60)
# sample_y_tilt_deg = 70.0
# sample_x_tilt_deg = 0.0
# detector_tilt_deg = 0.0
# pattern_center_guess = (0.4221, 0.2179, 0.4954)
# n_rows, n_cols = detector_shape

# geometry = PointEBSDGeometry(
#     detector_shape=detector_shape,
#     sample_y_tilt_deg=sample_y_tilt_deg,
#     sample_x_tilt_deg=sample_x_tilt_deg,
#     detector_tilt_deg=detector_tilt_deg,
#     pattern_center_guess=pattern_center_guess,
# ).to(device)

# # create the projector
# projector = EBSDBasicProjector(
#     geometry=geometry, master_pattern=master_pattern, laue_group=11
# ).to(device)

# # project the dictionary
# n_so3_samples_pca = 300000
# n_so3_samples_dict = 300000
# batch_size = 4096 * 4
# binning = 1.0

# # for scan_id in range(1, 11):
# for scan_id in [
#     10,
# ]:

#     s = kp.data.ni_gain(number=scan_id, allow_download=True)
#     # s.remove_static_background()
#     # s.remove_dynamic_background()
#     H, W = 149, 200

#     exp_pats = torch.tensor(s.data, dtype=torch.float32).to(device)

#     from ebsdtorch.preprocessing.nlpar import nlpar
#     from ebsdtorch.preprocessing.clahe import clahe_grayscale

#     # # subtract the mean background pattern
#     # exp_pats = exp_pats - torch.mean(exp_pats, dim=(0, 1), keepdim=True)

#     def min_max_normalize_4d(x):
#         signal_shape = x.shape[:-2]
#         x_mins = torch.min(x.view(signal_shape[0], signal_shape[1], -1), dim=-1).values
#         x_maxs = torch.max(x.view(signal_shape[0], signal_shape[1], -1), dim=-1).values
#         return (x - x_mins[:, :, None, None]) / (1e-4 + x_maxs - x_mins)[
#             :, :, None, None
#         ]

#     for _ in range(10):
#         exp_pats = min_max_normalize_4d(exp_pats)
#         exp_pats = nlpar(exp_pats, k_rad=1, coeff=0.5, center_logit_bias=3.5)
#         print(f"min: {torch.min(exp_pats).item()}, max: {torch.max(exp_pats).item()}")

#     # exp_pats = nlpar(exp_pats, k_rad=3, coeff=0.375)

#     # apply clahe to the patterns
#     exp_pats = min_max_normalize_4d(exp_pats)
#     # exp_pats = clahe_grayscale(exp_pats)

#     indexer = DictionaryIndexer(geometry=geometry, projector=projector)

#     # indexed_orientations = indexer.index_patterns_static(
#     #     exp_pats=exp_pats,
#     #     binning=binning,
#     #     ori_fz_n_samples=n_so3_samples_dict,
#     #     ori_batch_size=batch_size,
#     #     signal_mask=None,
#     #     zero_mean=True,
#     #     metric="angular",
#     #     topk=1,
#     #     quantized_if_x86=True,
#     # )

#     indexed_orientations = indexer.index_patterns_dynamic(
#         exp_pats=exp_pats,
#         binning=binning,
#         ori_fz_n_samples=n_so3_samples_dict,
#         ori_batch_size=batch_size,
#         signal_mask=None,
#         zero_mean=True,
#         metric="angular",
#         topk=1,
#         quantized_if_x86=True,
#     )

#     # indexed_orientations = indexer.index_patterns_pca_dynamic(
#     #     exp_pats=exp_pats,
#     #     binning=binning,
#     #     ori_fz_n_samples_pca=n_so3_samples_pca,
#     #     ori_fz_n_samples_dict=n_so3_samples_dict,
#     #     pca_n_max_components=1000,
#     #     ori_batch_size_pca=batch_size,
#     #     ori_batch_size_dict=batch_size,
#     #     signal_mask=None,
#     #     zero_mean=True,
#     #     metric="angular",
#     #     topk=1,
#     #     quantized_if_x86=True,
#     # )

#     print(indexed_orientations.shape)

#     # plot the indexed orientations
#     import numpy as np
#     from orix import plot
#     from orix.quaternion import Orientation
#     from orix.vector import Vector3d
#     from plotly.express import imshow

#     # get the orientation colors
#     ckey_m3m = plot.IPFColorKeyTSL(pg_m3m, direction=Vector3d.zvector())

#     # get the indexed orientations
#     indexed_orientations = indexed_orientations.cpu().numpy()

#     # save rgb image
#     orientations = Orientation(indexed_orientations)
#     rgb = ckey_m3m.orientation2color(orientations)
#     rgb_byte = (rgb.reshape(s.data.shape[0], s.data.shape[1], 3) * 255).astype(np.uint8)

#     # plot the indexed orientations
#     fig = imshow(rgb_byte)
#     # set title
#     fig.update_layout(title_text=f"Scan {scan_id}")
#     fig.show()
