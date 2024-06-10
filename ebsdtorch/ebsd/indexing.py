from torch import Tensor
import torch
from ebsdtorch.ebsd.ebsd_master_patterns import MasterPattern
from ebsdtorch.ebsd.ebsd_experiment_pats import ExperimentPatterns
from ebsdtorch.ebsd.geometry import EBSDGeometry
from typing import Optional, Union, List
from ebsdtorch.s2_and_so3.laue_fz_ori import sample_ori_fz_laue_angle
from ebsdtorch.s2_and_so3.quaternions import qu_apply
from ebsdtorch.preprocessing.radial_mask import get_radial_mask
from ebsdtorch.utils.knn import ChunkedKNN
from ebsdtorch.utils.progressbar import progressbar
from ebsdtorch.utils.pca import OnlineCovMatrix


def dictionary_index_orientations(
    master_patterns: Union[MasterPattern, List[MasterPattern]],
    geometry: EBSDGeometry,
    experiment_patterns: ExperimentPatterns,
    signal_mask: Optional[Tensor] = None,
    subtract_exp_pat_mean: bool = True,
    experiment_chunk_size: int = 256,
    dictionary_resolution_degrees: float = 0.5,
    dictionary_chunk_size: int = 256,
    virtual_binning: int = 1,
    top_k_matches: int = 1,
    distance_metric: str = "angular",
    match_dtype: torch.dtype = torch.float16,
    quantized_if_x86: bool = True,
    average_pattern_center: bool = True,
) -> None:
    """
    Index the EBSD pattern orientations using dictionary indexing.

    Args:
        master_patterns (Union[MasterPattern, List[MasterPattern]]): The master patterns.
        geometry (EBSDGeometry): The EBSD geometry object.
        experiment_patterns (ExperimentPatterns): The experiment patterns.
        signal_mask (Optional[Tensor]): The signal mask.
        subtract_exp_pat_mean (bool): Whether to zero mean each experimental patter.
        experiment_chunk_size (int): The size of the experiment pattern chunks.
        dictionary_resolution_degrees (float): The resolution of the dictionary in degrees.
        dictionary_chunk_size (int): The size of the dictionary chunks.
        virtual_binning (int): The virtual binning factor.
        top_k_matches (int): The number of top matches to retrieve.
        n_pca_components (int): The number of PCA components to use (if 0, PCA is not used)
        distance_metric (str): The distance metric to use.
        match_dtype (torch.dtype): The data type of the matches.
        quantized_if_x86 (bool): Whether to quantize the matches if running on x86.
        average_pattern_center (bool): Whether to consider all scan points as at the origin.

    Returns:
        Experiment patterns with indexed orientations.

    """

    # create a dictionary object for each master pattern
    if isinstance(master_patterns, MasterPattern):
        master_patterns = [master_patterns]

    # we could save some compute by only computing the cosine vectors once
    # per unique Laue group ID and reuse those coordinates for all master patterns
    # with the same Laue group ID. But more than a few phases in a sample is rare.
    for i, mp in enumerate(master_patterns):
        # get an orientation dictionary
        ori_tensor = sample_ori_fz_laue_angle(
            laue_id=mp.laue_group,
            angular_resolution_deg=dictionary_resolution_degrees,
            device=mp.master_pattern.device,
        )

        # make an object to do the pattern comparisons
        knn = ChunkedKNN(
            data_size=len(ori_tensor),
            query_size=experiment_patterns.n_patterns,
            topk=top_k_matches,
            match_device=mp.master_pattern.device,
            distance_metric=distance_metric,
            match_dtype=match_dtype,
            quantized_via_ao=quantized_if_x86,
        )

        # get a helper function to sync devices
        if experiment_patterns.patterns.device.type == "cuda":
            sync = torch.cuda.synchronize
        elif experiment_patterns.patterns.device.type == "mps":
            sync = torch.mps.synchronize
        elif (
            experiment_patterns.patterns.device.type == "xpu"
            or experiment_patterns.patterns.device.type == "xla"
        ):
            sync = torch.xpu.synchronize
        else:
            sync = lambda: None

        # get the coordinates of the detector pixels in the sample reference frame
        detector_coords = geometry.get_coords_sample_frame(
            binning=(virtual_binning, virtual_binning)
        )

        if not average_pattern_center:
            # broadcast subtract the sample scan coordinates from the detector coords
            detector_coords = detector_coords - experiment_patterns.spatial_coords
            # uses an individual pattern for every position in the scan

        # normalize the detector coordinates to be unit vectors
        detector_coords = detector_coords / detector_coords.norm(dim=-1, keepdim=True)

        pb = progressbar(
            list(torch.split(ori_tensor, dictionary_chunk_size)),
            prefix=f"INDX MP {i+1:01d}/{len(master_patterns):01d} ",
        )

        # for ori_batch in list(torch.split(ori_tensor, dictionary_batch_size)):
        for ori_batch in pb:
            # use orientations to rotate the rays to detector pixel positions
            # (n_ori, 4) -> (n_ori, 1, 4) and (H*W, 3) -> (1, H*W, 3) for broadcasting
            batch_rotated_coords = qu_apply(
                ori_batch[:, None, :],
                detector_coords[None, ...],
            )

            # interpolate the master pattern at the rotated coordinates
            simulated_patterns = mp.interpolate(
                batch_rotated_coords,
                mode="bilinear",
                align_corners=True,
                normalize_coords=False,  # already normalized above
                virtual_binning=virtual_binning,  # coarser grid requires blur of MP
            ).squeeze()

            # flatten simulated patterns to (n_ori, H*W)
            simulated_patterns = simulated_patterns.view(len(ori_batch), -1)

            # apply the signal mask if provided
            if signal_mask is not None:
                simulated_patterns = simulated_patterns[:, signal_mask.flatten()]

            # must remove mean from each simulated pattern
            simulated_patterns = simulated_patterns - torch.mean(
                simulated_patterns,
                dim=-1,
                keepdim=True,
            )

            # set the data for the KNN object
            knn.set_data_chunk(simulated_patterns)
            if experiment_chunk_size > experiment_patterns.n_patterns:
                exp_pats = experiment_patterns.get_patterns(
                    torch.arange(experiment_patterns.n_patterns),
                    binning=virtual_binning,
                ).view(experiment_patterns.n_patterns, -1)

                # query all the experiment patterns
                # first apply the signal mask if provided
                if signal_mask is not None:
                    exp_pats = exp_pats[:, signal_mask.flatten()]

                # subtract the mean from each pattern and query the KNN object
                if subtract_exp_pat_mean:
                    exp_pats = exp_pats - torch.mean(exp_pats, dim=-1, keepdim=True)
                knn.query_all(exp_pats)

                # synchronize the device for the progress bar
                sync()
            else:
                # loop over the experiment patterns in chunks and feed them to the KNN object
                for exp_pat_batch_indices in list(
                    torch.split(
                        torch.arange(experiment_patterns.n_patterns),
                        experiment_chunk_size,
                    )
                ):
                    # get a chunk of experiment patterns
                    query_chunk = experiment_patterns.get_patterns(
                        exp_pat_batch_indices,
                        binning=virtual_binning,
                    ).view(len(exp_pat_batch_indices), -1)

                    # apply the signal mask if provided
                    if signal_mask is not None:
                        query_chunk = query_chunk[:, signal_mask.flatten()]

                    # subtract the mean
                    query_chunk = query_chunk - torch.mean(
                        query_chunk, dim=-1, keepdim=True
                    )

                    # query the KNN object
                    knn.query_chunk(
                        query_chunk.view(len(query_chunk), -1),
                        query_start=exp_pat_batch_indices[0],
                    )

                    # synchronize the device for the progress bar
                    sync()

        # get the matches and distances
        matches, metric_values = knn.retrieve_topk()

        # set the matches and distances in the experiment patterns object
        experiment_patterns.set_raw_indexing_results(
            ori_tensor[matches],
            metric_values,
            phase_id=i,
        )

    # combine the indexing results
    experiment_patterns.combine_indexing_results(
        higher_is_better=(distance_metric == "angular")
    )


def pca_dictionary_index_orientations(
    master_patterns: Union[MasterPattern, List[MasterPattern]],
    geometry: EBSDGeometry,
    experiment_patterns: ExperimentPatterns,
    signal_mask: Optional[Tensor] = None,
    subtract_exp_pat_mean: bool = True,
    experiment_chunk_size: int = 256,
    dictionary_resolution_degrees: float = 0.5,
    dictionary_chunk_size: int = 256,
    virtual_binning: int = 1,
    top_k_matches: int = 1,
    n_pca_components: int = 0,
    distance_metric: str = "angular",
    match_dtype: torch.dtype = torch.float16,
    quantized_via_ao: bool = True,
    average_pattern_center: bool = True,
) -> None:
    """
    Index the EBSD pattern orientations using dictionary indexing.

    Args:
        master_patterns (Union[MasterPattern, List[MasterPattern]]): The master patterns.
        geometry (EBSDGeometry): The EBSD geometry object.
        experiment_patterns (ExperimentPatterns): The experiment patterns.
        signal_mask (Optional[Tensor]): The signal mask.
        subtract_exp_pat_mean (bool): Whether to zero mean each experimental patter.
        experiment_chunk_size (int): The size of the experiment pattern chunks.
        dictionary_resolution_degrees (float): The resolution of the dictionary in degrees.
        dictionary_chunk_size (int): The size of the dictionary chunks.
        virtual_binning (int): The virtual binning factor.
        top_k_matches (int): The number of top matches to retrieve.
        n_pca_components (int): The number of PCA components to use (if 0, PCA is not used)
        distance_metric (str): The distance metric to use.
        match_dtype (torch.dtype): The data type of the matches.
        quantized_if_x86 (bool): Whether to quantize the matches if running on x86.
        average_pattern_center (bool): Whether to consider all scan points as at the origin.

    Returns:
        Experiment patterns with indexed orientations.

    """

    # create a dictionary object for each master pattern
    if isinstance(master_patterns, MasterPattern):
        master_patterns = [master_patterns]

    # we could save some compute by only computing the cosine vectors once
    # per unique Laue group ID and reuse those coordinates for all master patterns
    # with the same Laue group ID. But more than a few phases in a sample is rare.
    for i, mp in enumerate(master_patterns):
        # get an orientation dictionary
        ori_tensor = sample_ori_fz_laue_angle(
            laue_id=mp.laue_group,
            angular_resolution_deg=dictionary_resolution_degrees,
            device=mp.master_pattern.device,
        )

        # make an object to do the pattern comparisons
        knn = ChunkedKNN(
            data_size=len(ori_tensor),
            query_size=experiment_patterns.n_patterns,
            topk=top_k_matches,
            match_device=mp.master_pattern.device,
            distance_metric=distance_metric,
            match_dtype=match_dtype,
            quantized_via_ao=quantized_via_ao,
        )

        # get the dimensionality of each pattern
        if signal_mask is not None:
            n_pixels = signal_mask.sum().item()
        else:
            n_pixels = experiment_patterns.n_pixels

        # make an object to do the PCA
        pcacovmat = OnlineCovMatrix(
            n_features=n_pixels,
        ).to(mp.master_pattern.device)

        # get a helper function to sync devices
        if experiment_patterns.patterns.device.type == "cuda":
            sync = torch.cuda.synchronize
        elif experiment_patterns.patterns.device.type == "mps":
            sync = torch.mps.synchronize
        elif (
            experiment_patterns.patterns.device.type == "xpu"
            or experiment_patterns.patterns.device.type == "xla"
        ):
            sync = torch.xpu.synchronize
        else:
            sync = lambda: None

        pb = progressbar(
            list(torch.split(ori_tensor, dictionary_chunk_size)),
            prefix=f" PCA MP {i+1:01d}/{len(master_patterns):01d} ",
        )

        # iterate over the dictionary in chunks
        for ori_batch in pb:
            # use orientations to rotate the rays to detector pixel positions
            # (n_ori, 4) -> (n_ori, 1, 4) and (H*W, 3) -> (1, H*W, 3) for broadcasting
            batch_rotated_coords = qu_apply(
                ori_batch[:, None, :],
                geometry.get_coords_sample_frame(
                    binning=(virtual_binning, virtual_binning)
                )[None, ...],
            )

            # interpolate the master pattern at the rotated coordinates
            simulated_patterns = mp.interpolate(
                batch_rotated_coords,
                mode="bilinear",
                align_corners=True,
                normalize_coords=True,  # not already normalized above
                virtual_binning=virtual_binning,  # coarser grid requires blur of MP
            ).squeeze()

            # flatten simulated patterns to (n_ori, H*W)
            simulated_patterns = simulated_patterns.view(len(ori_batch), -1)

            # apply the signal mask if provided
            if signal_mask is not None:
                simulated_patterns = simulated_patterns[:, signal_mask.flatten()]

            # must remove mean from each simulated pattern
            simulated_patterns = simulated_patterns - torch.mean(
                simulated_patterns,
                dim=-1,
                keepdim=True,
            )

            # update the PCA object
            pcacovmat(simulated_patterns)

        # get the eigenvectors and eigenvalues
        eigenvectors = pcacovmat.get_eigenvectors()

        # trim the eigenvectors to the number of components
        # they are returned in ascending order of eigenvalue
        pca_matrix = eigenvectors[:, -n_pca_components:]

        # project the dictionary onto the PCA components
        pb = progressbar(
            list(torch.split(ori_tensor, dictionary_chunk_size)),
            prefix=f"INDX MP {i+1:01d}/{len(master_patterns):01d} ",
        )

        # iterate over the dictionary in chunks
        for ori_batch in pb:
            # use orientations to rotate the rays to detector pixel positions
            # (n_ori, 4) -> (n_ori, 1, 4) and (H*W, 3) -> (1, H*W, 3) for broadcasting
            batch_rotated_coords = qu_apply(
                ori_batch[:, None, :],
                geometry.get_coords_sample_frame(
                    binning=(virtual_binning, virtual_binning)
                )[None, ...],
            )

            # interpolate the master pattern at the rotated coordinates
            simulated_patterns = mp.interpolate(
                batch_rotated_coords,
                mode="bilinear",
                align_corners=True,
                normalize_coords=True,  # not already normalized above
                virtual_binning=virtual_binning,  # coarser grid requires blur of MP
            ).squeeze()

            # flatten simulated patterns to (n_ori, H*W)
            simulated_patterns = simulated_patterns.view(len(ori_batch), -1)

            # apply the signal mask if provided
            if signal_mask is not None:
                simulated_patterns = simulated_patterns[:, signal_mask.flatten()]

            # must remove mean from each simulated pattern
            simulated_patterns = simulated_patterns - torch.mean(
                simulated_patterns,
                dim=-1,
                keepdim=True,
            )

            # project the simulated patterns onto the PCA components
            simulated_patterns_pca = torch.matmul(simulated_patterns, pca_matrix)

            # set the data for the KNN object
            knn.set_data_chunk(simulated_patterns_pca)

            # # retrieve and project the experiment patterns
            if experiment_chunk_size > experiment_patterns.n_patterns:
                exp_pats = experiment_patterns.patterns.view(
                    experiment_patterns.n_patterns, -1
                )
                # query all the experiment patterns
                # first apply the signal mask if provided
                if signal_mask is not None:
                    exp_pats = exp_pats[:, signal_mask.flatten()]

                # subtract the mean from each pattern and query the KNN object
                if subtract_exp_pat_mean:
                    exp_pats = exp_pats - torch.mean(exp_pats, dim=-1, keepdim=True)

                # PCA projection
                exp_pats = torch.matmul(exp_pats, pca_matrix)

                # query the KNN object
                knn.query_all(exp_pats)

                # synchronize the device for the progress bar
                sync()
            else:
                # loop over the experiment patterns in chunks and feed them to the KNN object
                for exp_pat_batch_indices in list(
                    torch.split(
                        torch.arange(experiment_patterns.n_patterns),
                        experiment_chunk_size,
                    )
                ):
                    # get a chunk of experiment patterns
                    query_chunk = experiment_patterns.get_patterns(
                        exp_pat_batch_indices,
                        binning=virtual_binning,
                    ).view(len(exp_pat_batch_indices), -1)

                    # apply the signal mask if provided
                    if signal_mask is not None:
                        query_chunk = query_chunk[:, signal_mask.flatten()]

                    # subtract the mean
                    query_chunk = query_chunk - torch.mean(
                        query_chunk, dim=-1, keepdim=True
                    )

                    # project the query chunk onto the PCA components
                    query_chunk = torch.matmul(query_chunk, pca_matrix)

                    # query the KNN object
                    knn.query_chunk(
                        query_chunk.view(len(query_chunk), -1),
                        query_start=exp_pat_batch_indices[0],
                    )

                    # synchronize the device for the progress bar
                    sync()

        # get the matches and distances
        matches, metric_values = knn.retrieve_topk()

        # set the matches and distances in the experiment patterns object
        experiment_patterns.set_raw_indexing_results(
            ori_tensor[matches],
            metric_values,
            phase_id=i,
        )

    # combine the indexing results
    experiment_patterns.combine_indexing_results(
        higher_is_better=(distance_metric == "angular")
    )


# import torch
# import kikuchipy as kp
# import numpy as np

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# geom = EBSDGeometry(
#     detector_shape=(60, 60),
#     proj_center=(0.4221, 0.2179, 0.4954),
# ).to(device)

# # create the master pattern
# kp_mp = kp.data.nickel_ebsd_master_pattern_small(
#     projection="lambert", hemisphere="both"
# )
# mp_nh = torch.from_numpy(kp_mp.data[0, :, :].astype(np.float32)).to(torch.float32)
# mp_sh = torch.from_numpy(kp_mp.data[1, :, :].astype(np.float32)).to(torch.float32)
# master_pattern = torch.concat((mp_nh, mp_sh), dim=-1)
# mp = MasterPattern(
#     master_pattern,
#     laue_group=11,
# ).to(device)
# mp.normalize(norm_type="minmax")

# # create the experiment patterns object
# exp_pats = (
#     torch.tensor(kp.data.ni_gain(number=1, allow_download=True).data)
#     .to(device)
#     .to(torch.float32)
# )
# coords = torch.stack(
#     torch.meshgrid(
#         torch.arange(exp_pats.shape[0]),
#         torch.arange(exp_pats.shape[1]),
#         indexing="ij",
#     ),
#     dim=-1,
# )
# exp_pats = ExperimentPatterns(
#     exp_pats,
#     spatial_coords=coords,
# )

# # subtract background and do clahe
# exp_pats.standard_clean()
# exp_pats.do_nlpar()

# # get radial mask
# mask = get_radial_mask(
#     shape=(60, 60),
# ).to(device)
# # mask = None

# # index the orientations
# dictionary_index_orientations(
#     mp,
#     geom,
#     exp_pats,
#     dictionary_resolution_degrees=0.85,
#     dictionary_chunk_size=4096,
#     signal_mask=mask,
#     virtual_binning=1,
#     experiment_chunk_size=4096,
#     match_dtype=torch.float16,
# )

# # # index the orientations
# # pca_dictionary_index_orientations(
# #     mp,
# #     geom,
# #     exp_pats,
# #     dictionary_resolution_degrees=0.85,
# #     dictionary_chunk_size=4096,
# #     signal_mask=mask,
# #     virtual_binning=1,
# #     experiment_chunk_size=4096,
# #     match_dtype=torch.float16,
# #     n_pca_components=1200,
# # )


# orientations = exp_pats.get_orientations().cpu().numpy()

# # plot the indexed orientations
# import numpy as np
# from orix import plot
# from orix.quaternion import Orientation
# from orix.vector import Vector3d
# from plotly.express import imshow

# # get the orientation colors
# pg_m3m = kp_mp.phase.point_group.laue
# ckey_m3m = plot.IPFColorKeyTSL(pg_m3m, direction=Vector3d.zvector())

# # save rgb image
# orientations = Orientation(orientations)
# rgb = ckey_m3m.orientation2color(orientations)
# rgb_byte = (rgb.reshape(149, 200, 3) * 255).astype(np.uint8)

# # plot the indexed orientations
# fig = imshow(rgb_byte)
# # set title
# fig.update_layout(title_text=f"Scan {1}")

# fig.show()
