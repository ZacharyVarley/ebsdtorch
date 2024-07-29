from typing import List, Union
import torch
from torch import Tensor
from ebsdtorch.s2_and_so3.orientations import cu2qu
from ebsdtorch.s2_and_so3.quaternions import qu_prod, qu_apply
from ebsdtorch.s2_and_so3.laue_fz_ori import ori_to_fz_laue
from ebsdtorch.ebsd.master_pattern import MasterPattern
from ebsdtorch.ebsd.experiment_pats import ExperimentPatterns
from ebsdtorch.ebsd.geometry import EBSDGeometry
from ebsdtorch.utils.progressbar import progressbar


@torch.jit.script
def get_local_orientation_grid(
    semi_edge_in_degrees: float,
    kernel_radius_in_steps: int,
    axial_grid_dimension: int = 3,
) -> Tensor:
    """
    Get the local orientation grid using cubochoric coordinates.
    The grid will be a cube with 2*semi_edge_in_degrees side length
    and (2*kernel_radius + 1)^3 grid points.

    The cubochoric box with the identity orientation centered at the origin
    extends from -0.5 * (pi)^(2/3) to 0.5 * (pi)^(2/3) along each axis. The
    points on the outside surface are 180 degree rotations.

    A convient equal volume subgrid around the origin can be mapped to quaternions
    and used to explore the space of orientations around the initial guess via
    rotational composition.

    Args:
        semi_edge_in_degrees (float): The semi-edge in degrees.
        divisions_per_dimension (int): The number of divisions per dimension.
        biaxial (bool): If True, grid is 3 orthogonal planes. Points of form:
            [a, b, 0], [0, b, c], [a, 0, c]. For 3x3x3, 4 corners are missing.

    Returns:
        Tuple[Tensor, Tensor]: The local orientation grid in cu and qu form.

    """

    # get the cubochoric grid edge length
    semi_edge_cubochoric = semi_edge_in_degrees * torch.pi ** (2 / 3) / 360.0

    # make a meshgrid of cubochoric coordinates
    cu = torch.linspace(
        -semi_edge_cubochoric,
        semi_edge_cubochoric,
        2 * kernel_radius_in_steps + 1,
    )

    # get cartesian product meshgrid
    cu = torch.stack(torch.meshgrid(cu, cu, cu, indexing="ij"), dim=-1).reshape(-1, 3)

    if axial_grid_dimension == 3:
        pass
    elif axial_grid_dimension == 2:
        # biaxial grid: points require at least 1 zero
        mask = torch.any(cu == 0, dim=-1)
        cu = cu[mask]
    elif axial_grid_dimension == 1:
        # uniaxial grid: points require at least 2 zeros
        mask = torch.sum(cu == 0, dim=-1) >= 2
        cu = cu[mask]
    else:
        raise ValueError("axial_grid_dimension must be 1, 2, or 3.")
    return cu


def orientation_grid_refinement(
    master_patterns: Union[MasterPattern, List[MasterPattern]],
    geometry: EBSDGeometry,
    experiment_patterns: ExperimentPatterns,
    grid_semi_edge_in_degrees: float,
    batch_size: int,
    virtual_binning: int = 1,
    n_iter: int = 3,
    axial_grid_dimension: int = 3,
    kernel_radius_in_steps: int = 1,
    shrink_factor: float = 0.5,
    average_pattern_center: bool = True,
    match_dtype: torch.dtype = torch.float16,
) -> None:
    """
    Refine the orientation of the EBSD patterns.

    Args:
        master_patterns (Union[MasterPattern, List[MasterPattern]]): The master patterns.
        geometry (EBSDGeometry): The EBSD geometry.
        experiment_patterns (ExperimentPatterns): The experiment patterns.
        batch_size (int): The batch size.
        virtual_binning (int): The virtual binning.
        n_iter (int): The number of iterations.
        grid_semi_edge_in_degrees (float): The semi-edge of the grid in degrees.
        kernel_radius_in_steps (int): The kernel radius in steps.
        average_pattern_center (bool): Assume all patterns come from a point source.

    Returns:
        None. The experiment orientations are updated in place.

    """

    # check that the indexing results have been combined
    if not hasattr(experiment_patterns, "orientations"):
        raise ValueError("The experiment patterns must be indexed before refinement.")

    # create a dictionary object for each master pattern
    if isinstance(master_patterns, MasterPattern):
        master_patterns = [master_patterns]

    # get the coordinates of the detector pixels in the sample reference frame
    detector_coords = geometry.get_coords_sample_frame(
        binning=(virtual_binning, virtual_binning)
    ).view(-1, 3)

    if not average_pattern_center:
        # broadcast subtract the sample scan coordinates from the detector coords
        detector_coords = detector_coords - experiment_patterns.spatial_coords
        # uses an individual pattern for every position in the scan

    # normalize the detector coordinates to be unit vectors
    detector_coords = detector_coords / detector_coords.norm(dim=-1, keepdim=True)

    # get a list of indices into the experiment patterns for each phase
    phase_indices = experiment_patterns.get_indices_per_phase()

    for i, indices in enumerate(phase_indices):
        mp = master_patterns[i]
        pb = progressbar(
            list(torch.split(indices, batch_size)),
            prefix=f"REFINE MP {i+1:01d}/{len(master_patterns):01d} ",
        )

        # make a reference
        cu_grid = (
            get_local_orientation_grid(
                grid_semi_edge_in_degrees,
                kernel_radius_in_steps,
                axial_grid_dimension,
            )
            .view(1, -1, 3)
            .to(detector_coords.device)
        )

        for indices_batch in pb:
            # get the local orientation grid
            # (N_EXP_PATS, N_GRID_POINTS, 3)
            cu_grid_current = cu_grid.clone().repeat(len(indices_batch), 1, 1)

            # get the experiment patterns for this batch
            # (N_EXP_PATS, H, W)
            exp_pats = experiment_patterns.get_patterns(
                indices_batch, binning=virtual_binning
            )

            # reshape from (N_EXP_PATS, H, W) to (N_EXP_PATS, H*W)
            exp_pats = exp_pats.view(exp_pats.shape[0], -1)

            # subtract the mean from the patterns
            exp_pats = exp_pats - torch.mean(exp_pats, dim=-1, keepdim=True)

            # get the current orientations
            # shape (N_EXP_PATS, 4)
            qu_current = experiment_patterns.get_orientations(indices_batch)

            for _ in range(n_iter):
                # convert the cu_grid to quaternions
                qu_grid = cu2qu(cu_grid_current)

                # augment with the grid via broadcasted quaternion multiplication
                # shape (N_EXP_PATS, N_GRID_POINTS, 4)
                qu_augmented = qu_prod(qu_grid, qu_current[:, None, :])

                # make sure the quaternions are in the fundamental zone
                qu_augmented = ori_to_fz_laue(qu_augmented, mp.laue_group)

                # apply the qu_augmented to the detector coordinates
                # shape (N_EXP_PATS, N_GRID_POINTS = N_SIM_PATS, N_DETECTOR_PIXELS, 3)
                detector_coords_rotated = qu_apply(
                    qu_augmented[:, :, None, :], detector_coords[None, None, :, :]
                )

                # interpolate the master pattern
                # shape (N_EXP_PATS, N_SIM_PATS, N_DETECTOR_PIXELS)
                sim_pats = mp.interpolate(
                    detector_coords_rotated,
                    mode="bilinear",
                    align_corners=True,
                    normalize_coords=False,  # already normalized above
                    virtual_binning=virtual_binning,  # coarser grid requires blur of MP
                ).squeeze()

                # zero mean the sim_pats
                sim_pats = sim_pats - torch.mean(sim_pats, dim=-1, keepdim=True)

                # use einsum to do the dot product
                # (N_EXP_PATS, N_DETECTOR_PIXELS) & (N_EXP_PATS, N_SIM_PATS, N_DETECTOR_PIXELS)
                dot_products = torch.einsum(
                    "ij,ikj->ik",
                    exp_pats.to(match_dtype),
                    sim_pats.to(match_dtype),
                )

                # get the best dot product
                # shape (N_EXP_PATS,)
                _, best_indices = torch.max(dot_products, dim=-1)

                # update center of the grid
                qu_current = qu_augmented[
                    torch.arange(qu_current.shape[0]), best_indices
                ]

                # update the grid size is the center was the best
                cu_grid_current[
                    (best_indices == ((cu_grid_current.shape[1] - 1) / 2))
                ] *= shrink_factor

            # update the experiment patterns
            experiment_patterns.set_orientations(qu_current, indices_batch)


def orientation_grad_refinement(
    master_patterns: Union[MasterPattern, List[MasterPattern]],
    geometry: EBSDGeometry,
    experiment_patterns: ExperimentPatterns,
    batch_size: int,
    virtual_binning: int = 1,
    n_iter: int = 50,
    learning_rate: float = 0.001,
    average_pattern_center: bool = True,
    match_dtype: torch.dtype = torch.float16,
) -> None:
    """
    Refine the orientation of the EBSD patterns using gradient descent.

    Args:
        master_patterns (Union[MasterPattern, List[MasterPattern]]): The master patterns.
        geometry (EBSDGeometry): The EBSD geometry.
        experiment_patterns (ExperimentPatterns): The experiment patterns.
        batch_size (int): The batch size.
        virtual_binning (int): The virtual binning.
        n_iter (int): The number of iterations.
        grid_semi_edge_in_degrees (float): The semi-edge of the grid in degrees.
        kernel_radius_in_steps (int): The kernel radius in steps.
        average_pattern_center (bool): Assume all patterns come from a point source.

    Returns:
        None. The experiment orientations are updated in place.

    """

    # check that the indexing results have been combined
    if not hasattr(experiment_patterns, "orientations"):
        raise ValueError("The experiment patterns must be indexed before refinement.")

    # create a dictionary object for each master pattern
    if isinstance(master_patterns, MasterPattern):
        master_patterns = [master_patterns]

    # get the coordinates of the detector pixels in the sample reference frame
    detector_coords = geometry.get_coords_sample_frame(
        binning=(virtual_binning, virtual_binning)
    ).view(-1, 3)

    # normalize the detector coordinates to be unit vectors
    detector_coords = detector_coords / detector_coords.norm(dim=-1, keepdim=True)

    # get a list of indices into the experiment patterns for each phase
    phase_indices = experiment_patterns.get_indices_per_phase()

    for i, indices in enumerate(phase_indices):
        mp = master_patterns[i]
        pb = progressbar(
            list(torch.split(indices, batch_size)),
            prefix=f"REFINE MP {i+1:01d}/{len(master_patterns):01d} ",
        )
        for indices_batch in pb:
            # get the experiment patterns for this batch
            # (N_EXP_PATS, H, W)
            exp_pats = experiment_patterns.get_patterns(
                indices_batch, binning=virtual_binning
            )

            # reshape from (N_EXP_PATS, H, W) to (N_EXP_PATS, H*W)
            exp_pats = exp_pats.view(exp_pats.shape[0], -1)

            # subtract the mean from the patterns
            exp_pats = exp_pats - torch.mean(exp_pats, dim=-1, keepdim=True)

            # normalize the patterns before dot product
            exp_pats = exp_pats / exp_pats.norm(dim=-1, keepdim=True)

            # get the current orientations
            # shape (N_EXP_PATS, 4)
            qu_current = experiment_patterns.get_orientations(indices_batch)

            # make a differentiable tensor to update the orientations
            qu_current = qu_current.clone().detach().requires_grad_(True)

            # pass off to the optimizer
            optimizer = torch.optim.Adam([qu_current], lr=learning_rate)

            initial_dots = None

            for step in range(n_iter):
                # apply the state of the lattice to the detector coordinates
                detector_coords_rotated = qu_apply(
                    qu_current[:, None, :], detector_coords[None, :, :]
                )

                # interpolate the master pattern
                # shape (N_SIM_PATS, N_PIX)
                sim_pats = mp.interpolate(
                    detector_coords_rotated,
                    mode="bicubic",
                    align_corners=True,
                    normalize_coords=False,  # already normalized above
                    virtual_binning=virtual_binning,  # coarser grid requires blur of MP
                ).squeeze()

                # zero mean the sim_pats
                sim_pats = sim_pats - torch.mean(sim_pats, dim=-1, keepdim=True)

                # normalize the patterns before dot product
                sim_pats = sim_pats / sim_pats.norm(dim=-1, keepdim=True)

                # use einsum to do the dot product
                # (N_PATS, N_PIX) and (N_PATS, N_PIX) to (N_PATS,)
                dot_products = torch.einsum(
                    "ij,ij->i", exp_pats.to(match_dtype), sim_pats.to(match_dtype)
                )

                if step == 0:
                    initial_dots = -dot_products

                # backpropagate the dot products
                loss = -torch.mean(dot_products)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # make sure the quaternions are unit norm
                with torch.no_grad():
                    qu_current /= qu_current.norm(dim=-1, keepdim=True)

            # detach the quaternions from the graph
            qu_current = qu_current.detach()

            # make the quaternions have positive scalar part
            qu_current *= (qu_current[..., :1] >= 0).float() * 2 - 1

            # return quaternions to the fundamental zone
            qu_current = ori_to_fz_laue(qu_current, mp.laue_group)

            # update the experiment patterns
            # experiment_patterns.set_orientations(qu_current.detach(), indices_batch)
            # only update if the final loss is better than the initial loss
            updates = dot_products < initial_dots
            experiment_patterns.set_orientations(
                qu_current[updates].detach(),
                indices_batch[updates],
            )


# # plot the indexed orientations
# import numpy as np
# from orix import plot
# from orix.quaternion import Orientation
# from orix.vector import Vector3d
# from plotly.express import imshow
# import torch
# import kikuchipy as kp
# import numpy as np
# from ebsdtorch.ebsd.ebsd_master_patterns import MasterPattern
# from ebsdtorch.ebsd.ebsd_experiment_pats import ExperimentPatterns
# from ebsdtorch.ebsd.geometry import EBSDGeometry
# from ebsdtorch.ebsd.indexing import dictionary_index_orientations
# from ebsdtorch.preprocessing.radial_mask import get_radial_mask

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
# ).to(device)

# exp_pats = ExperimentPatterns(
#     exp_pats,
#     spatial_coords=coords,
# )

# # subtract background and do clahe
# exp_pats.standard_clean()
# # exp_pats.do_nlpar()

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
#     dictionary_resolution_degrees=0.5,
#     dictionary_chunk_size=4096 * 4,
#     signal_mask=mask,
#     virtual_binning=1,
#     experiment_chunk_size=4096 * 4,
#     match_dtype=torch.float16,
# )

# # plot the indexed orientations
# orientations = exp_pats.get_orientations().cpu().numpy()

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

# # refine the orientations
# orientation_grid_refinement(
#     master_patterns=mp,
#     geometry=geom,
#     experiment_patterns=exp_pats,
#     batch_size=4096,
#     virtual_binning=1,
#     n_iter=5,
#     grid_semi_edge_in_degrees=0.5,
#     kernel_radius_in_steps=1,
#     axial_grid_dimension=1,
#     average_pattern_center=True,
#     match_dtype=torch.float32,
# )

# # plot the indexed orientations
# orientations = exp_pats.get_orientations().cpu().numpy()

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


# # # refine the orientations
# # orientation_grad_refinement(
# #     master_patterns=mp,
# #     geometry=geom,
# #     experiment_patterns=exp_pats,
# #     batch_size=2048,
# #     virtual_binning=1,
# #     n_iter=50,
# #     learning_rate=0.0001,
# #     average_pattern_center=False,
# #     match_dtype=torch.float32,
# # )

# # # plot the indexed orientations
# # orientations = exp_pats.get_orientations().cpu().numpy()

# # # get the orientation colors
# # pg_m3m = kp_mp.phase.point_group.laue
# # ckey_m3m = plot.IPFColorKeyTSL(pg_m3m, direction=Vector3d.zvector())

# # # save rgb image
# # orientations = Orientation(orientations)
# # rgb = ckey_m3m.orientation2color(orientations)
# # rgb_byte = (rgb.reshape(149, 200, 3) * 255).astype(np.uint8)

# # # plot the indexed orientations
# # fig = imshow(rgb_byte)
# # # set title
# # fig.update_layout(title_text=f"Scan {1}")

# # fig.show()
