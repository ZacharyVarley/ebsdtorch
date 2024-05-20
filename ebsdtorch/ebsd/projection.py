from typing import Optional, Union
import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import Dataset

from ebsdtorch.ebsd_dictionary_indexing.utils_progress_bar import progressbar
from ebsdtorch.s2_and_so3.laue_fz_ori import sample_ori_fz_laue
from ebsdtorch.ebsd.geometry import PointEBSDGeometry
from ebsdtorch.s2_and_so3.sphere import rosca_lambert_side_by_side
from ebsdtorch.s2_and_so3.quaternions import qu_apply


class EBSDBasicProjector(Module):
    """
    Class for projecting reference signals on the sphere to the detector plane.

    Args:
        master_pattern (Tensor):
            The master pattern for the Northern Hemisphere. It should be stored
            via a modified square Lambert projection. Shape is (2, H, W). Where
            the [0, :, :] and [1, :, :] are the North / South hemispheres.
        space_group (int, optional):
            The space group number. Must be between 1 and 230 inclusive.
        point_group (str, optional):
            The point group symbol in Hermann-Mauguin notation.
        laue_group (int, str, optional):
            The Laue group number or symbol in Schönflies notation.

    Notes:
        -acceptable data types are bool, uint8, int8/16/32/64, bfloat16, float16/32/64
        -A space group, point group, or Laue group must be specified.

    """

    def __init__(
        self,
        geometry: PointEBSDGeometry,
        master_pattern: Tensor,
        space_group: Optional[int] = None,
        point_group: Optional[str] = None,
        laue_group: Optional[Union[str, int]] = None,
    ):
        super(EBSDBasicProjector, self).__init__()

        # the geometry is a tunable parameter
        self.geometry = geometry

        # the master pattern is not a tunable parameter so it is a buffer
        # to avoid multiply grid_sample calls, we concatenate the master patterns side by side
        master_pattern_cat = torch.cat(
            (master_pattern[0, :, :], master_pattern[1, :, :]), dim=-1
        )
        self.register_buffer("master_pattern", master_pattern_cat)

        # check that at least one of the space group, point group, or Laue group is specified
        if space_group is None and point_group is None and laue_group is None:
            raise ValueError(
                "At least one of the space group, point group, or Laue group must be specified."
            )
        # check that only one of the space group, point group, or Laue group is specified
        if (
            sum(
                [
                    space_group is not None,
                    point_group is not None,
                    laue_group is not None,
                ]
            )
            > 1
        ):
            raise ValueError(
                "Only one of the space group, point group, or Laue group should be specified."
            )
        if space_group is not None:
            if space_group < 1 or space_group > 230:
                raise ValueError("The space group must be between 1 and 230 inclusive.")
            if space_group > 206:
                self.laue_group = 11
            elif space_group > 193:
                self.laue_group = 10
            elif space_group > 176:
                self.laue_group = 9
            elif space_group > 167:
                self.laue_group = 8
            elif space_group > 155:
                self.laue_group = 7
            elif space_group > 142:
                self.laue_group = 6
            elif space_group > 88:
                self.laue_group = 5
            elif space_group > 74:
                self.laue_group = 4
            elif space_group > 15:
                self.laue_group = 3
            elif space_group > 2:
                self.laue_group = 2
            else:
                self.laue_group = 1
        if point_group is not None:
            if point_group in ["m3-m", "4-3m", "432"]:
                self.laue_group = 11
            elif point_group in ["m3-", "23"]:
                self.laue_group = 10
            elif point_group in ["6/mmm", "6-m2", "6mm", "622"]:
                self.laue_group = 9
            elif point_group in ["6/m", "6-", "6"]:
                self.laue_group = 8
            elif point_group in ["3-m", "3m", "32"]:
                self.laue_group = 7
            elif point_group in ["3-", "3"]:
                self.laue_group = 6
            elif point_group in ["4/mmm", "4-2m", "4mm", "422"]:
                self.laue_group = 5
            elif point_group in ["4/m", "4-", "4"]:
                self.laue_group = 4
            elif point_group in ["mmm", "mm2", "222"]:
                self.laue_group = 3
            elif point_group in ["2/m", "m", "2"]:
                self.laue_group = 2
            elif point_group in ["1-", "1"]:
                self.laue_group = 1
            else:
                raise ValueError(
                    f"The point group symbol is not recognized, as one of, m3-m, 4-3m, 432, m3-, 23, 6/mmm, 6-m2, 6mm, 622, 6/m, 6-, 6, 3-m, 3m, 32, 3-, 3, 4/mmm, 4-2m, 4mm, 422, 4/m, 4-, 4, mmm, mm2, 222, 2/m, m, 2, 1-, 1"
                )
        if laue_group is not None:
            self.laue_group = laue_group

    def _get_sample_ref_frame_coords(
        self,
        binning: float = 1.0,
        norm: bool = True,
    ) -> Tensor:
        """
        Get the coordinates of the detector pixels in the sample reference frame.

        Args:
            binning (float):
                The binning factor for the projection. Default is 1. The binning
            diffable (bool):
                Whether to return a differentiable projection. Default is False.
            norm (bool):
                Whether to normalize the directions. Default is True.
        """
        # generate the pixel coordinates using linspace
        i_index = (
            torch.linspace(
                0,
                1,
                round(self.geometry.detector_shape[0] / binning),
                device=self.master_pattern.device,
            )
            * self.geometry.detector_shape[0]
        )
        j_index = (
            torch.linspace(
                0,
                1,
                round(self.geometry.detector_shape[1] / binning),
                device=self.master_pattern.device,
            )
            * self.geometry.detector_shape[1]
        )
        ii, jj = torch.meshgrid(i_index, j_index, indexing="ij")

        # stack to be (H, W, 2)
        detector_pixel_coords = torch.stack((ii, jj), dim=-1)

        # project the detector pixel coordinates to the sample reference frame
        with torch.no_grad():
            sample_ref_frame_coords = self.geometry.transform(detector_pixel_coords)

        # normalize the directions from the origin of the sample reference frame to each pixel
        if norm:
            sample_ref_frame_coords /= sample_ref_frame_coords.norm(
                dim=-1, keepdim=True
            )

        return sample_ref_frame_coords

    def project_batch(
        self,
        ori_batch: Tensor,
        sample_ref_frame_coords: Tensor,
        mode: str = "bilinear",
        padding_mode: str = "zeros",
        align_corners: bool = True,
    ) -> Tensor:
        """
        Project a batch of patterns.

        Args:
            so3_samples_fz (Tensor):
                The SO(3) samples within the Laue group fundamental zone.
            sample_ref_frame_coords (Tensor):
                The coordinates of the detector pixels in the sample reference frame.
            batch_size (int):
                The batch size for the projection.
            diffable (bool):
                Whether to return a differentiable projection.
            mode (str):
                The interpolation mode for the grid_sample function. Default is bilinear.
            padding_mode (str):
                The padding mode for the grid_sample function. Default is zeros.
            align_corners (bool):
                Whether to align the corners of the grid_sample function. Default is True.
        """

        # rotate the directions according to the crystal orientations
        rotated_vectors = qu_apply(
            ori_batch[:, None, None, :], sample_ref_frame_coords[None, :, :, :]
        )

        # equal area mapping from sphere to square [-1, 1] x [-1, 1]
        coords_within_square = rosca_lambert_side_by_side(rotated_vectors)

        return torch.nn.functional.grid_sample(
            # master patterns are viewed as (1, 1, H_master, 2*W_master)
            self.master_pattern[None, None, ...],
            # coordinates are viewed as (1, n_ori, H_pat*W_pat, 2)
            coords_within_square.view(
                1,
                -1,
                sample_ref_frame_coords.shape[0] * sample_ref_frame_coords.shape[1],
                2,
            ),
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
        )[0, 0].view(
            ori_batch.shape[0],
            sample_ref_frame_coords.shape[0],
            sample_ref_frame_coords.shape[1],
        )

    def project_dictionary(
        self,
        n_target_so3_samples: int,
        batch_size: int,
        binning: float = 1.0,
    ) -> Tensor:
        """
        Project the signal onto the detector plane.

        Args:
            n_target_so3_samples (int):
                The number of target SO(3) samples to project (rejection sampling)
            postprocessing_binning (float):
                The binning factor for the projection. Default is 1. The binning
                factor can be less than 1 and higher resolution patterns than the
                detector size specified in the geometry can be generated.
            diffable (bool):
                Whether to return a differentiable projection. Default is False.
        """
        # generate the permuted SO(3) samples within the Laue group fundamental zone
        so3_samples_fz = sample_ori_fz_laue(
            self.laue_group, n_target_so3_samples, self.master_pattern.device
        )

        # get the coordinates of the detector pixels in the sample reference frame
        sample_ref_frame_coords = self._get_sample_ref_frame_coords(binning, True)

        # loop over the batches of orientations and project the patterns
        pb = progressbar(
            list(torch.split(so3_samples_fz, batch_size)),
            prefix="Dictionary Projection",
        )
        patterns = []
        for ori_batch in pb:
            patterns.append(self.project_batch(ori_batch, sample_ref_frame_coords))
        return torch.cat(patterns, dim=0), so3_samples_fz


class EBSDDictionaryChunked(Dataset):
    """

    This is an iterator class for chunking the dictionary projection. This is
    useful because the dictionary can easily be too large to fit into memory. We
    assume that the so3 samples can fit into memory, as each million quaternions
    will occupy 16 MB at 32-bit precision.

    Args:
        projector (EBSDBasicProjector):
            The projector object that will be used to project the dictionary.
        n_target_so3_samples (int):
            The number of target SO(3) samples to project (rejection sampling)
        batch_size (int):
            The batch size for the projection.
        binning (float):
            The binning factor for the projection. Default is 1. The binning
            factor can be less than 1 and higher resolution patterns than the
            detector size specified in the geometry can be generated.
        mode (str):
            The interpolation mode for the grid_sample function. Default is bilinear.
        padding_mode (str):
            The padding mode for the grid_sample function. Default is zeros.
        align_corners (bool):
            Whether to align the corners of the grid_sample function. Default is True.

    """

    def __init__(
        self,
        projector: EBSDBasicProjector,
        n_target_so3_samples: int,
        batch_size: int,
        binning: float = 1.0,
        mode: str = "bilinear",
        padding_mode: str = "zeros",
        align_corners: bool = True,
    ):
        self.projector = projector
        self.batch_size = batch_size
        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners

        self.so3_samples_fz = sample_ori_fz_laue(
            self.projector.laue_group,
            n_target_so3_samples,
            self.projector.master_pattern.device,
        )
        self.sample_ref_frame_coords = self.projector._get_sample_ref_frame_coords(
            binning, True
        )
        self.ori_batches = list(torch.split(self.so3_samples_fz, self.batch_size))

    def get_n_so3_samples(self):
        return len(self.so3_samples_fz)

    def get_so3_fz(self):
        return self.so3_samples_fz

    def __len__(self):
        return len(self.ori_batches)

    def __next__(self):
        for ori_batch in self.ori_batches:
            yield self.projector.project_batch(
                ori_batch,
                self.sample_ref_frame_coords,
                self.mode,
                self.padding_mode,
                self.align_corners,
            )
        raise StopIteration

    def __getitem__(self, idx):
        return self.projector.project_batch(
            self.ori_batches[idx],
            self.sample_ref_frame_coords,
            self.mode,
            self.padding_mode,
            self.align_corners,
        )
