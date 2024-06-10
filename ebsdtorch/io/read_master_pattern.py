import h5py as hf
import numpy as np
import torch
from ebsdtorch.ebsd.ebsd_master_patterns import MasterPattern


def read_master_pattern(file_path) -> MasterPattern:
    """
    Read the master pattern from a file.

    Args:
        :file_path (str): Path to the file containing the master pattern.

    Returns:
        :master_pattern_obj (MasterPattern): The master pattern object.

    """
    with hf.File(file_path, "r") as f:
        # load in (H, W, n_energies) square lambert projection Monte Carlo data
        e_accum = f["EMData/MCOpenCL/accum_e"][:].astype(np.float32)

        # load in the actual master patterns per energy
        # mLPNH = modified Lambert projection of North hemisphere
        # this will be (1, n_energies, H, W) shaped
        north_master_patterns = (
            f["EMData/EBSDmaster/mLPNH"][:].astype(np.float32).squeeze()
        )
        south_master_patterns = (
            f["EMData/EBSDmaster/mLPSH"][:].astype(np.float32).squeeze()
        )

        # load the space group number to have the Laue group
        space_group = f["CrystalData/SpaceGroupNumber"][0]

    # get (n_energies,) shape pdf by accumulating over the lambert projection
    probabilities = e_accum.sum(axis=(0, 1))
    # normalize the probabilities
    pdf = probabilities / probabilities.sum()

    # use the pdf to weight and sum the master patterns across energies
    master_pattern_NH = (pdf[:, None, None] * north_master_patterns).sum(axis=0)
    master_pattern_SH = (pdf[:, None, None] * south_master_patterns).sum(axis=0)

    # concatenate the master patterns along the first dimension
    # this make interpolation easier and faster
    master_pattern = np.concatenate([master_pattern_NH, master_pattern_SH], axis=0)

    # create the master pattern object
    master_pattern_obj = MasterPattern(
        torch.tensor(master_pattern), space_group=space_group
    )

    return master_pattern_obj
