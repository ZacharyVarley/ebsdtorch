from ebsdtorch.ebsd.geometry import EBSDGeometry
from ebsdtorch.ebsd.master_pattern import MasterPattern
from ebsdtorch.ebsd.experiment_pats import ExperimentPatterns
from ebsdtorch.ebsd.indexing import dictionary_index_orientations
import torch
import pytest


@pytest.fixture
def test_ebsd_di():
    geom = EBSDGeometry(
        detector_shape=(5, 5),
        proj_center=(0.4221, 0.2179, 0.4954),
    )

    # create the master pattern
    mp = MasterPattern(
        torch.randn((10, 10), dtype=torch.float32),
        laue_group=11,
    )

    # create the experiment patterns object
    exp_pats = ExperimentPatterns(
        torch.randn((2, 5, 5), dtype=torch.float32),
    )

    return geom, mp, exp_pats


def test_ebsd_di_ni(test_ebsd_di):
    geom, mp, exp_pats = test_ebsd_di

    # index the orientations
    dictionary_index_orientations(
        mp,
        geom,
        exp_pats,
        dictionary_resolution_degrees=10.0,
        dictionary_chunk_size=4096,
        signal_mask=None,
        virtual_binning=1,
        experiment_chunk_size=4096,
        match_dtype=torch.float16,
    )

    orientations = exp_pats.get_orientations().cpu().numpy()

    assert orientations.shape == (2, 4)
