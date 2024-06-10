from ebsdtorch.ebsd.geometry import EBSDGeometry
from ebsdtorch.ebsd.ebsd_master_patterns import MasterPattern
from ebsdtorch.ebsd.ebsd_experiment_pats import ExperimentPatterns
from ebsdtorch.ebsd.indexing import pca_dictionary_index_orientations
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
        spatial_coords=torch.randn((2, 2), dtype=torch.float32),
    )

    return geom, mp, exp_pats


def test_ebsd_di_ni_pca(test_ebsd_di):
    geom, mp, exp_pats = test_ebsd_di

    # index the orientations
    pca_dictionary_index_orientations(
        mp,
        geom,
        exp_pats,
        dictionary_resolution_degrees=10.0,
        dictionary_chunk_size=4096,
        signal_mask=None,
        virtual_binning=1,
        experiment_chunk_size=4096,
        match_dtype=torch.float16,
        n_pca_components=100,
    )

    orientations = exp_pats.get_orientations().cpu().numpy()

    assert orientations.shape == (2, 4)
