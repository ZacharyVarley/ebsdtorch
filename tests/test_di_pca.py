from ebsdtorch.ebsd_dictionary_indexing import EBSDDIwithPCA
import kikuchipy as kp
import torch
import pytest


@pytest.fixture
def test_ebsd_di_pca():
    ebsd_di_pca = EBSDDIwithPCA(
        laue_group=11,
        master_pattern_MSLNH=torch.randn((10, 10), dtype=torch.float32),
        master_pattern_MSLSH=torch.randn((10, 10), dtype=torch.float32),
        pattern_center=(0.4221, 0.2179, 0.4954),
        detector_height=60,
        detector_width=60,
        detector_tilt_deg=0.0,
        azimuthal_deg=0.0,
        sample_tilt_deg=70.0,
        signal_mask=None,
    )

    ebsd_di_pca.compute_PCA_detector_plane(
        so3_n_samples=10,
        so3_batch_size=10,
        correlation=False,
    )

    ebsd_di_pca.project_dictionary_pca(
        pca_n_max_components=10,
        so3_n_samples=10,
        so3_batch_size=10,
    )

    return ebsd_di_pca


def test_ebsd_di_pca_ni(test_ebsd_di_pca):
    # s = kp.data.ni_gain(number=1, allow_download=True).inav[:2, :2]
    # patterns = s.data.reshape(-1, 60 * 60)
    # patterns = torch.from_numpy(patterns).to(torch.float32)

    patterns = torch.randn((2, 60 * 60), dtype=torch.float32)

    indices = test_ebsd_di_pca.pca_di_patterns(
        experimental_data=patterns,
        n_pca_components=10,
        topk=1,
        match_device=torch.device("cpu"),
        metric="angular",
        match_dtype=torch.float32,
    )
