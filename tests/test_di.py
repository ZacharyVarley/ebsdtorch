from ebsdtorch.ebsd_dictionary_indexing import EBSDDI
import kikuchipy as kp
import torch
import pytest


@pytest.fixture
def test_ebsd_di():
    ebsd_di = EBSDDI(
        laue_group=11,
        master_pattern_MSLNH=torch.zeros((10, 10), dtype=torch.float32),
        master_pattern_MSLSH=torch.zeros((10, 10), dtype=torch.float32),
        pattern_center=(0.4221, 0.2179, 0.4954),
        detector_height=60,
        detector_width=60,
        detector_tilt_deg=0.0,
        azimuthal_deg=0.0,
        sample_tilt_deg=70.0,
        signal_mask=None,
    )

    # project dictionary
    ebsd_di.project_dictionary(
        so3_n_samples=10,
        so3_batch_size=10,
    )

    return ebsd_di


def test_ebsd_di_ni(test_ebsd_di):
    # s = kp.data.ni_gain(number=1, allow_download=True).inav[:2, :2]
    # patterns = s.data.reshape(-1, 60 * 60)
    # patterns = torch.from_numpy(patterns).to(torch.float32)
    patterns = torch.randn((2, 60 * 60), dtype=torch.float32)

    indices = test_ebsd_di.di_patterns(
        experimental_data=patterns,
        topk=1,
        match_device=torch.device("cpu"),
        metric="angular",
        match_dtype=torch.float32,
    )
