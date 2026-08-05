import numpy as np
import pytest
from scipy.sparse import csc_matrix

from _tesseract_py_util.gari_dem_utils import (
    assign_prior_weights,
    get_two_stage_layout,
)


def test_mode_pr_cost_partition_and_layout_metadata():
    u = csc_matrix([[1]], dtype=np.uint8)
    v = csc_matrix([[1]], dtype=np.uint8)
    structure = {
        "i_hx_only": np.array([0]),
        "i_hz_only": np.array([1]),
        "i_hy": np.array([2]),
        "U": u,
        "V": v,
        "nx_virt": 1,
        "nz_virt": 1,
        "nx_real": 1,
        "nz_real": 1,
        "num_original_detectors": 2,
        "num_gari_detectors": 4,
        "dx": csc_matrix([[1]], dtype=np.uint8),
        "dz": csc_matrix([[1]], dtype=np.uint8),
        "gari_matrix": csc_matrix(
            [
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],
                [1, 0, 1, 1, 0],
                [0, 1, 1, 0, 1],
            ],
            dtype=np.uint8,
        ),
    }
    source_probabilities = np.array([0.01, 0.02, 0.04])
    mode_pr_probabilities = assign_prior_weights(
        structure, "modePR", source_probabilities
    )
    source_costs = np.log1p(-source_probabilities) - np.log(source_probabilities)
    mode_pr_costs = np.log1p(-mode_pr_probabilities) - np.log(
        mode_pr_probabilities
    )
    np.testing.assert_allclose(
        [
            mode_pr_costs[0] + mode_pr_costs[3],
            mode_pr_costs[1] + mode_pr_costs[4],
            mode_pr_costs[2] + mode_pr_costs[3] + mode_pr_costs[4],
        ],
        source_costs,
    )
    assert np.all(mode_pr_costs > 0)

    layout = get_two_stage_layout(structure, "tiny_ogL_modePR.dem", "modePR")
    assert layout["dem_file"] == "tiny_ogL_modePR.dem"
    assert layout["top_prior_policy"] == "modePR"
    assert layout["bottom_prior_policy"] == "lp_residual"
    assert layout["prior_policy_id"] == "lp_floor_then_max_barred_v1"
    assert layout["final_cost_policy"] == "reconstructed_original_physical"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
