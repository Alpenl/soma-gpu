import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moshpp.transformed_lm_torch import (
    _load_no_eyeball_vids,
    build_marker_attachment,
    decode_marker_attachment,
)


def test_marker_attachment_decode_matches_expected_local_frame_geometry():
    can_body = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    markers_latent = torch.tensor([[0.2, 0.3, 0.4]], dtype=torch.float32)

    attachment = build_marker_attachment(can_body, markers_latent, surface_model_type="smplx")

    rotation = torch.tensor(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    posed_body = (can_body @ rotation.T + torch.tensor([1.0, 2.0, 3.0])).to(torch.float64)
    decoded = decode_marker_attachment(attachment, posed_body)

    expected = (markers_latent @ rotation.T + torch.tensor([1.0, 2.0, 3.0])).to(torch.float64)
    assert decoded.dtype == posed_body.dtype
    assert torch.allclose(decoded, expected, atol=1e-5)


def test_marker_attachment_maps_filtered_indices_back_to_original_space(monkeypatch):
    can_body = torch.full((10475, 3), 1000.0, dtype=torch.float32)
    can_body[10] = torch.tensor([0.0, 0.0, 0.0])
    can_body[20] = torch.tensor([1.0, 0.0, 0.0])
    can_body[30] = torch.tensor([0.0, 1.0, 0.0])
    can_body[40] = torch.tensor([0.0, 0.0, 1.0])
    markers_latent = torch.tensor([[0.2, 0.3, 0.4]], dtype=torch.float32)

    monkeypatch.setattr(
        "moshpp.transformed_lm_torch._load_no_eyeball_vids",
        lambda support_base_dir=None: np.array([10, 20, 30, 40], dtype=np.int64),
    )

    attachment = build_marker_attachment(can_body, markers_latent, surface_model_type="smplx")

    assert attachment.closest.shape == (1, 3)
    assert attachment.closest.dtype == torch.long
    assert set(attachment.closest[0].tolist()).issubset({10, 20, 30, 40})


def test_load_no_eyeball_vids_keeps_last_non_eyeball_vertex(tmp_path):
    np.savez(tmp_path / "smplx_eyeballs.npz", eyeballs=np.array([1, 2, 3], dtype=np.int64))

    no_eyeballs = _load_no_eyeball_vids(str(tmp_path))

    assert 10474 in no_eyeballs
