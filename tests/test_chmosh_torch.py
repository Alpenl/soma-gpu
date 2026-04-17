import importlib
import importlib.util
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_chmosh_torch_module():
    try:
        spec = importlib.util.find_spec("moshpp.chmosh_torch")
    except ModuleNotFoundError:
        spec = None
    assert spec is not None, "moshpp.chmosh_torch is not available on main yet"
    return importlib.import_module("moshpp.chmosh_torch")


class DummyBodyOutput:
    def __init__(self, vertices, joints):
        self.vertices = vertices
        self.joints = joints


class TranslOnlyBodyModel:
    def __init__(self, canonical_vertices):
        self.canonical_vertices = canonical_vertices

    def __call__(self, **kwargs):
        transl = kwargs["transl"]
        vertices = self.canonical_vertices.unsqueeze(0) + transl[:, None, :]
        joints = torch.zeros(transl.shape[0], 3, 3, dtype=vertices.dtype, device=vertices.device)
        return DummyBodyOutput(vertices=vertices, joints=joints)


class ZeroPosePrior:
    def __init__(self, dim):
        self.means = torch.zeros(1, dim)

    def __call__(self, x):
        return torch.zeros(x.shape[0], x.shape[1] + 1, dtype=x.dtype, device=x.device)


def _ns(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{key: _ns(value) for key, value in obj.items()})
    return obj


def test_mosh_stageii_torch_fits_synthetic_sequence_and_returns_stageii_payload(tmp_path):
    module = _load_chmosh_torch_module()
    markers_latent = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    ).numpy()
    latent_labels = ["A", "B", "C", "D"]
    marker_offsets = torch.tensor(
        [
            [0.10, 0.00, 0.00],
            [0.25, -0.15, 0.35],
        ],
        dtype=torch.float32,
    )
    markers = markers_latent[None, :, :] + marker_offsets[:, None, :].numpy()
    mocap_fname = tmp_path / "synthetic_mocap.pkl"
    with mocap_fname.open("wb") as handle:
        pickle.dump({"markers": markers, "labels": latent_labels, "frame_rate": 120.0}, handle)

    cfg = _ns(
        {
            "mocap": {
                "unit": "m",
                "rotate": None,
                "subject_name": None,
                "multi_subject": False,
                "start_fidx": 0,
                "end_fidx": -1,
                "ds_rate": 1,
            },
            "surface_model": {
                "type": "smplx",
                "num_betas": 10,
                "dof_per_hand": 24,
                "num_expressions": 10,
                "use_hands_mean": True,
                "betas_expr_start_id": 300,
            },
            "moshpp": {
                "optimize_fingers": False,
                "optimize_face": False,
                "optimize_toes": False,
                "optimize_dynamics": False,
                "verbosity": 0,
            },
            "opt_settings": {
                "maxiter": 25,
                "weights": {
                    "stageii_wt_data": 1.0,
                    "stageii_wt_poseB": 0.0,
                    "stageii_wt_poseH": 0.0,
                    "stageii_wt_poseF": 0.0,
                    "stageii_wt_expr": 0.0,
                    "stageii_wt_dmpl": 0.0,
                    "stageii_wt_velo": 0.0,
                    "stageii_wt_annealing": 0.0,
                },
            },
            "runtime": {
                "backend": "torch",
                "device": "cpu",
            },
        }
    )

    stageii_data = module.mosh_stageii_torch(
        mocap_fname=str(mocap_fname),
        cfg=cfg,
        markers_latent=markers_latent,
        latent_labels=latent_labels,
        betas=torch.zeros(10).numpy(),
        marker_meta={"marker_type_mask": {}, "marker_type": {}, "surface_model_type": "smplx"},
        body_model_factory=lambda: TranslOnlyBodyModel(torch.as_tensor(markers_latent, dtype=torch.float32)),
        pose_prior=ZeroPosePrior(63),
        device="cpu",
    )

    assert stageii_data["fullpose"].shape == (2, 165)
    assert stageii_data["trans"].shape == (2, 3)
    assert stageii_data["stageii_debug_details"]["stageii_errs"]["data"].shape == (2,)
    assert len(stageii_data["stageii_debug_details"]["markers_obs"]) == 2
    assert len(stageii_data["stageii_debug_details"]["markers_sim"]) == 2
    assert torch.allclose(torch.as_tensor(stageii_data["trans"][1]), marker_offsets[1], atol=1e-3)
