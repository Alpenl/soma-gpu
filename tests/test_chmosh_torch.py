import importlib
import importlib.util
import pickle
import shutil
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


def test_load_torch_mocap_session_supports_mcp_alias_for_c3d_payload(tmp_path):
    module = _load_chmosh_torch_module()
    mocap_path = tmp_path / "out1.mcp"
    shutil.copyfile(ROOT / "out1.c3d", mocap_path)

    loaded = module.load_torch_mocap_session(
        str(mocap_path),
        mocap_unit="m",
        mocap_rotate=None,
        labels_map=None,
    )

    assert loaded.markers.shape[0] > 0
    assert loaded.markers.shape[1:] == (54, 3)
    assert len(loaded.labels) == 54
    assert loaded.frame_rate == torch.tensor(60.0).item()


def test_mosh_stageii_torch_routes_to_sequence_solver_when_chunking_enabled(tmp_path, monkeypatch):
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
            [0.05, 0.20, -0.10],
            [-0.15, 0.10, 0.05],
        ],
        dtype=torch.float32,
    )
    markers = markers_latent[None, :, :] + marker_offsets[:, None, :].numpy()
    mocap_fname = tmp_path / "synthetic_sequence_chunk.pkl"
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
                "maxiter": 0,
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
                "sequence_chunk_size": 2,
                "sequence_chunk_overlap": 1,
                "sequence_optimizer": "adam",
            },
        }
    )

    recorded = {"sequence_calls": 0, "frame_calls": 0}

    def fake_fit_stageii_frame_torch(**kwargs):
        recorded["frame_calls"] += 1
        markers_obs = torch.as_tensor(kwargs["marker_observations"], dtype=torch.float32)
        transl = torch.as_tensor(kwargs["transl_init"], dtype=torch.float32)
        return SimpleNamespace(
            latent_pose=torch.as_tensor(kwargs["latent_pose_init"], dtype=torch.float32),
            fullpose=torch.zeros(1, 165, dtype=torch.float32),
            transl=transl,
            expression=None,
            predicted_markers=markers_obs.clone(),
            vertices=markers_obs.unsqueeze(0).clone(),
            joints=torch.zeros(1, 3, 3, dtype=torch.float32),
            loss_terms={"data": 0.0, "poseB": 0.0, "poseH": 0.0, "poseF": 0.0, "expr": 0.0, "velo": 0.0},
        )

    def fake_fit_stageii_sequence_torch(**kwargs):
        recorded["sequence_calls"] += 1
        markers_obs = torch.as_tensor(kwargs["marker_observations"], dtype=torch.float32)
        transl = torch.as_tensor(kwargs["transl_init"], dtype=torch.float32)
        fullpose = torch.zeros(markers_obs.shape[0], 165, dtype=torch.float32)
        latent_pose = torch.as_tensor(kwargs["latent_pose_init"], dtype=torch.float32)
        if latent_pose.ndim == 2 and latent_pose.shape[0] == 1 and markers_obs.shape[0] > 1:
            latent_pose = latent_pose.expand(markers_obs.shape[0], -1).clone()
        return SimpleNamespace(
            latent_pose=latent_pose,
            fullpose=fullpose,
            transl=transl,
            expression=None,
            predicted_markers=markers_obs.clone(),
            vertices=markers_obs.clone(),
            joints=torch.zeros(markers_obs.shape[0], 3, 3, dtype=torch.float32),
            loss_terms={
                "data": torch.zeros(markers_obs.shape[0], dtype=torch.float32),
                "poseB": torch.zeros(markers_obs.shape[0], dtype=torch.float32),
                "poseH": torch.zeros(markers_obs.shape[0], dtype=torch.float32),
                "poseF": torch.zeros(markers_obs.shape[0], dtype=torch.float32),
                "expr": torch.zeros(markers_obs.shape[0], dtype=torch.float32),
                "velo": torch.zeros(markers_obs.shape[0], dtype=torch.float32),
                "accel": torch.zeros(markers_obs.shape[0], dtype=torch.float32),
            },
        )

    monkeypatch.setattr(module, "fit_stageii_frame_torch", fake_fit_stageii_frame_torch)
    monkeypatch.setattr(module, "fit_stageii_sequence_torch", fake_fit_stageii_sequence_torch, raising=False)

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

    assert recorded["sequence_calls"] > 0
    assert recorded["frame_calls"] == 0
    assert stageii_data["fullpose"].shape == (4, 165)
