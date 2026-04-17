import importlib
import importlib.util
import pickle
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
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


def test_mosh_stageii_torch_sequence_chunking_runs_with_real_sequence_evaluator(tmp_path):
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
    mocap_fname = tmp_path / "synthetic_sequence_real_eval.pkl"
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

    assert stageii_data["fullpose"].shape == (4, 165)
    assert stageii_data["trans"].shape == (4, 3)
    assert stageii_data["stageii_debug_details"]["stageii_errs"]["data"].shape == (4,)


def test_mosh_stageii_torch_builds_stageii_evaluator_once_and_reuses_it(tmp_path, monkeypatch):
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
    mocap_fname = tmp_path / "synthetic_evaluator_reuse.pkl"
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
                "compile_evaluator": True,
                "compile_mode": "max-autotune",
                "compile_fullgraph": True,
            },
        }
    )

    recorded = {
        "builds": 0,
        "compile_module": None,
        "compile_mode": None,
        "compile_fullgraph": None,
        "evaluator_ids": [],
    }
    shared_evaluator = object()

    def fake_build_stageii_evaluator(**kwargs):
        recorded["builds"] += 1
        recorded["compile_module"] = kwargs["compile_module"]
        recorded["compile_mode"] = kwargs["compile_mode"]
        recorded["compile_fullgraph"] = kwargs["compile_fullgraph"]
        return shared_evaluator

    def fake_fit_stageii_frame_torch(**kwargs):
        recorded["evaluator_ids"].append(id(kwargs["evaluator"]))
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

    monkeypatch.setattr(module, "build_stageii_evaluator", fake_build_stageii_evaluator)
    monkeypatch.setattr(module, "fit_stageii_frame_torch", fake_fit_stageii_frame_torch)

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

    assert recorded["builds"] == 1
    assert recorded["compile_module"] is True
    assert recorded["compile_mode"] == "max-autotune"
    assert recorded["compile_fullgraph"] is True
    assert len(set(recorded["evaluator_ids"])) == 1
    assert stageii_data["fullpose"].shape == (2, 165)


def test_mosh_stageii_torch_aggregates_tensor_loss_terms_to_numeric_arrays(tmp_path, monkeypatch):
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
    mocap_fname = tmp_path / "synthetic_tensor_losses.pkl"
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
            },
        }
    )

    call_count = {"value": 0}

    def fail_tensor_array(self, dtype=None):
        raise AssertionError("tensor __array__ should not be used during per-frame aggregation")

    monkeypatch.setattr(torch.Tensor, "__array__", fail_tensor_array)

    def fake_fit_stageii_frame_torch(**kwargs):
        call_idx = call_count["value"]
        call_count["value"] += 1
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
            loss_terms={
                "data": torch.tensor(0.5 + call_idx, dtype=torch.float32),
                "poseB": torch.tensor(0.0, dtype=torch.float32),
                "poseH": torch.tensor(0.0, dtype=torch.float32),
                "poseF": torch.tensor(0.0, dtype=torch.float32),
                "expr": torch.tensor(0.0, dtype=torch.float32),
                "velo": torch.tensor(0.0, dtype=torch.float32),
            },
        )

    monkeypatch.setattr(module, "fit_stageii_frame_torch", fake_fit_stageii_frame_torch)

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

    assert stageii_data["stageii_debug_details"]["stageii_errs"]["data"].dtype != object
    assert stageii_data["stageii_debug_details"]["stageii_errs"]["data"].tolist() == pytest.approx([0.5, 1.5])


def test_mosh_stageii_torch_sequence_seed_prepass_builds_full_cache_before_chunk_solves(tmp_path, monkeypatch):
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
    mocap_fname = tmp_path / "synthetic_sequence_seed.pkl"
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
                "sequence_seed_refine_iters": 1,
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
        latent_pose = torch.as_tensor(kwargs["latent_pose_init"], dtype=torch.float32)
        return SimpleNamespace(
            latent_pose=latent_pose,
            fullpose=torch.zeros(markers_obs.shape[0], 165, dtype=torch.float32),
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

    assert recorded["frame_calls"] == 4
    assert recorded["sequence_calls"] > 0
    assert stageii_data["fullpose"].shape == (4, 165)


def test_mosh_stageii_torch_sequence_seed_chunk_init_reuses_cache_slices(tmp_path, monkeypatch):
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
    mocap_fname = tmp_path / "synthetic_sequence_seed_slice.pkl"
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
                "sequence_seed_refine_iters": 1,
            },
        }
    )

    recorded = {"sequence_transl_inits": []}
    frame_call_idx = {"value": 0}

    def fake_fit_stageii_frame_torch(**kwargs):
        call_idx = frame_call_idx["value"]
        frame_call_idx["value"] += 1
        markers_obs = torch.as_tensor(kwargs["marker_observations"], dtype=torch.float32)
        transl_value = torch.full((1, 3), 10.0 + float(call_idx), dtype=torch.float32)
        return SimpleNamespace(
            latent_pose=torch.as_tensor(kwargs["latent_pose_init"], dtype=torch.float32),
            fullpose=torch.zeros(1, 165, dtype=torch.float32),
            transl=transl_value,
            expression=None,
            predicted_markers=markers_obs.clone(),
            vertices=markers_obs.unsqueeze(0).clone(),
            joints=torch.zeros(1, 3, 3, dtype=torch.float32),
            loss_terms={"data": 0.0, "poseB": 0.0, "poseH": 0.0, "poseF": 0.0, "expr": 0.0, "velo": 0.0},
        )

    def fake_fit_stageii_sequence_torch(**kwargs):
        markers_obs = torch.as_tensor(kwargs["marker_observations"], dtype=torch.float32)
        transl = torch.as_tensor(kwargs["transl_init"], dtype=torch.float32)
        recorded["sequence_transl_inits"].append(transl.clone())
        latent_pose = torch.as_tensor(kwargs["latent_pose_init"], dtype=torch.float32)
        return SimpleNamespace(
            latent_pose=latent_pose,
            fullpose=torch.zeros(markers_obs.shape[0], 165, dtype=torch.float32),
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

    assert stageii_data["fullpose"].shape == (4, 165)
    assert len(recorded["sequence_transl_inits"]) == 3
    expected = [
        torch.tensor([[10.0, 10.0, 10.0], [11.0, 11.0, 11.0]], dtype=torch.float32),
        torch.tensor([[11.0, 11.0, 11.0], [12.0, 12.0, 12.0]], dtype=torch.float32),
        torch.tensor([[12.0, 12.0, 12.0], [13.0, 13.0, 13.0]], dtype=torch.float32),
    ]
    for got, want in zip(recorded["sequence_transl_inits"], expected):
        assert torch.allclose(got, want)
