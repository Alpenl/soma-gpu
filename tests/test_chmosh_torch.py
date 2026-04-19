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


def test_runtime_sequence_fit_options_inherit_refine_solver_defaults():
    module = _load_chmosh_torch_module()
    runtime = _ns(
        {
            "refine_optimizer": "lbfgs",
            "refine_iters": 33,
            "refine_lr": 0.7,
            "lbfgs_history_size": 55,
            "lbfgs_tolerance_grad": 1e-5,
            "lbfgs_tolerance_change": 1e-6,
            "lbfgs_max_eval": 77,
        }
    )
    cfg = _ns(
        {
            "opt_settings": {
                "maxiter": 20,
            }
        }
    )

    options = module._runtime_sequence_fit_options(cfg, runtime)

    assert options.optimizer == "lbfgs"
    assert options.max_iters == 33
    assert options.lr == pytest.approx(0.7)
    assert options.history_size == 55
    assert options.tolerance_grad == pytest.approx(1e-5)
    assert options.tolerance_change == pytest.approx(1e-6)
    assert options.max_eval == 77


def test_runtime_sequence_seed_options_inherit_runtime_seed_solver_defaults():
    module = _load_chmosh_torch_module()
    runtime = _ns(
        {
            "sequence_seed_refine_iters": 9,
            "sequence_seed_refine_lr": 0.03,
            "sequence_seed_refine_optimizer": "adam",
            "sequence_seed_refine_max_eval": 41,
        }
    )
    sequence_options = module.TorchSequenceFitOptions(
        max_iters=33,
        lr=0.7,
        optimizer="lbfgs",
        history_size=55,
        tolerance_grad=1e-5,
        tolerance_change=1e-6,
        max_eval=77,
    )

    seed_options = module._runtime_sequence_seed_options(sequence_options, runtime)

    assert seed_options.refine_iters == 9
    assert seed_options.refine_lr == pytest.approx(0.03)
    assert seed_options.refine_optimizer == "adam"
    assert seed_options.history_size == 55
    assert seed_options.tolerance_grad == pytest.approx(1e-5)
    assert seed_options.tolerance_change == pytest.approx(1e-6)
    assert seed_options.max_eval == 41
    assert seed_options.refine_max_eval == 41


def test_mosh_stageii_torch_frame_solver_inherits_runtime_stage_solver_defaults(tmp_path, monkeypatch):
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
    mocap_fname = tmp_path / "synthetic_frame_runtime_defaults.pkl"
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
                "maxiter": 20,
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
                "rigid_iters": 3,
                "warmup_iters": 4,
                "refine_iters": 5,
                "rigid_lr": 0.11,
                "warmup_lr": 0.22,
                "refine_lr": 0.33,
                "rigid_optimizer": "adam",
                "warmup_optimizer": "lbfgs",
                "refine_optimizer": "adam",
                "lbfgs_history_size": 13,
                "lbfgs_tolerance_grad": 1e-5,
                "lbfgs_tolerance_change": 1e-6,
                "rigid_max_eval": 31,
                "warmup_max_eval": 41,
                "refine_max_eval": 51,
            },
        }
    )

    recorded = {"options": []}

    def fake_fit_stageii_frame_torch(**kwargs):
        recorded["options"].append(kwargs["options"])
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

    monkeypatch.setattr(module, "fit_stageii_frame_torch", fake_fit_stageii_frame_torch)

    module.mosh_stageii_torch(
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

    assert len(recorded["options"]) == 2
    options = recorded["options"][0]
    assert options.rigid_iters == 3
    assert options.warmup_iters == 4
    assert options.refine_iters == 5
    assert options.rigid_lr == pytest.approx(0.11)
    assert options.warmup_lr == pytest.approx(0.22)
    assert options.refine_lr == pytest.approx(0.33)
    assert options.rigid_optimizer == "adam"
    assert options.warmup_optimizer == "lbfgs"
    assert options.refine_optimizer == "adam"
    assert options.history_size == 13
    assert options.tolerance_grad == pytest.approx(1e-5)
    assert options.tolerance_change == pytest.approx(1e-6)
    assert options.rigid_max_eval == 31
    assert options.warmup_max_eval == 41
    assert options.refine_max_eval == 51


def test_mosh_stageii_torch_sequence_solver_inherits_refine_runtime_defaults(tmp_path, monkeypatch):
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
    mocap_fname = tmp_path / "synthetic_sequence_inherit_refine_defaults.pkl"
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
                "refine_optimizer": "lbfgs",
                "refine_iters": 33,
                "refine_lr": 0.7,
                "lbfgs_history_size": 55,
                "lbfgs_tolerance_grad": 1e-5,
                "lbfgs_tolerance_change": 1e-6,
                "lbfgs_max_eval": 77,
            },
        }
    )

    recorded = {"sequence_calls": 0}

    def fail_if_frame_solver_called(**kwargs):
        del kwargs
        pytest.fail("frame solver should not run when chunked sequence path is active without seed prepass")

    def fake_fit_stageii_sequence_torch(**kwargs):
        recorded["sequence_calls"] += 1
        recorded["options"] = kwargs["options"]
        markers_obs = torch.as_tensor(kwargs["marker_observations"], dtype=torch.float32)
        transl = torch.as_tensor(kwargs["transl_init"], dtype=torch.float32)
        latent_pose = torch.as_tensor(kwargs["latent_pose_init"], dtype=torch.float32)
        if latent_pose.ndim == 2 and latent_pose.shape[0] == 1 and markers_obs.shape[0] > 1:
            latent_pose = latent_pose.expand(markers_obs.shape[0], -1).clone()
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

    monkeypatch.setattr(module, "fit_stageii_frame_torch", fail_if_frame_solver_called)
    monkeypatch.setattr(module, "fit_stageii_sequence_torch", fake_fit_stageii_sequence_torch, raising=False)

    module.mosh_stageii_torch(
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
    assert recorded["options"].optimizer == "lbfgs"
    assert recorded["options"].max_iters == 33
    assert recorded["options"].lr == pytest.approx(0.7)
    assert recorded["options"].history_size == 55
    assert recorded["options"].tolerance_grad == pytest.approx(1e-5)
    assert recorded["options"].tolerance_change == pytest.approx(1e-6)
    assert recorded["options"].max_eval == 77


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


def test_mosh_stageii_torch_sequence_solver_can_reseed_only_selected_overlap_chunk_from_previous_chunk(
    tmp_path,
    monkeypatch,
):
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
    mocap_fname = tmp_path / "synthetic_sequence_chunk_local_reseed.pkl"
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
                "sequence_overlap_pose_seed_from_previous_chunk_indices": "1",
                "sequence_overlap_transl_seed_from_previous_chunk_indices": "1",
            },
        }
    )

    frame_call_idx = {"value": 0}
    recorded = {"sequence_kwargs": []}

    def fake_fit_stageii_frame_torch(**kwargs):
        call_idx = frame_call_idx["value"]
        frame_call_idx["value"] += 1
        markers_obs = torch.as_tensor(kwargs["marker_observations"], dtype=torch.float32)
        latent_pose = torch.zeros(1, kwargs["layout"].latent_dim, dtype=torch.float32)
        latent_pose[0, 0] = float(10 + call_idx)
        transl = torch.full((1, 3), float(20 + call_idx), dtype=torch.float32)
        return SimpleNamespace(
            latent_pose=latent_pose,
            fullpose=torch.zeros(1, 165, dtype=torch.float32),
            transl=transl,
            expression=None,
            predicted_markers=markers_obs.clone(),
            vertices=markers_obs.unsqueeze(0).clone(),
            joints=torch.zeros(1, 3, 3, dtype=torch.float32),
            loss_terms={"data": 0.0, "poseB": 0.0, "poseH": 0.0, "poseF": 0.0, "expr": 0.0, "velo": 0.0},
        )

    def fake_fit_stageii_sequence_torch(**kwargs):
        chunk_idx = len(recorded["sequence_kwargs"])
        latent_pose_init = torch.as_tensor(kwargs["latent_pose_init"], dtype=torch.float32).clone()
        transl_init = torch.as_tensor(kwargs["transl_init"], dtype=torch.float32).clone()
        recorded["sequence_kwargs"].append(
            {
                "latent_pose_init": latent_pose_init.clone(),
                "transl_init": transl_init.clone(),
            }
        )
        markers_obs = torch.as_tensor(kwargs["marker_observations"], dtype=torch.float32)
        latent_pose = latent_pose_init.clone()
        transl = transl_init.clone()
        if chunk_idx == 0:
            latent_pose[:, 0] = torch.tensor([100.0, 101.0], dtype=torch.float32)
            transl[:] = torch.tensor([[200.0, 200.0, 200.0], [201.0, 201.0, 201.0]], dtype=torch.float32)
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
    assert len(recorded["sequence_kwargs"]) == 3
    assert recorded["sequence_kwargs"][0]["latent_pose_init"][:, 0].tolist() == pytest.approx([10.0, 11.0])
    assert recorded["sequence_kwargs"][0]["transl_init"][:, 0].tolist() == pytest.approx([20.0, 21.0])
    assert recorded["sequence_kwargs"][1]["latent_pose_init"][:, 0].tolist() == pytest.approx([101.0, 12.0])
    assert recorded["sequence_kwargs"][1]["transl_init"][:, 0].tolist() == pytest.approx([201.0, 22.0])
    assert recorded["sequence_kwargs"][2]["latent_pose_init"][:, 0].tolist() == pytest.approx([12.0, 13.0])
    assert recorded["sequence_kwargs"][2]["transl_init"][:, 0].tolist() == pytest.approx([22.0, 23.0])
    stitch_diagnostics = stageii_data["stageii_debug_details"]["sequence_chunk_stitch_diagnostics"]
    assert stitch_diagnostics[1]["overlap_pose_seeded_from_previous"] is True
    assert stitch_diagnostics[1]["overlap_transl_seeded_from_previous"] is True
    assert stitch_diagnostics[2]["overlap_pose_seeded_from_previous"] is False
    assert stitch_diagnostics[2]["overlap_transl_seeded_from_previous"] is False


def test_mosh_stageii_torch_sequence_solver_can_localize_overlap_pose_seed_window(tmp_path, monkeypatch):
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
            [0.18, -0.08, 0.12],
        ],
        dtype=torch.float32,
    )
    markers = markers_latent[None, :, :] + marker_offsets[:, None, :].numpy()
    mocap_fname = tmp_path / "synthetic_sequence_chunk_local_pose_seed_window.pkl"
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
                "sequence_chunk_size": 3,
                "sequence_chunk_overlap": 2,
                "sequence_optimizer": "adam",
                "sequence_seed_refine_iters": 1,
                "sequence_overlap_pose_seed_from_previous_chunk_indices": "1",
                "sequence_overlap_pose_seed_window_start": 1,
                "sequence_overlap_pose_seed_window": 1,
            },
        }
    )

    frame_call_idx = {"value": 0}
    recorded = {"sequence_kwargs": []}

    def fake_fit_stageii_frame_torch(**kwargs):
        call_idx = frame_call_idx["value"]
        frame_call_idx["value"] += 1
        markers_obs = torch.as_tensor(kwargs["marker_observations"], dtype=torch.float32)
        latent_pose = torch.zeros(1, kwargs["layout"].latent_dim, dtype=torch.float32)
        latent_pose[0, 0] = float(10 + call_idx)
        transl = torch.full((1, 3), float(20 + call_idx), dtype=torch.float32)
        return SimpleNamespace(
            latent_pose=latent_pose,
            fullpose=torch.zeros(1, 165, dtype=torch.float32),
            transl=transl,
            expression=None,
            predicted_markers=markers_obs.clone(),
            vertices=markers_obs.unsqueeze(0).clone(),
            joints=torch.zeros(1, 3, 3, dtype=torch.float32),
            loss_terms={"data": 0.0, "poseB": 0.0, "poseH": 0.0, "poseF": 0.0, "expr": 0.0, "velo": 0.0},
        )

    def fake_fit_stageii_sequence_torch(**kwargs):
        chunk_idx = len(recorded["sequence_kwargs"])
        latent_pose_init = torch.as_tensor(kwargs["latent_pose_init"], dtype=torch.float32).clone()
        recorded["sequence_kwargs"].append({"latent_pose_init": latent_pose_init.clone()})
        markers_obs = torch.as_tensor(kwargs["marker_observations"], dtype=torch.float32)
        latent_pose = latent_pose_init.clone()
        if chunk_idx == 0:
            latent_pose[:, 0] = torch.tensor([100.0, 101.0, 102.0], dtype=torch.float32)
        return SimpleNamespace(
            latent_pose=latent_pose,
            fullpose=torch.zeros(markers_obs.shape[0], 165, dtype=torch.float32),
            transl=torch.zeros(markers_obs.shape[0], 3, dtype=torch.float32),
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

    assert stageii_data["fullpose"].shape == (5, 165)
    assert len(recorded["sequence_kwargs"]) == 3
    assert recorded["sequence_kwargs"][0]["latent_pose_init"][:, 0].tolist() == pytest.approx([10.0, 11.0, 12.0])
    assert recorded["sequence_kwargs"][1]["latent_pose_init"][:, 0].tolist() == pytest.approx([11.0, 102.0, 13.0])
    assert recorded["sequence_kwargs"][2]["latent_pose_init"][:, 0].tolist() == pytest.approx([12.0, 13.0, 14.0])
    stitch_diagnostics = stageii_data["stageii_debug_details"]["sequence_chunk_stitch_diagnostics"]
    assert stitch_diagnostics[1]["overlap_pose_seeded_from_previous"] is True
    assert stitch_diagnostics[2]["overlap_pose_seeded_from_previous"] is False


def test_mosh_stageii_torch_sequence_solver_receives_boundary_references_from_previous_chunk(tmp_path, monkeypatch):
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
    mocap_fname = tmp_path / "synthetic_sequence_boundary_refs.pkl"
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
                    "stageii_wt_velo": 1.0,
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
                "sequence_boundary_velocity_reference": True,
                "sequence_boundary_transl_velocity_reference": True,
                "sequence_delta_pose": 3.0,
                "sequence_delta_trans": 5.0,
                "sequence_transl_velocity": 7.0,
            },
        }
    )

    frame_call_idx = {"value": 0}
    recorded = {"sequence_kwargs": []}

    def fake_fit_stageii_frame_torch(**kwargs):
        call_idx = frame_call_idx["value"]
        frame_call_idx["value"] += 1
        markers_obs = torch.as_tensor(kwargs["marker_observations"], dtype=torch.float32)
        latent_pose = torch.zeros(1, kwargs["layout"].latent_dim, dtype=torch.float32)
        latent_pose[0, 0] = 10.0 + float(call_idx)
        transl = torch.full((1, 3), 20.0 + float(call_idx), dtype=torch.float32)
        return SimpleNamespace(
            latent_pose=latent_pose,
            fullpose=torch.zeros(1, 165, dtype=torch.float32),
            transl=transl,
            expression=None,
            predicted_markers=markers_obs.clone(),
            vertices=markers_obs.unsqueeze(0).clone(),
            joints=torch.zeros(1, 3, 3, dtype=torch.float32),
            loss_terms={"data": 0.0, "poseB": 0.0, "poseH": 0.0, "poseF": 0.0, "expr": 0.0, "velo": 0.0},
        )

    def fake_fit_stageii_sequence_torch(**kwargs):
        chunk_idx = len(recorded["sequence_kwargs"])
        recorded["sequence_kwargs"].append(
            {
                "weights_transl_velocity": float(getattr(kwargs["weights"], "transl_velocity", 0.0)),
                "velocity_reference": None
                if kwargs.get("velocity_reference") is None
                else torch.as_tensor(kwargs["velocity_reference"], dtype=torch.float32).clone(),
                "transl_velocity_reference": None
                if kwargs.get("transl_velocity_reference") is None
                else torch.as_tensor(kwargs["transl_velocity_reference"], dtype=torch.float32).clone(),
                "transl_velocity_reference_index": kwargs.get("transl_velocity_reference_index"),
                "latent_pose_reference": None
                if kwargs.get("latent_pose_reference") is None
                else torch.as_tensor(kwargs["latent_pose_reference"], dtype=torch.float32).clone(),
                "transl_reference": None
                if kwargs.get("transl_reference") is None
                else torch.as_tensor(kwargs["transl_reference"], dtype=torch.float32).clone(),
            }
        )
        markers_obs = torch.as_tensor(kwargs["marker_observations"], dtype=torch.float32)
        latent_pose = torch.as_tensor(kwargs["latent_pose_init"], dtype=torch.float32).clone()
        transl = torch.as_tensor(kwargs["transl_init"], dtype=torch.float32).clone()
        if chunk_idx == 0:
            latent_pose[:, 0] = torch.tensor([100.0, 101.0], dtype=torch.float32)
            transl[:] = torch.tensor([[200.0, 200.0, 200.0], [201.0, 201.0, 201.0]], dtype=torch.float32)
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
                "veloT": torch.zeros(markers_obs.shape[0], dtype=torch.float32),
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
    assert len(recorded["sequence_kwargs"]) == 3

    second_chunk = recorded["sequence_kwargs"][1]
    assert torch.allclose(
        second_chunk["velocity_reference"][:, :1],
        torch.tensor([[101.0]], dtype=torch.float32),
        atol=0.0,
    )
    assert second_chunk["weights_transl_velocity"] == pytest.approx(7.0)
    assert torch.allclose(
        second_chunk["transl_velocity_reference"],
        torch.tensor([[201.0, 201.0, 201.0]], dtype=torch.float32),
        atol=0.0,
    )
    assert second_chunk["transl_velocity_reference_index"] == 1
    assert torch.allclose(
        second_chunk["latent_pose_reference"][:, :1],
        torch.tensor([[101.0], [12.0]], dtype=torch.float32),
        atol=0.0,
    )
    assert torch.allclose(
        second_chunk["transl_reference"],
        torch.tensor([[201.0, 201.0, 201.0], [201.0, 201.0, 201.0]], dtype=torch.float32),
        atol=0.0,
    )


def test_splice_chunk_overlap_reference_can_extend_local_window_past_keep_seam():
    module = _load_chmosh_torch_module()

    base_reference = torch.tensor(
        [
            [22.0, 22.0, 22.0],
            [23.0, 23.0, 23.0],
            [24.0, 24.0, 24.0],
            [25.0, 25.0, 25.0],
        ],
        dtype=torch.float32,
    )
    previous_tail = torch.tensor(
        [
            [202.0, 202.0, 202.0],
            [203.0, 203.0, 203.0],
        ],
        dtype=torch.float32,
    )

    spliced = module._splice_chunk_overlap_reference(
        base_reference,
        previous_tail,
        2,
        include_keep_seam=True,
        keep_seam_window=2,
    )

    assert torch.allclose(
        spliced,
        torch.tensor(
            [
                [202.0, 202.0, 202.0],
                [203.0, 203.0, 203.0],
                [203.0, 203.0, 203.0],
                [204.0, 204.0, 204.0],
            ],
            dtype=torch.float32,
        ),
        atol=0.0,
    )


def test_select_chunk_keep_start_can_minimize_translation_jump():
    module = _load_chmosh_torch_module()

    keep_start = module._select_chunk_keep_start(
        stitch_mode="adaptive_transl_jump",
        overlap_count=2,
        previous_fullpose_tail=torch.tensor([[0.0], [0.0]], dtype=torch.float32),
        previous_transl_tail=torch.tensor(
            [
                [90.0, 0.0, 0.0],
                [100.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        current_fullpose=torch.zeros((4, 1), dtype=torch.float32),
        current_transl=torch.tensor(
            [
                [89.0, 0.0, 0.0],
                [91.0, 0.0, 0.0],
                [92.0, 0.0, 0.0],
                [93.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
    )

    assert keep_start == 1


def test_select_chunk_keep_start_can_preserve_legacy_pose_jump_when_guarded():
    module = _load_chmosh_torch_module()

    keep_start = module._select_chunk_keep_start(
        stitch_mode="adaptive_transl_jump_pose_guard",
        overlap_count=2,
        previous_fullpose_tail=torch.tensor([[0.0], [10.0]], dtype=torch.float32),
        previous_transl_tail=torch.tensor(
            [
                [90.0, 0.0, 0.0],
                [100.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        current_fullpose=torch.tensor(
            [
                [0.0],
                [50.0],
                [11.0],
                [12.0],
            ],
            dtype=torch.float32,
        ),
        current_transl=torch.tensor(
            [
                [89.0, 0.0, 0.0],
                [91.0, 0.0, 0.0],
                [92.0, 0.0, 0.0],
                [93.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
    )

    assert keep_start == 2


def test_select_chunk_keep_start_can_use_pose_second_key_for_translation_near_tie():
    module = _load_chmosh_torch_module()

    keep_start = module._select_chunk_keep_start(
        stitch_mode="adaptive_transl_jump_pose_guard",
        overlap_count=2,
        previous_fullpose_tail=torch.tensor([[0.0], [10.0]], dtype=torch.float32),
        previous_transl_tail=torch.tensor(
            [
                [90.0, 0.0, 0.0],
                [100.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        current_fullpose=torch.tensor(
            [
                [0.0],
                [1.0],
                [50.0],
                [51.0],
            ],
            dtype=torch.float32,
        ),
        current_transl=torch.tensor(
            [
                [89.0, 0.0, 0.0],
                [91.00044, 0.0, 0.0],
                [101.0, 0.0, 0.0],
                [102.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
    )

    assert keep_start == 1


def test_select_chunk_keep_start_keeps_translation_first_when_gap_is_not_near_tie():
    module = _load_chmosh_torch_module()

    keep_start = module._select_chunk_keep_start(
        stitch_mode="adaptive_transl_jump_pose_guard",
        overlap_count=2,
        previous_fullpose_tail=torch.tensor([[0.0], [10.0]], dtype=torch.float32),
        previous_transl_tail=torch.tensor(
            [
                [90.0, 0.0, 0.0],
                [100.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        current_fullpose=torch.tensor(
            [
                [0.0],
                [1.0],
                [50.0],
                [51.0],
            ],
            dtype=torch.float32,
        ),
        current_transl=torch.tensor(
            [
                [89.0, 0.0, 0.0],
                [91.0006, 0.0, 0.0],
                [101.0, 0.0, 0.0],
                [102.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
    )

    assert keep_start == 2


def test_select_chunk_keep_start_can_choose_earlier_boundary_when_mesh_guard_holds():
    module = _load_chmosh_torch_module()

    keep_start = module._select_chunk_keep_start(
        stitch_mode="adaptive_transl_jump_pose_mesh_guard",
        overlap_count=2,
        previous_fullpose_tail=torch.tensor([[90.0], [100.0]], dtype=torch.float32),
        previous_transl_tail=torch.tensor(
            [
                [90.0, 0.0, 0.0],
                [100.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        previous_vertices_tail=torch.tensor(
            [
                [[90.0, 0.0, 0.0]],
                [[100.0, 0.0, 0.0]],
            ],
            dtype=torch.float32,
        ),
        current_fullpose=torch.tensor(
            [
                [89.0],
                [91.0],
                [101.0],
                [102.0],
            ],
            dtype=torch.float32,
        ),
        current_transl=torch.tensor(
            [
                [89.0, 0.0, 0.0],
                [91.0, 0.0, 0.0],
                [109.0, 0.0, 0.0],
                [110.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        current_vertices=torch.tensor(
            [
                [[89.0, 0.0, 0.0]],
                [[91.0, 0.0, 0.0]],
                [[101.0, 0.0, 0.0]],
                [[102.0, 0.0, 0.0]],
            ],
            dtype=torch.float32,
        ),
    )

    assert keep_start == 1


def test_select_chunk_keep_start_can_preserve_legacy_mesh_jump_when_guarded():
    module = _load_chmosh_torch_module()

    keep_start = module._select_chunk_keep_start(
        stitch_mode="adaptive_transl_jump_pose_mesh_guard",
        overlap_count=2,
        previous_fullpose_tail=torch.tensor([[90.0], [100.0]], dtype=torch.float32),
        previous_transl_tail=torch.tensor(
            [
                [90.0, 0.0, 0.0],
                [100.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        previous_vertices_tail=torch.tensor(
            [
                [[90.0, 0.0, 0.0]],
                [[100.0, 0.0, 0.0]],
            ],
            dtype=torch.float32,
        ),
        current_fullpose=torch.tensor(
            [
                [89.0],
                [91.0],
                [101.0],
                [102.0],
            ],
            dtype=torch.float32,
        ),
        current_transl=torch.tensor(
            [
                [89.0, 0.0, 0.0],
                [91.0, 0.0, 0.0],
                [109.0, 0.0, 0.0],
                [110.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        current_vertices=torch.tensor(
            [
                [[89.0, 0.0, 0.0]],
                [[120.0, 0.0, 0.0]],
                [[101.0, 0.0, 0.0]],
                [[102.0, 0.0, 0.0]],
            ],
            dtype=torch.float32,
        ),
    )

    assert keep_start == 2


def test_select_chunk_keep_start_can_return_candidate_diagnostics():
    module = _load_chmosh_torch_module()

    keep_start, diagnostics = module._select_chunk_keep_start(
        stitch_mode="adaptive_transl_jump_pose_guard",
        overlap_count=2,
        previous_fullpose_tail=torch.tensor([[0.0], [10.0]], dtype=torch.float32),
        previous_transl_tail=torch.tensor(
            [
                [90.0, 0.0, 0.0],
                [100.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        current_fullpose=torch.tensor(
            [
                [0.0],
                [1.0],
                [11.0],
                [12.0],
            ],
            dtype=torch.float32,
        ),
        current_transl=torch.tensor(
            [
                [89.0, 0.0, 0.0],
                [91.0, 0.0, 0.0],
                [109.0, 0.0, 0.0],
                [110.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        return_diagnostics=True,
    )

    assert keep_start == 1
    assert diagnostics["default_keep_start"] == 2
    assert diagnostics["selected_keep_start"] == 1
    assert diagnostics["selected_transl_jump"] == pytest.approx(1.0)
    assert diagnostics["selected_pose_jump"] == pytest.approx(1.0)
    assert diagnostics["selected_mesh_jump"] == pytest.approx(0.0)
    assert diagnostics["default_pose_jump"] == pytest.approx(1.0)
    assert diagnostics["default_mesh_jump"] is None
    assert [row["keep_start"] for row in diagnostics["candidate_metrics"]] == [1, 2]
    assert diagnostics["candidate_metrics"][0]["selected"] is True
    assert diagnostics["candidate_metrics"][0]["transl_jump"] == pytest.approx(1.0)
    assert diagnostics["candidate_metrics"][0]["pose_jump"] == pytest.approx(1.0)
    assert diagnostics["candidate_metrics"][0]["passed_pose_guard"] is True
    assert diagnostics["candidate_metrics"][1]["selected"] is False
    assert diagnostics["candidate_metrics"][1]["transl_jump"] == pytest.approx(9.0)


def test_mosh_stageii_torch_can_stitch_chunk_outputs_at_adaptive_translation_boundary(
    tmp_path,
    monkeypatch,
):
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
            [0.30, 0.10, 0.15],
            [-0.20, -0.05, 0.10],
        ],
        dtype=torch.float32,
    )
    markers = markers_latent[None, :, :] + marker_offsets[:, None, :].numpy()
    mocap_fname = tmp_path / "synthetic_sequence_adaptive_stitch.pkl"
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
                "sequence_chunk_size": 4,
                "sequence_chunk_overlap": 2,
                "sequence_optimizer": "adam",
                "sequence_chunk_stitch_mode": "adaptive_transl_jump",
            },
        }
    )

    recorded = {"sequence_calls": 0}

    def fake_fit_stageii_sequence_torch(**kwargs):
        chunk_idx = recorded["sequence_calls"]
        recorded["sequence_calls"] += 1
        markers_obs = torch.as_tensor(kwargs["marker_observations"], dtype=torch.float32)
        transl_values = (
            [0.0, 1.0, 90.0, 100.0]
            if chunk_idx == 0
            else [89.0, 91.0, 92.0, 93.0]
        )
        transl = torch.tensor([[value, 0.0, 0.0] for value in transl_values], dtype=torch.float32)
        fullpose = torch.zeros((len(transl_values), 165), dtype=torch.float32)
        fullpose[:, 0] = torch.tensor(transl_values, dtype=torch.float32)
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

    assert stageii_data["trans"][:, 0].tolist() == pytest.approx([0.0, 1.0, 90.0, 91.0, 92.0, 93.0])
    assert stageii_data["fullpose"][:, 0].tolist() == pytest.approx([0.0, 1.0, 90.0, 91.0, 92.0, 93.0])
    assert stageii_data["stageii_debug_details"]["sequence_chunk_keep_starts"] == [0, 1]
    stitch_diagnostics = stageii_data["stageii_debug_details"]["sequence_chunk_stitch_diagnostics"]
    assert len(stitch_diagnostics) == 2
    assert stitch_diagnostics[0]["selected_keep_start"] == 0
    assert stitch_diagnostics[0]["candidate_metrics"] == []
    assert stitch_diagnostics[1]["chunk_index"] == 1
    assert stitch_diagnostics[1]["selected_keep_start"] == 1
    assert stitch_diagnostics[1]["default_keep_start"] == 2
    assert [row["keep_start"] for row in stitch_diagnostics[1]["candidate_metrics"]] == [1, 2]
    assert stitch_diagnostics[1]["candidate_metrics"][0]["selected"] is True


def test_mosh_stageii_torch_pose_guarded_stitching_can_keep_legacy_boundary_when_pose_would_regress(
    tmp_path,
    monkeypatch,
):
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
            [0.30, 0.10, 0.15],
            [-0.20, -0.05, 0.10],
        ],
        dtype=torch.float32,
    )
    markers = markers_latent[None, :, :] + marker_offsets[:, None, :].numpy()
    mocap_fname = tmp_path / "synthetic_sequence_pose_guarded_stitch.pkl"
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
                "sequence_chunk_size": 4,
                "sequence_chunk_overlap": 2,
                "sequence_optimizer": "adam",
                "sequence_chunk_stitch_mode": "adaptive_transl_jump_pose_guard",
            },
        }
    )

    recorded = {"sequence_calls": 0}

    def fake_fit_stageii_sequence_torch(**kwargs):
        chunk_idx = recorded["sequence_calls"]
        recorded["sequence_calls"] += 1
        markers_obs = torch.as_tensor(kwargs["marker_observations"], dtype=torch.float32)
        transl_values = (
            [0.0, 1.0, 90.0, 100.0]
            if chunk_idx == 0
            else [89.0, 91.0, 92.0, 93.0]
        )
        fullpose_values = (
            [0.0, 1.0, 0.0, 10.0]
            if chunk_idx == 0
            else [0.0, 50.0, 11.0, 12.0]
        )
        transl = torch.tensor([[value, 0.0, 0.0] for value in transl_values], dtype=torch.float32)
        fullpose = torch.zeros((len(fullpose_values), 165), dtype=torch.float32)
        fullpose[:, 0] = torch.tensor(fullpose_values, dtype=torch.float32)
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

    assert stageii_data["trans"][:, 0].tolist() == pytest.approx([0.0, 1.0, 90.0, 100.0, 92.0, 93.0])
    assert stageii_data["stageii_debug_details"]["sequence_chunk_keep_starts"] == [0, 2]


def test_mosh_stageii_torch_sequence_solver_can_override_chunk_keep_start_for_target_indices(
    tmp_path,
    monkeypatch,
):
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
            [0.30, 0.10, 0.15],
            [-0.20, -0.05, 0.10],
        ],
        dtype=torch.float32,
    )
    markers = markers_latent[None, :, :] + marker_offsets[:, None, :].numpy()
    mocap_fname = tmp_path / "synthetic_sequence_keep_start_override.pkl"
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
                "sequence_chunk_size": 4,
                "sequence_chunk_overlap": 2,
                "sequence_optimizer": "adam",
                "sequence_chunk_stitch_mode": "adaptive_transl_jump_pose_guard",
                "sequence_chunk_keep_start_override_indices": 1,
                "sequence_chunk_keep_start_override_value": 1,
            },
        }
    )

    recorded = {"sequence_calls": 0}

    def fake_fit_stageii_sequence_torch(**kwargs):
        chunk_idx = recorded["sequence_calls"]
        recorded["sequence_calls"] += 1
        markers_obs = torch.as_tensor(kwargs["marker_observations"], dtype=torch.float32)
        transl_values = (
            [0.0, 1.0, 90.0, 100.0]
            if chunk_idx == 0
            else [89.0, 91.0, 92.0, 93.0]
        )
        fullpose_values = (
            [0.0, 1.0, 0.0, 10.0]
            if chunk_idx == 0
            else [0.0, 50.0, 11.0, 12.0]
        )
        transl = torch.tensor([[value, 0.0, 0.0] for value in transl_values], dtype=torch.float32)
        fullpose = torch.zeros((len(fullpose_values), 165), dtype=torch.float32)
        fullpose[:, 0] = torch.tensor(fullpose_values, dtype=torch.float32)
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

    assert stageii_data["trans"][:, 0].tolist() == pytest.approx([0.0, 1.0, 90.0, 91.0, 92.0, 93.0])
    assert stageii_data["stageii_debug_details"]["sequence_chunk_keep_starts"] == [0, 1]
    stitch_diagnostics = stageii_data["stageii_debug_details"]["sequence_chunk_stitch_diagnostics"][1]
    assert stitch_diagnostics["selected_keep_start_before_override"] == 2
    assert stitch_diagnostics["selected_keep_start"] == 1
    assert stitch_diagnostics["keep_start_override_value"] == 1
    assert stitch_diagnostics["keep_start_override_applied"] is True
    assert stitch_diagnostics["trim_count"] == 1
    candidate_metrics = stitch_diagnostics["candidate_metrics"]
    keep_start_to_selected = {row["keep_start"]: row["selected"] for row in candidate_metrics}
    assert keep_start_to_selected == {1: True, 2: False}


def test_mosh_stageii_torch_mesh_guarded_stitching_can_keep_legacy_boundary_when_mesh_would_regress(
    tmp_path,
    monkeypatch,
):
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
            [0.30, 0.10, 0.15],
            [-0.20, -0.05, 0.10],
        ],
        dtype=torch.float32,
    )
    markers = markers_latent[None, :, :] + marker_offsets[:, None, :].numpy()
    mocap_fname = tmp_path / "synthetic_sequence_mesh_guarded_stitch.pkl"
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
                "sequence_chunk_size": 4,
                "sequence_chunk_overlap": 2,
                "sequence_optimizer": "adam",
                "sequence_chunk_stitch_mode": "adaptive_transl_jump_pose_mesh_guard",
            },
        }
    )

    recorded = {"sequence_calls": 0}

    def fake_fit_stageii_sequence_torch(**kwargs):
        chunk_idx = recorded["sequence_calls"]
        recorded["sequence_calls"] += 1
        markers_obs = torch.as_tensor(kwargs["marker_observations"], dtype=torch.float32)
        transl_values = (
            [0.0, 1.0, 90.0, 100.0]
            if chunk_idx == 0
            else [89.0, 91.0, 109.0, 110.0]
        )
        fullpose_values = transl_values
        vertex_values = (
            [0.0, 1.0, 90.0, 100.0]
            if chunk_idx == 0
            else [89.0, 120.0, 101.0, 102.0]
        )
        transl = torch.tensor([[value, 0.0, 0.0] for value in transl_values], dtype=torch.float32)
        fullpose = torch.zeros((len(fullpose_values), 165), dtype=torch.float32)
        fullpose[:, 0] = torch.tensor(fullpose_values, dtype=torch.float32)
        vertices = torch.tensor(
            [[[value, 0.0, 0.0]] for value in vertex_values],
            dtype=torch.float32,
        )
        latent_pose = torch.as_tensor(kwargs["latent_pose_init"], dtype=torch.float32)
        if latent_pose.ndim == 2 and latent_pose.shape[0] == 1 and markers_obs.shape[0] > 1:
            latent_pose = latent_pose.expand(markers_obs.shape[0], -1).clone()
        return SimpleNamespace(
            latent_pose=latent_pose,
            fullpose=fullpose,
            transl=transl,
            expression=None,
            predicted_markers=markers_obs.clone(),
            vertices=vertices,
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

    assert stageii_data["trans"][:, 0].tolist() == pytest.approx([0.0, 1.0, 90.0, 100.0, 109.0, 110.0])
    assert stageii_data["stageii_debug_details"]["sequence_chunk_keep_starts"] == [0, 2]


def test_mosh_stageii_torch_sequence_solver_receives_local_transl_window_reference_from_previous_chunk(
    tmp_path,
    monkeypatch,
):
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
            [0.30, 0.05, -0.20],
            [-0.05, -0.10, 0.15],
        ],
        dtype=torch.float32,
    )
    markers = markers_latent[None, :, :] + marker_offsets[:, None, :].numpy()
    mocap_fname = tmp_path / "synthetic_sequence_transl_local_window.pkl"
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
                "sequence_chunk_size": 4,
                "sequence_chunk_overlap": 2,
                "sequence_optimizer": "adam",
                "sequence_seed_refine_iters": 1,
                "sequence_delta_trans": 5.0,
            },
        }
    )

    frame_call_idx = {"value": 0}
    recorded = {"sequence_kwargs": []}

    def fake_fit_stageii_frame_torch(**kwargs):
        call_idx = frame_call_idx["value"]
        frame_call_idx["value"] += 1
        markers_obs = torch.as_tensor(kwargs["marker_observations"], dtype=torch.float32)
        latent_pose = torch.zeros(1, kwargs["layout"].latent_dim, dtype=torch.float32)
        latent_pose[0, 0] = 10.0 + float(call_idx)
        transl = torch.full((1, 3), 20.0 + float(call_idx), dtype=torch.float32)
        return SimpleNamespace(
            latent_pose=latent_pose,
            fullpose=torch.zeros(1, 165, dtype=torch.float32),
            transl=transl,
            expression=None,
            predicted_markers=markers_obs.clone(),
            vertices=markers_obs.unsqueeze(0).clone(),
            joints=torch.zeros(1, 3, 3, dtype=torch.float32),
            loss_terms={"data": 0.0, "poseB": 0.0, "poseH": 0.0, "poseF": 0.0, "expr": 0.0, "velo": 0.0},
        )

    def fake_fit_stageii_sequence_torch(**kwargs):
        chunk_idx = len(recorded["sequence_kwargs"])
        recorded["sequence_kwargs"].append(
            {
                "transl_reference": None
                if kwargs.get("transl_reference") is None
                else torch.as_tensor(kwargs["transl_reference"], dtype=torch.float32).clone(),
            }
        )
        markers_obs = torch.as_tensor(kwargs["marker_observations"], dtype=torch.float32)
        transl = torch.as_tensor(kwargs["transl_init"], dtype=torch.float32).clone()
        latent_pose = torch.as_tensor(kwargs["latent_pose_init"], dtype=torch.float32).clone()
        if chunk_idx == 0:
            transl[:] = torch.tensor(
                [
                    [200.0, 200.0, 200.0],
                    [201.0, 201.0, 201.0],
                    [202.0, 202.0, 202.0],
                    [203.0, 203.0, 203.0],
                ],
                dtype=torch.float32,
            )
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
                "veloT": torch.zeros(markers_obs.shape[0], dtype=torch.float32),
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

    assert stageii_data["fullpose"].shape == (6, 165)
    assert len(recorded["sequence_kwargs"]) == 2
    assert torch.allclose(
        recorded["sequence_kwargs"][1]["transl_reference"],
        torch.tensor(
            [
                [202.0, 202.0, 202.0],
                [203.0, 203.0, 203.0],
                [203.0, 203.0, 203.0],
                [204.0, 204.0, 204.0],
            ],
            dtype=torch.float32,
        ),
        atol=0.0,
    )


def test_splice_chunk_overlap_reference_can_target_mid_overlap_local_window():
    module = _load_chmosh_torch_module()

    base_reference = torch.tensor(
        [
            [20.0, 20.0, 20.0],
            [21.0, 21.0, 21.0],
            [22.0, 22.0, 22.0],
            [23.0, 23.0, 23.0],
        ],
        dtype=torch.float32,
    )
    previous_tail = torch.tensor(
        [
            [200.0, 200.0, 200.0],
            [201.0, 201.0, 201.0],
        ],
        dtype=torch.float32,
    )

    spliced = module._splice_chunk_overlap_reference(
        base_reference,
        previous_tail,
        2,
        include_keep_seam=True,
        keep_seam_window=2,
        window_start=1,
        window_size=3,
    )

    assert torch.allclose(
        spliced,
        torch.tensor(
            [
                [20.0, 20.0, 20.0],
                [201.0, 201.0, 201.0],
                [201.0, 201.0, 201.0],
                [202.0, 202.0, 202.0],
            ],
            dtype=torch.float32,
        ),
        atol=0.0,
    )


def test_mosh_stageii_torch_sequence_solver_can_override_local_transl_window_reference_length(
    tmp_path,
    monkeypatch,
):
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
            [0.30, 0.05, -0.20],
            [-0.05, -0.10, 0.15],
        ],
        dtype=torch.float32,
    )
    markers = markers_latent[None, :, :] + marker_offsets[:, None, :].numpy()
    mocap_fname = tmp_path / "synthetic_sequence_transl_local_window_override.pkl"
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
                "sequence_chunk_size": 4,
                "sequence_chunk_overlap": 2,
                "sequence_optimizer": "adam",
                "sequence_seed_refine_iters": 1,
                "sequence_delta_trans": 5.0,
                "sequence_boundary_transl_delta_reference_window": 1,
            },
        }
    )

    frame_call_idx = {"value": 0}
    recorded = {"sequence_kwargs": []}

    def fake_fit_stageii_frame_torch(**kwargs):
        call_idx = frame_call_idx["value"]
        frame_call_idx["value"] += 1
        markers_obs = torch.as_tensor(kwargs["marker_observations"], dtype=torch.float32)
        latent_pose = torch.zeros(1, kwargs["layout"].latent_dim, dtype=torch.float32)
        latent_pose[0, 0] = 10.0 + float(call_idx)
        transl = torch.full((1, 3), 20.0 + float(call_idx), dtype=torch.float32)
        return SimpleNamespace(
            latent_pose=latent_pose,
            fullpose=torch.zeros(1, 165, dtype=torch.float32),
            transl=transl,
            expression=None,
            predicted_markers=markers_obs.clone(),
            vertices=markers_obs.unsqueeze(0).clone(),
            joints=torch.zeros(1, 3, 3, dtype=torch.float32),
            loss_terms={"data": 0.0, "poseB": 0.0, "poseH": 0.0, "poseF": 0.0, "expr": 0.0, "velo": 0.0},
        )

    def fake_fit_stageii_sequence_torch(**kwargs):
        chunk_idx = len(recorded["sequence_kwargs"])
        recorded["sequence_kwargs"].append(
            {
                "transl_reference": None
                if kwargs.get("transl_reference") is None
                else torch.as_tensor(kwargs["transl_reference"], dtype=torch.float32).clone(),
            }
        )
        markers_obs = torch.as_tensor(kwargs["marker_observations"], dtype=torch.float32)
        transl = torch.as_tensor(kwargs["transl_init"], dtype=torch.float32).clone()
        latent_pose = torch.as_tensor(kwargs["latent_pose_init"], dtype=torch.float32).clone()
        if chunk_idx == 0:
            transl[:] = torch.tensor(
                [
                    [200.0, 200.0, 200.0],
                    [201.0, 201.0, 201.0],
                    [202.0, 202.0, 202.0],
                    [203.0, 203.0, 203.0],
                ],
                dtype=torch.float32,
            )
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
                "veloT": torch.zeros(markers_obs.shape[0], dtype=torch.float32),
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

    assert stageii_data["fullpose"].shape == (6, 165)
    assert len(recorded["sequence_kwargs"]) == 2
    assert torch.allclose(
        recorded["sequence_kwargs"][1]["transl_reference"],
        torch.tensor(
            [
                [202.0, 202.0, 202.0],
                [203.0, 203.0, 203.0],
                [203.0, 203.0, 203.0],
                [25.0, 25.0, 25.0],
            ],
            dtype=torch.float32,
        ),
        atol=0.0,
    )


def test_mosh_stageii_torch_sequence_solver_can_target_local_pose_reference_and_data_window(
    tmp_path,
    monkeypatch,
):
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
            [0.30, 0.05, -0.20],
            [-0.05, -0.10, 0.15],
        ],
        dtype=torch.float32,
    )
    markers = markers_latent[None, :, :] + marker_offsets[:, None, :].numpy()
    mocap_fname = tmp_path / "synthetic_sequence_local_pose_reference_window.pkl"
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
                "sequence_chunk_size": 4,
                "sequence_chunk_overlap": 2,
                "sequence_optimizer": "adam",
                "sequence_seed_refine_iters": 1,
                "sequence_delta_pose": 3.0,
                "sequence_local_window_start": 1,
                "sequence_local_window": 3,
                "sequence_local_pose_reference_from_previous_chunk_indices": 1,
                "sequence_local_data_chunk_indices": 1,
                "sequence_local_data_scale": 0.25,
            },
        }
    )

    frame_call_idx = {"value": 0}
    recorded = {"sequence_kwargs": []}

    def fake_fit_stageii_frame_torch(**kwargs):
        call_idx = frame_call_idx["value"]
        frame_call_idx["value"] += 1
        markers_obs = torch.as_tensor(kwargs["marker_observations"], dtype=torch.float32)
        latent_pose = torch.zeros(1, kwargs["layout"].latent_dim, dtype=torch.float32)
        latent_pose[0, 0] = 10.0 + float(call_idx)
        transl = torch.full((1, 3), 20.0 + float(call_idx), dtype=torch.float32)
        return SimpleNamespace(
            latent_pose=latent_pose,
            fullpose=torch.zeros(1, 165, dtype=torch.float32),
            transl=transl,
            expression=None,
            predicted_markers=markers_obs.clone(),
            vertices=markers_obs.unsqueeze(0).clone(),
            joints=torch.zeros(1, 3, 3, dtype=torch.float32),
            loss_terms={"data": 0.0, "poseB": 0.0, "poseH": 0.0, "poseF": 0.0, "expr": 0.0, "velo": 0.0},
        )

    def fake_fit_stageii_sequence_torch(**kwargs):
        chunk_idx = len(recorded["sequence_kwargs"])
        recorded["sequence_kwargs"].append(
            {
                "latent_pose_reference": None
                if kwargs.get("latent_pose_reference") is None
                else torch.as_tensor(kwargs["latent_pose_reference"], dtype=torch.float32).clone(),
                "marker_data_weights": None
                if kwargs.get("marker_data_weights") is None
                else torch.as_tensor(kwargs["marker_data_weights"], dtype=torch.float32).clone(),
            }
        )
        markers_obs = torch.as_tensor(kwargs["marker_observations"], dtype=torch.float32)
        latent_pose = torch.as_tensor(kwargs["latent_pose_init"], dtype=torch.float32).clone()
        transl = torch.as_tensor(kwargs["transl_init"], dtype=torch.float32).clone()
        if chunk_idx == 0:
            latent_pose[:, 0] = torch.tensor([100.0, 101.0, 102.0, 103.0], dtype=torch.float32)
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
                "veloT": torch.zeros(markers_obs.shape[0], dtype=torch.float32),
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

    assert stageii_data["fullpose"].shape == (6, 165)
    assert len(recorded["sequence_kwargs"]) == 2
    assert recorded["sequence_kwargs"][0]["marker_data_weights"] is None
    assert torch.allclose(
        recorded["sequence_kwargs"][1]["latent_pose_reference"][:, :1],
        torch.tensor([[102.0], [103.0], [103.0], [104.0]], dtype=torch.float32),
        atol=0.0,
    )
    assert torch.allclose(
        recorded["sequence_kwargs"][1]["marker_data_weights"],
        torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0],
                [0.25, 0.25, 0.25, 0.25],
                [0.25, 0.25, 0.25, 0.25],
                [0.25, 0.25, 0.25, 0.25],
            ],
            dtype=torch.float32,
        ),
        atol=0.0,
    )
    assert stageii_data["stageii_debug_details"]["sequence_chunk_stitch_diagnostics"][1][
        "local_pose_reference_from_previous"
    ]
    assert stageii_data["stageii_debug_details"]["sequence_chunk_stitch_diagnostics"][1]["local_data_scale_applied"]


def test_mosh_stageii_torch_sequence_solver_can_limit_local_pose_reference_to_hand_regions(
    tmp_path,
    monkeypatch,
):
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
            [0.30, 0.05, -0.20],
            [-0.05, -0.10, 0.15],
        ],
        dtype=torch.float32,
    )
    markers = markers_latent[None, :, :] + marker_offsets[:, None, :].numpy()
    mocap_fname = tmp_path / "synthetic_sequence_local_pose_reference_regions.pkl"
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
                "sequence_chunk_size": 4,
                "sequence_chunk_overlap": 2,
                "sequence_optimizer": "adam",
                "sequence_seed_refine_iters": 1,
                "sequence_delta_pose": 3.0,
                "sequence_local_window_start": 1,
                "sequence_local_window": 3,
                "sequence_local_pose_reference_from_previous_chunk_indices": 1,
                "sequence_local_pose_reference_regions": "all_hands_pose",
            },
        }
    )

    frame_call_idx = {"value": 0}
    recorded = {"sequence_kwargs": []}

    def fake_fit_stageii_frame_torch(**kwargs):
        call_idx = frame_call_idx["value"]
        frame_call_idx["value"] += 1
        markers_obs = torch.as_tensor(kwargs["marker_observations"], dtype=torch.float32)
        latent_pose = torch.zeros(1, kwargs["layout"].latent_dim, dtype=torch.float32)
        latent_pose[0, 0] = 10.0 + float(call_idx)
        latent_pose[0, 75] = 110.0 + float(call_idx)
        latent_pose[0, 99] = 210.0 + float(call_idx)
        transl = torch.full((1, 3), 20.0 + float(call_idx), dtype=torch.float32)
        return SimpleNamespace(
            latent_pose=latent_pose,
            fullpose=torch.zeros(1, 165, dtype=torch.float32),
            transl=transl,
            expression=None,
            predicted_markers=markers_obs.clone(),
            vertices=markers_obs.unsqueeze(0).clone(),
            joints=torch.zeros(1, 3, 3, dtype=torch.float32),
            loss_terms={"data": 0.0, "poseB": 0.0, "poseH": 0.0, "poseF": 0.0, "expr": 0.0, "velo": 0.0},
        )

    def fake_fit_stageii_sequence_torch(**kwargs):
        chunk_idx = len(recorded["sequence_kwargs"])
        recorded["sequence_kwargs"].append(
            {
                "latent_pose_reference": None
                if kwargs.get("latent_pose_reference") is None
                else torch.as_tensor(kwargs["latent_pose_reference"], dtype=torch.float32).clone(),
            }
        )
        markers_obs = torch.as_tensor(kwargs["marker_observations"], dtype=torch.float32)
        latent_pose = torch.as_tensor(kwargs["latent_pose_init"], dtype=torch.float32).clone()
        transl = torch.as_tensor(kwargs["transl_init"], dtype=torch.float32).clone()
        if chunk_idx == 0:
            latent_pose[:, 0] = torch.tensor([100.0, 101.0, 102.0, 103.0], dtype=torch.float32)
            latent_pose[:, 75] = torch.tensor([300.0, 301.0, 302.0, 303.0], dtype=torch.float32)
            latent_pose[:, 99] = torch.tensor([400.0, 401.0, 402.0, 403.0], dtype=torch.float32)
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
                "veloT": torch.zeros(markers_obs.shape[0], dtype=torch.float32),
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

    assert stageii_data["fullpose"].shape == (6, 165)
    assert len(recorded["sequence_kwargs"]) == 2
    assert torch.allclose(
        recorded["sequence_kwargs"][1]["latent_pose_reference"][:, :1],
        torch.tensor([[102.0], [103.0], [14.0], [15.0]], dtype=torch.float32),
        atol=0.0,
    )
    assert torch.allclose(
        recorded["sequence_kwargs"][1]["latent_pose_reference"][:, 75:76],
        torch.tensor([[302.0], [303.0], [303.0], [304.0]], dtype=torch.float32),
        atol=0.0,
    )
    assert torch.allclose(
        recorded["sequence_kwargs"][1]["latent_pose_reference"][:, 99:100],
        torch.tensor([[402.0], [403.0], [403.0], [404.0]], dtype=torch.float32),
        atol=0.0,
    )
    assert stageii_data["stageii_debug_details"]["sequence_chunk_stitch_diagnostics"][1][
        "local_pose_reference_from_previous"
    ]


def test_mosh_stageii_torch_sequence_solver_can_decouple_local_data_window_from_pose_reference_window(
    tmp_path,
    monkeypatch,
):
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
            [0.30, 0.05, -0.20],
            [-0.05, -0.10, 0.15],
        ],
        dtype=torch.float32,
    )
    markers = markers_latent[None, :, :] + marker_offsets[:, None, :].numpy()
    mocap_fname = tmp_path / "synthetic_sequence_local_data_window_decoupled.pkl"
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
                "sequence_chunk_size": 4,
                "sequence_chunk_overlap": 2,
                "sequence_optimizer": "adam",
                "sequence_seed_refine_iters": 1,
                "sequence_delta_pose": 3.0,
                "sequence_local_window_start": 1,
                "sequence_local_window": 3,
                "sequence_local_pose_reference_from_previous_chunk_indices": 1,
                "sequence_local_data_chunk_indices": 1,
                "sequence_local_data_scale": 0.25,
                "sequence_local_data_window_start": 0,
                "sequence_local_data_window": 3,
            },
        }
    )

    frame_call_idx = {"value": 0}
    recorded = {"sequence_kwargs": []}

    def fake_fit_stageii_frame_torch(**kwargs):
        call_idx = frame_call_idx["value"]
        frame_call_idx["value"] += 1
        markers_obs = torch.as_tensor(kwargs["marker_observations"], dtype=torch.float32)
        latent_pose = torch.zeros(1, kwargs["layout"].latent_dim, dtype=torch.float32)
        latent_pose[0, 0] = 10.0 + float(call_idx)
        transl = torch.full((1, 3), 20.0 + float(call_idx), dtype=torch.float32)
        return SimpleNamespace(
            latent_pose=latent_pose,
            fullpose=torch.zeros(1, 165, dtype=torch.float32),
            transl=transl,
            expression=None,
            predicted_markers=markers_obs.clone(),
            vertices=markers_obs.unsqueeze(0).clone(),
            joints=torch.zeros(1, 3, 3, dtype=torch.float32),
            loss_terms={"data": 0.0, "poseB": 0.0, "poseH": 0.0, "poseF": 0.0, "expr": 0.0, "velo": 0.0},
        )

    def fake_fit_stageii_sequence_torch(**kwargs):
        chunk_idx = len(recorded["sequence_kwargs"])
        recorded["sequence_kwargs"].append(
            {
                "latent_pose_reference": None
                if kwargs.get("latent_pose_reference") is None
                else torch.as_tensor(kwargs["latent_pose_reference"], dtype=torch.float32).clone(),
                "marker_data_weights": None
                if kwargs.get("marker_data_weights") is None
                else torch.as_tensor(kwargs["marker_data_weights"], dtype=torch.float32).clone(),
            }
        )
        markers_obs = torch.as_tensor(kwargs["marker_observations"], dtype=torch.float32)
        latent_pose = torch.as_tensor(kwargs["latent_pose_init"], dtype=torch.float32).clone()
        transl = torch.as_tensor(kwargs["transl_init"], dtype=torch.float32).clone()
        if chunk_idx == 0:
            latent_pose[:, 0] = torch.tensor([100.0, 101.0, 102.0, 103.0], dtype=torch.float32)
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
                "veloT": torch.zeros(markers_obs.shape[0], dtype=torch.float32),
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

    assert stageii_data["fullpose"].shape == (6, 165)
    assert len(recorded["sequence_kwargs"]) == 2
    assert torch.allclose(
        recorded["sequence_kwargs"][1]["latent_pose_reference"][:, :1],
        torch.tensor([[102.0], [103.0], [103.0], [104.0]], dtype=torch.float32),
        atol=0.0,
    )
    assert torch.allclose(
        recorded["sequence_kwargs"][1]["marker_data_weights"],
        torch.tensor(
            [
                [0.25, 0.25, 0.25, 0.25],
                [0.25, 0.25, 0.25, 0.25],
                [0.25, 0.25, 0.25, 0.25],
                [1.0, 1.0, 1.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        atol=0.0,
    )
    assert stageii_data["stageii_debug_details"]["sequence_chunk_stitch_diagnostics"][1][
        "local_pose_reference_from_previous"
    ]
    assert stageii_data["stageii_debug_details"]["sequence_chunk_stitch_diagnostics"][1]["local_data_scale_applied"]


def test_mosh_stageii_torch_sequence_solver_can_limit_local_data_scale_to_finger_markers(
    tmp_path,
    monkeypatch,
):
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
            [0.30, 0.05, -0.20],
            [-0.05, -0.10, 0.15],
        ],
        dtype=torch.float32,
    )
    markers = markers_latent[None, :, :] + marker_offsets[:, None, :].numpy()
    mocap_fname = tmp_path / "synthetic_sequence_local_data_finger_only.pkl"
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
                "sequence_chunk_size": 4,
                "sequence_chunk_overlap": 2,
                "sequence_optimizer": "adam",
                "sequence_seed_refine_iters": 1,
                "sequence_delta_pose": 3.0,
                "sequence_local_window_start": 1,
                "sequence_local_window": 3,
                "sequence_local_pose_reference_from_previous_chunk_indices": 1,
                "sequence_local_data_chunk_indices": 1,
                "sequence_local_data_scale": 0.25,
                "sequence_local_data_window_start": 0,
                "sequence_local_data_window": 3,
                "sequence_local_data_marker_types": "finger",
            },
        }
    )

    frame_call_idx = {"value": 0}
    recorded = {"sequence_kwargs": []}

    def fake_fit_stageii_frame_torch(**kwargs):
        call_idx = frame_call_idx["value"]
        frame_call_idx["value"] += 1
        markers_obs = torch.as_tensor(kwargs["marker_observations"], dtype=torch.float32)
        latent_pose = torch.zeros(1, kwargs["layout"].latent_dim, dtype=torch.float32)
        latent_pose[0, 0] = 10.0 + float(call_idx)
        transl = torch.full((1, 3), 20.0 + float(call_idx), dtype=torch.float32)
        return SimpleNamespace(
            latent_pose=latent_pose,
            fullpose=torch.zeros(1, 165, dtype=torch.float32),
            transl=transl,
            expression=None,
            predicted_markers=markers_obs.clone(),
            vertices=markers_obs.unsqueeze(0).clone(),
            joints=torch.zeros(1, 3, 3, dtype=torch.float32),
            loss_terms={"data": 0.0, "poseB": 0.0, "poseH": 0.0, "poseF": 0.0, "expr": 0.0, "velo": 0.0},
        )

    def fake_fit_stageii_sequence_torch(**kwargs):
        chunk_idx = len(recorded["sequence_kwargs"])
        recorded["sequence_kwargs"].append(
            {
                "marker_data_weights": None
                if kwargs.get("marker_data_weights") is None
                else torch.as_tensor(kwargs["marker_data_weights"], dtype=torch.float32).clone(),
            }
        )
        markers_obs = torch.as_tensor(kwargs["marker_observations"], dtype=torch.float32)
        latent_pose = torch.as_tensor(kwargs["latent_pose_init"], dtype=torch.float32).clone()
        transl = torch.as_tensor(kwargs["transl_init"], dtype=torch.float32).clone()
        if chunk_idx == 0:
            latent_pose[:, 0] = torch.tensor([100.0, 101.0, 102.0, 103.0], dtype=torch.float32)
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
                "veloT": torch.zeros(markers_obs.shape[0], dtype=torch.float32),
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
        marker_meta={
            "marker_type_mask": {
                "finger": [True, False, True, False],
                "body": [False, True, False, True],
            },
            "marker_type": {"A": "finger", "B": "body", "C": "finger", "D": "body"},
            "surface_model_type": "smplx",
        },
        body_model_factory=lambda: TranslOnlyBodyModel(torch.as_tensor(markers_latent, dtype=torch.float32)),
        pose_prior=ZeroPosePrior(63),
        device="cpu",
    )

    assert len(recorded["sequence_kwargs"]) == 2
    assert torch.allclose(
        recorded["sequence_kwargs"][1]["marker_data_weights"],
        torch.tensor(
            [
                [0.25, 1.0, 0.25, 1.0],
                [0.25, 1.0, 0.25, 1.0],
                [0.25, 1.0, 0.25, 1.0],
                [1.0, 1.0, 1.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        atol=0.0,
    )
    stitch_diagnostics = stageii_data["stageii_debug_details"]["sequence_chunk_stitch_diagnostics"][1]
    assert stitch_diagnostics["local_data_scale_applied"]
    assert stitch_diagnostics["local_data_marker_types"] == ["finger"]


def test_mosh_stageii_torch_rejects_unknown_local_data_marker_type(tmp_path):
    module = _load_chmosh_torch_module()

    markers_latent = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    ).numpy()
    latent_labels = ["A", "B"]
    markers = markers_latent[None, :, :]
    mocap_fname = tmp_path / "synthetic_sequence_local_data_invalid_marker_type.pkl"
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
                "sequence_local_data_chunk_indices": 0,
                "sequence_local_data_scale": 0.25,
                "sequence_local_data_window_start": 0,
                "sequence_local_data_window": 1,
                "sequence_local_data_marker_types": "toe",
            },
        }
    )

    with pytest.raises(ValueError, match="sequence_local_data_marker_types"):
        module.mosh_stageii_torch(
            mocap_fname=str(mocap_fname),
            cfg=cfg,
            markers_latent=markers_latent,
            latent_labels=latent_labels,
            betas=torch.zeros(10).numpy(),
            marker_meta={
                "marker_type_mask": {
                    "finger": [True, False],
                    "body": [False, True],
                },
                "marker_type": {"A": "finger", "B": "body"},
                "surface_model_type": "smplx",
            },
            body_model_factory=lambda: TranslOnlyBodyModel(torch.as_tensor(markers_latent, dtype=torch.float32)),
            pose_prior=ZeroPosePrior(63),
            device="cpu",
        )


def test_mosh_stageii_torch_can_record_sequence_probe_for_target_chunk(
    tmp_path,
    monkeypatch,
):
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
            [0.30, 0.05, -0.20],
            [-0.05, -0.10, 0.15],
        ],
        dtype=torch.float32,
    )
    markers = markers_latent[None, :, :] + marker_offsets[:, None, :].numpy()
    mocap_fname = tmp_path / "synthetic_sequence_probe.pkl"
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
                "sequence_chunk_size": 4,
                "sequence_chunk_overlap": 2,
                "sequence_optimizer": "adam",
                "sequence_seed_refine_iters": 1,
                "sequence_probe_chunk_indices": 1,
            },
        }
    )

    frame_call_idx = {"value": 0}

    def fake_fit_stageii_frame_torch(**kwargs):
        call_idx = frame_call_idx["value"]
        frame_call_idx["value"] += 1
        markers_obs = torch.as_tensor(kwargs["marker_observations"], dtype=torch.float32)
        latent_pose = torch.zeros(1, kwargs["layout"].latent_dim, dtype=torch.float32)
        latent_pose[0, 0] = 10.0 + float(call_idx)
        transl = torch.full((1, 3), 20.0 + float(call_idx), dtype=torch.float32)
        return SimpleNamespace(
            latent_pose=latent_pose,
            fullpose=torch.zeros(1, 165, dtype=torch.float32),
            transl=transl,
            expression=None,
            predicted_markers=markers_obs.clone(),
            vertices=markers_obs.unsqueeze(0).clone(),
            joints=torch.zeros(1, 3, 3, dtype=torch.float32),
            loss_terms={"data": 0.0, "poseB": 0.0, "poseH": 0.0, "poseF": 0.0, "expr": 0.0, "velo": 0.0},
        )

    def fake_fit_stageii_sequence_torch(**kwargs):
        markers_obs = torch.as_tensor(kwargs["marker_observations"], dtype=torch.float32)
        latent_pose = torch.as_tensor(kwargs["latent_pose_init"], dtype=torch.float32).clone()
        transl = torch.as_tensor(kwargs["transl_init"], dtype=torch.float32).clone()
        if markers_obs.shape[0] == 4 and float(latent_pose[0, 0].item()) == 12.0:
            latent_pose[:, 0] += torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
            transl[:, 0] += torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32)
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
                "veloT": torch.zeros(markers_obs.shape[0], dtype=torch.float32),
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

    probes = stageii_data["stageii_debug_details"]["sequence_chunk_solver_probes"]
    assert len(probes) == 1
    assert probes[0]["chunk_index"] == 1
    assert probes[0]["row_start"] == 2
    assert probes[0]["row_end"] == 6
    assert probes[0]["overlap_count"] == 2
    assert probes[0]["selected_keep_start"] == 2
    assert probes[0]["latent_delta_overlap_l2"] == pytest.approx([1.0, 2.0])
    assert probes[0]["latent_delta_nonoverlap_mean_l2"] == pytest.approx(3.5)
    assert probes[0]["transl_delta_overlap_l2"] == pytest.approx([0.1, 0.2], abs=1e-6)
    assert probes[0]["transl_delta_nonoverlap_mean_l2"] == pytest.approx(0.35, abs=1e-6)
    assert probes[0]["visible_counts_overlap"] == [4, 4]


def test_mosh_stageii_torch_sequence_solver_receives_local_transl_velocity_window_reference_from_previous_chunk(
    tmp_path,
    monkeypatch,
):
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
            [0.30, 0.05, -0.20],
            [-0.05, -0.10, 0.15],
        ],
        dtype=torch.float32,
    )
    markers = markers_latent[None, :, :] + marker_offsets[:, None, :].numpy()
    mocap_fname = tmp_path / "synthetic_sequence_transl_velocity_local_window.pkl"
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
                "sequence_chunk_size": 4,
                "sequence_chunk_overlap": 2,
                "sequence_optimizer": "adam",
                "sequence_seed_refine_iters": 1,
                "sequence_boundary_transl_velocity_reference": True,
                "sequence_transl_velocity": 7.0,
            },
        }
    )

    frame_call_idx = {"value": 0}
    recorded = {"sequence_kwargs": []}

    def fake_fit_stageii_frame_torch(**kwargs):
        call_idx = frame_call_idx["value"]
        frame_call_idx["value"] += 1
        markers_obs = torch.as_tensor(kwargs["marker_observations"], dtype=torch.float32)
        latent_pose = torch.zeros(1, kwargs["layout"].latent_dim, dtype=torch.float32)
        latent_pose[0, 0] = 10.0 + float(call_idx)
        transl = torch.full((1, 3), 20.0 + float(call_idx), dtype=torch.float32)
        return SimpleNamespace(
            latent_pose=latent_pose,
            fullpose=torch.zeros(1, 165, dtype=torch.float32),
            transl=transl,
            expression=None,
            predicted_markers=markers_obs.clone(),
            vertices=markers_obs.unsqueeze(0).clone(),
            joints=torch.zeros(1, 3, 3, dtype=torch.float32),
            loss_terms={"data": 0.0, "poseB": 0.0, "poseH": 0.0, "poseF": 0.0, "expr": 0.0, "velo": 0.0},
        )

    def fake_fit_stageii_sequence_torch(**kwargs):
        chunk_idx = len(recorded["sequence_kwargs"])
        recorded["sequence_kwargs"].append(
            {
                "weights_transl_velocity": float(getattr(kwargs["weights"], "transl_velocity", 0.0)),
                "transl_velocity_reference": None
                if kwargs.get("transl_velocity_reference") is None
                else torch.as_tensor(kwargs["transl_velocity_reference"], dtype=torch.float32).clone(),
                "transl_velocity_reference_index": kwargs.get("transl_velocity_reference_index"),
            }
        )
        markers_obs = torch.as_tensor(kwargs["marker_observations"], dtype=torch.float32)
        transl = torch.as_tensor(kwargs["transl_init"], dtype=torch.float32).clone()
        latent_pose = torch.as_tensor(kwargs["latent_pose_init"], dtype=torch.float32).clone()
        if chunk_idx == 0:
            transl[:] = torch.tensor(
                [
                    [200.0, 200.0, 200.0],
                    [201.0, 201.0, 201.0],
                    [202.0, 202.0, 202.0],
                    [203.0, 203.0, 203.0],
                ],
                dtype=torch.float32,
            )
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
                "veloT": torch.zeros(markers_obs.shape[0], dtype=torch.float32),
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

    assert stageii_data["fullpose"].shape == (6, 165)
    assert len(recorded["sequence_kwargs"]) == 2
    assert recorded["sequence_kwargs"][1]["weights_transl_velocity"] == pytest.approx(7.0)
    assert torch.allclose(
        recorded["sequence_kwargs"][1]["transl_velocity_reference"],
        torch.tensor(
            [
                [203.0, 203.0, 203.0],
                [204.0, 204.0, 204.0],
                [205.0, 205.0, 205.0],
            ],
            dtype=torch.float32,
        ),
        atol=0.0,
    )
    assert recorded["sequence_kwargs"][1]["transl_velocity_reference_index"] == 2


def test_mosh_stageii_torch_sequence_solver_can_override_local_transl_velocity_window_length(
    tmp_path,
    monkeypatch,
):
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
            [0.30, 0.05, -0.20],
            [-0.05, -0.10, 0.15],
        ],
        dtype=torch.float32,
    )
    markers = markers_latent[None, :, :] + marker_offsets[:, None, :].numpy()
    mocap_fname = tmp_path / "synthetic_sequence_transl_velocity_local_window_override.pkl"
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
                "sequence_chunk_size": 4,
                "sequence_chunk_overlap": 2,
                "sequence_optimizer": "adam",
                "sequence_seed_refine_iters": 1,
                "sequence_boundary_transl_velocity_reference": True,
                "sequence_boundary_transl_velocity_reference_window": 1,
                "sequence_transl_velocity": 7.0,
            },
        }
    )

    frame_call_idx = {"value": 0}
    recorded = {"sequence_kwargs": []}

    def fake_fit_stageii_frame_torch(**kwargs):
        call_idx = frame_call_idx["value"]
        frame_call_idx["value"] += 1
        markers_obs = torch.as_tensor(kwargs["marker_observations"], dtype=torch.float32)
        latent_pose = torch.zeros(1, kwargs["layout"].latent_dim, dtype=torch.float32)
        latent_pose[0, 0] = 10.0 + float(call_idx)
        transl = torch.full((1, 3), 20.0 + float(call_idx), dtype=torch.float32)
        return SimpleNamespace(
            latent_pose=latent_pose,
            fullpose=torch.zeros(1, 165, dtype=torch.float32),
            transl=transl,
            expression=None,
            predicted_markers=markers_obs.clone(),
            vertices=markers_obs.unsqueeze(0).clone(),
            joints=torch.zeros(1, 3, 3, dtype=torch.float32),
            loss_terms={"data": 0.0, "poseB": 0.0, "poseH": 0.0, "poseF": 0.0, "expr": 0.0, "velo": 0.0},
        )

    def fake_fit_stageii_sequence_torch(**kwargs):
        chunk_idx = len(recorded["sequence_kwargs"])
        recorded["sequence_kwargs"].append(
            {
                "transl_velocity_reference": None
                if kwargs.get("transl_velocity_reference") is None
                else torch.as_tensor(kwargs["transl_velocity_reference"], dtype=torch.float32).clone(),
                "transl_velocity_reference_index": kwargs.get("transl_velocity_reference_index"),
            }
        )
        markers_obs = torch.as_tensor(kwargs["marker_observations"], dtype=torch.float32)
        transl = torch.as_tensor(kwargs["transl_init"], dtype=torch.float32).clone()
        latent_pose = torch.as_tensor(kwargs["latent_pose_init"], dtype=torch.float32).clone()
        if chunk_idx == 0:
            transl[:] = torch.tensor(
                [
                    [200.0, 200.0, 200.0],
                    [201.0, 201.0, 201.0],
                    [202.0, 202.0, 202.0],
                    [203.0, 203.0, 203.0],
                ],
                dtype=torch.float32,
            )
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
                "veloT": torch.zeros(markers_obs.shape[0], dtype=torch.float32),
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

    assert stageii_data["fullpose"].shape == (6, 165)
    assert len(recorded["sequence_kwargs"]) == 2
    assert torch.allclose(
        recorded["sequence_kwargs"][1]["transl_velocity_reference"],
        torch.tensor(
            [
                [203.0, 203.0, 203.0],
                [204.0, 204.0, 204.0],
            ],
            dtype=torch.float32,
        ),
        atol=0.0,
    )
    assert recorded["sequence_kwargs"][1]["transl_velocity_reference_index"] == 2


def test_mosh_stageii_torch_sequence_solver_can_zero_seam_transl_velocity_reference_from_previous_chunk(
    tmp_path,
    monkeypatch,
):
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
            [0.30, 0.05, -0.20],
            [-0.05, -0.10, 0.15],
        ],
        dtype=torch.float32,
    )
    markers = markers_latent[None, :, :] + marker_offsets[:, None, :].numpy()
    mocap_fname = tmp_path / "synthetic_sequence_transl_velocity_zero_seam.pkl"
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
                "sequence_chunk_size": 4,
                "sequence_chunk_overlap": 2,
                "sequence_optimizer": "adam",
                "sequence_seed_refine_iters": 1,
                "sequence_boundary_transl_velocity_reference": True,
                "sequence_boundary_transl_velocity_reference_zero_seam": True,
                "sequence_transl_velocity": 7.0,
            },
        }
    )

    frame_call_idx = {"value": 0}
    recorded = {"sequence_kwargs": []}

    def fake_fit_stageii_frame_torch(**kwargs):
        call_idx = frame_call_idx["value"]
        frame_call_idx["value"] += 1
        markers_obs = torch.as_tensor(kwargs["marker_observations"], dtype=torch.float32)
        latent_pose = torch.zeros(1, kwargs["layout"].latent_dim, dtype=torch.float32)
        latent_pose[0, 0] = 10.0 + float(call_idx)
        transl = torch.full((1, 3), 20.0 + float(call_idx), dtype=torch.float32)
        return SimpleNamespace(
            latent_pose=latent_pose,
            fullpose=torch.zeros(1, 165, dtype=torch.float32),
            transl=transl,
            expression=None,
            predicted_markers=markers_obs.clone(),
            vertices=markers_obs.unsqueeze(0).clone(),
            joints=torch.zeros(1, 3, 3, dtype=torch.float32),
            loss_terms={"data": 0.0, "poseB": 0.0, "poseH": 0.0, "poseF": 0.0, "expr": 0.0, "velo": 0.0},
        )

    def fake_fit_stageii_sequence_torch(**kwargs):
        chunk_idx = len(recorded["sequence_kwargs"])
        recorded["sequence_kwargs"].append(
            {
                "transl_velocity_reference": None
                if kwargs.get("transl_velocity_reference") is None
                else torch.as_tensor(kwargs["transl_velocity_reference"], dtype=torch.float32).clone(),
                "transl_velocity_reference_index": kwargs.get("transl_velocity_reference_index"),
            }
        )
        markers_obs = torch.as_tensor(kwargs["marker_observations"], dtype=torch.float32)
        transl = torch.as_tensor(kwargs["transl_init"], dtype=torch.float32).clone()
        latent_pose = torch.as_tensor(kwargs["latent_pose_init"], dtype=torch.float32).clone()
        if chunk_idx == 0:
            transl[:] = torch.tensor(
                [
                    [200.0, 200.0, 200.0],
                    [201.0, 201.0, 201.0],
                    [202.0, 202.0, 202.0],
                    [203.0, 203.0, 203.0],
                ],
                dtype=torch.float32,
            )
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
                "veloT": torch.zeros(markers_obs.shape[0], dtype=torch.float32),
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

    assert stageii_data["fullpose"].shape == (6, 165)
    assert len(recorded["sequence_kwargs"]) == 2
    assert torch.allclose(
        recorded["sequence_kwargs"][1]["transl_velocity_reference"],
        torch.tensor(
            [
                [203.0, 203.0, 203.0],
                [203.0, 203.0, 203.0],
                [204.0, 204.0, 204.0],
            ],
            dtype=torch.float32,
        ),
        atol=0.0,
    )
    assert recorded["sequence_kwargs"][1]["transl_velocity_reference_index"] == 2


def test_mosh_stageii_torch_sequence_solver_passes_boundary_transl_seam_reference_from_previous_chunk(
    tmp_path,
    monkeypatch,
):
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
    mocap_fname = tmp_path / "synthetic_sequence_transl_boundary_seam.pkl"
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
                "sequence_boundary_transl_seam": 11.0,
            },
        }
    )

    frame_call_idx = {"value": 0}
    recorded = {"sequence_kwargs": []}

    def fake_fit_stageii_frame_torch(**kwargs):
        call_idx = frame_call_idx["value"]
        frame_call_idx["value"] += 1
        markers_obs = torch.as_tensor(kwargs["marker_observations"], dtype=torch.float32)
        latent_pose = torch.zeros(1, kwargs["layout"].latent_dim, dtype=torch.float32)
        latent_pose[0, 0] = 10.0 + float(call_idx)
        transl = torch.full((1, 3), 20.0 + float(call_idx), dtype=torch.float32)
        return SimpleNamespace(
            latent_pose=latent_pose,
            fullpose=torch.zeros(1, 165, dtype=torch.float32),
            transl=transl,
            expression=None,
            predicted_markers=markers_obs.clone(),
            vertices=markers_obs.unsqueeze(0).clone(),
            joints=torch.zeros(1, 3, 3, dtype=torch.float32),
            loss_terms={"data": 0.0, "poseB": 0.0, "poseH": 0.0, "poseF": 0.0, "expr": 0.0, "velo": 0.0},
        )

    def fake_fit_stageii_sequence_torch(**kwargs):
        chunk_idx = len(recorded["sequence_kwargs"])
        recorded["sequence_kwargs"].append(
            {
                "weights_boundary_transl_seam": float(getattr(kwargs["weights"], "boundary_transl_seam", 0.0)),
                "transl_boundary_reference": None
                if kwargs.get("transl_boundary_reference") is None
                else torch.as_tensor(kwargs["transl_boundary_reference"], dtype=torch.float32).clone(),
                "transl_boundary_reference_index": kwargs.get("transl_boundary_reference_index"),
            }
        )
        markers_obs = torch.as_tensor(kwargs["marker_observations"], dtype=torch.float32)
        transl = torch.as_tensor(kwargs["transl_init"], dtype=torch.float32).clone()
        latent_pose = torch.as_tensor(kwargs["latent_pose_init"], dtype=torch.float32).clone()
        if chunk_idx == 0:
            transl[:] = torch.tensor(
                [
                    [200.0, 200.0, 200.0],
                    [201.0, 201.0, 201.0],
                ],
                dtype=torch.float32,
            )
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
                "veloT": torch.zeros(markers_obs.shape[0], dtype=torch.float32),
                "seamT": torch.zeros(markers_obs.shape[0], dtype=torch.float32),
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
    assert len(recorded["sequence_kwargs"]) == 3
    assert recorded["sequence_kwargs"][1]["weights_boundary_transl_seam"] == pytest.approx(11.0)
    assert torch.allclose(
        recorded["sequence_kwargs"][1]["transl_boundary_reference"],
        torch.tensor([[201.0, 201.0, 201.0]], dtype=torch.float32),
        atol=0.0,
    )
    assert recorded["sequence_kwargs"][1]["transl_boundary_reference_index"] == 1


def test_mosh_stageii_torch_sequence_solver_passes_boundary_transl_seam_window_from_previous_chunk(
    tmp_path,
    monkeypatch,
):
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
            [0.30, 0.05, -0.20],
            [-0.05, -0.10, 0.15],
        ],
        dtype=torch.float32,
    )
    markers = markers_latent[None, :, :] + marker_offsets[:, None, :].numpy()
    mocap_fname = tmp_path / "synthetic_sequence_transl_boundary_seam_window.pkl"
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
                "sequence_chunk_size": 4,
                "sequence_chunk_overlap": 2,
                "sequence_optimizer": "adam",
                "sequence_seed_refine_iters": 1,
                "sequence_boundary_transl_seam": 11.0,
                "sequence_boundary_transl_seam_window": 2,
            },
        }
    )

    frame_call_idx = {"value": 0}
    recorded = {"sequence_kwargs": []}

    def fake_fit_stageii_frame_torch(**kwargs):
        call_idx = frame_call_idx["value"]
        frame_call_idx["value"] += 1
        markers_obs = torch.as_tensor(kwargs["marker_observations"], dtype=torch.float32)
        latent_pose = torch.zeros(1, kwargs["layout"].latent_dim, dtype=torch.float32)
        latent_pose[0, 0] = 10.0 + float(call_idx)
        transl = torch.full((1, 3), 20.0 + float(call_idx), dtype=torch.float32)
        return SimpleNamespace(
            latent_pose=latent_pose,
            fullpose=torch.zeros(1, 165, dtype=torch.float32),
            transl=transl,
            expression=None,
            predicted_markers=markers_obs.clone(),
            vertices=markers_obs.unsqueeze(0).clone(),
            joints=torch.zeros(1, 3, 3, dtype=torch.float32),
            loss_terms={"data": 0.0, "poseB": 0.0, "poseH": 0.0, "poseF": 0.0, "expr": 0.0, "velo": 0.0},
        )

    def fake_fit_stageii_sequence_torch(**kwargs):
        chunk_idx = len(recorded["sequence_kwargs"])
        recorded["sequence_kwargs"].append(
            {
                "transl_boundary_reference": None
                if kwargs.get("transl_boundary_reference") is None
                else torch.as_tensor(kwargs["transl_boundary_reference"], dtype=torch.float32).clone(),
                "transl_boundary_reference_index": kwargs.get("transl_boundary_reference_index"),
            }
        )
        markers_obs = torch.as_tensor(kwargs["marker_observations"], dtype=torch.float32)
        transl = torch.as_tensor(kwargs["transl_init"], dtype=torch.float32).clone()
        latent_pose = torch.as_tensor(kwargs["latent_pose_init"], dtype=torch.float32).clone()
        if chunk_idx == 0:
            transl[:] = torch.tensor(
                [
                    [200.0, 200.0, 200.0],
                    [201.0, 201.0, 201.0],
                    [202.0, 202.0, 202.0],
                    [203.0, 203.0, 203.0],
                ],
                dtype=torch.float32,
            )
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
                "veloT": torch.zeros(markers_obs.shape[0], dtype=torch.float32),
                "seamT": torch.zeros(markers_obs.shape[0], dtype=torch.float32),
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

    assert stageii_data["fullpose"].shape == (6, 165)
    assert len(recorded["sequence_kwargs"]) == 2
    assert torch.allclose(
        recorded["sequence_kwargs"][1]["transl_boundary_reference"],
        torch.tensor(
            [
                [203.0, 203.0, 203.0],
                [203.0, 203.0, 203.0],
                [204.0, 204.0, 204.0],
            ],
            dtype=torch.float32,
        ),
        atol=0.0,
    )
    assert recorded["sequence_kwargs"][1]["transl_boundary_reference_index"] == 2


def test_mosh_stageii_torch_sequence_solver_can_downweight_boundary_data_window(
    tmp_path,
    monkeypatch,
):
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
            [0.30, 0.05, -0.20],
            [-0.05, -0.10, 0.15],
        ],
        dtype=torch.float32,
    )
    markers = markers_latent[None, :, :] + marker_offsets[:, None, :].numpy()
    mocap_fname = tmp_path / "synthetic_sequence_boundary_data_window.pkl"
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
                "sequence_chunk_size": 4,
                "sequence_chunk_overlap": 2,
                "sequence_optimizer": "adam",
                "sequence_seed_refine_iters": 1,
                "sequence_boundary_data_scale": 0.5,
                "sequence_boundary_data_window": 2,
            },
        }
    )

    frame_call_idx = {"value": 0}
    recorded = {"sequence_kwargs": []}

    def fake_fit_stageii_frame_torch(**kwargs):
        call_idx = frame_call_idx["value"]
        frame_call_idx["value"] += 1
        markers_obs = torch.as_tensor(kwargs["marker_observations"], dtype=torch.float32)
        latent_pose = torch.zeros(1, kwargs["layout"].latent_dim, dtype=torch.float32)
        latent_pose[0, 0] = 10.0 + float(call_idx)
        transl = torch.full((1, 3), 20.0 + float(call_idx), dtype=torch.float32)
        return SimpleNamespace(
            latent_pose=latent_pose,
            fullpose=torch.zeros(1, 165, dtype=torch.float32),
            transl=transl,
            expression=None,
            predicted_markers=markers_obs.clone(),
            vertices=markers_obs.unsqueeze(0).clone(),
            joints=torch.zeros(1, 3, 3, dtype=torch.float32),
            loss_terms={"data": 0.0, "poseB": 0.0, "poseH": 0.0, "poseF": 0.0, "expr": 0.0, "velo": 0.0},
        )

    def fake_fit_stageii_sequence_torch(**kwargs):
        recorded["sequence_kwargs"].append(
            None
            if kwargs.get("marker_data_weights") is None
            else torch.as_tensor(kwargs["marker_data_weights"], dtype=torch.float32).clone()
        )
        markers_obs = torch.as_tensor(kwargs["marker_observations"], dtype=torch.float32)
        transl = torch.as_tensor(kwargs["transl_init"], dtype=torch.float32).clone()
        latent_pose = torch.as_tensor(kwargs["latent_pose_init"], dtype=torch.float32).clone()
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
                "veloT": torch.zeros(markers_obs.shape[0], dtype=torch.float32),
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

    assert stageii_data["fullpose"].shape == (6, 165)
    assert len(recorded["sequence_kwargs"]) == 2
    assert recorded["sequence_kwargs"][0] is None
    assert torch.allclose(
        recorded["sequence_kwargs"][1],
        torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [0.5, 0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5, 0.5],
            ],
            dtype=torch.float32,
        ),
        atol=0.0,
    )


def test_build_chunk_transl_velocity_reference_preserves_seed_boundary_velocity():
    module = _load_chmosh_torch_module()

    base_reference = torch.tensor(
        [
            [22.0, 22.0, 22.0],
            [23.0, 23.0, 23.0],
            [24.0, 24.0, 24.0],
            [25.0, 25.0, 25.0],
        ],
        dtype=torch.float32,
    )
    previous_tail = torch.tensor(
        [
            [202.0, 202.0, 202.0],
            [203.0, 203.0, 203.0],
        ],
        dtype=torch.float32,
    )

    reference = module._build_chunk_transl_velocity_reference(
        base_reference,
        previous_tail,
        2,
        keep_seam_window=2,
    )

    assert torch.allclose(
        reference,
        torch.tensor(
            [
                [203.0, 203.0, 203.0],
                [204.0, 204.0, 204.0],
                [205.0, 205.0, 205.0],
            ],
            dtype=torch.float32,
        ),
        atol=0.0,
    )


def test_build_chunk_transl_boundary_reference_preserves_local_seed_window():
    module = _load_chmosh_torch_module()

    base_reference = torch.tensor(
        [
            [22.0, 22.0, 22.0],
            [23.0, 23.0, 23.0],
            [24.0, 24.0, 24.0],
            [25.0, 25.0, 25.0],
        ],
        dtype=torch.float32,
    )
    previous_tail = torch.tensor(
        [
            [202.0, 202.0, 202.0],
            [203.0, 203.0, 203.0],
        ],
        dtype=torch.float32,
    )

    reference = module._build_chunk_transl_boundary_reference(
        base_reference,
        previous_tail,
        2,
        keep_seam_window=2,
    )

    assert torch.allclose(
        reference,
        torch.tensor(
            [
                [203.0, 203.0, 203.0],
                [203.0, 203.0, 203.0],
                [204.0, 204.0, 204.0],
            ],
            dtype=torch.float32,
        ),
        atol=0.0,
    )


def test_build_chunk_full_transl_velocity_reference_translates_full_seed_profile():
    module = _load_chmosh_torch_module()

    base_reference = torch.tensor(
        [
            [22.0, 22.0, 22.0],
            [23.0, 23.0, 23.0],
            [24.0, 24.0, 24.0],
            [25.0, 25.0, 25.0],
        ],
        dtype=torch.float32,
    )
    previous_tail = torch.tensor(
        [
            [202.0, 202.0, 202.0],
            [203.0, 203.0, 203.0],
        ],
        dtype=torch.float32,
    )

    reference = module._build_chunk_full_transl_velocity_reference(
        base_reference,
        previous_tail,
        2,
    )

    assert torch.allclose(
        reference,
        torch.tensor(
            [
                [202.0, 202.0, 202.0],
                [203.0, 203.0, 203.0],
                [204.0, 204.0, 204.0],
                [205.0, 205.0, 205.0],
            ],
            dtype=torch.float32,
        ),
        atol=0.0,
    )


def test_build_chunk_full_velocity_reference_translates_full_seed_profile():
    module = _load_chmosh_torch_module()

    base_reference = torch.tensor(
        [
            [22.0, 220.0],
            [23.0, 230.0],
            [24.0, 240.0],
            [25.0, 250.0],
        ],
        dtype=torch.float32,
    )
    previous_tail = torch.tensor(
        [
            [202.0, 2002.0],
            [203.0, 2003.0],
        ],
        dtype=torch.float32,
    )

    reference = module._build_chunk_full_velocity_reference(
        base_reference,
        previous_tail,
        2,
    )

    assert torch.allclose(
        reference,
        torch.tensor(
            [
                [202.0, 1993.0],
                [203.0, 2003.0],
                [204.0, 2013.0],
                [205.0, 2023.0],
            ],
            dtype=torch.float32,
        ),
        atol=0.0,
    )


def test_mosh_stageii_torch_sequence_solver_receives_local_velocity_window_reference_from_previous_chunk(
    tmp_path,
    monkeypatch,
):
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
            [0.30, 0.05, -0.20],
            [-0.05, -0.10, 0.15],
        ],
        dtype=torch.float32,
    )
    markers = markers_latent[None, :, :] + marker_offsets[:, None, :].numpy()
    mocap_fname = tmp_path / "synthetic_sequence_velocity_local_window.pkl"
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
                    "stageii_wt_velo": 1.0,
                    "stageii_wt_annealing": 0.0,
                },
            },
            "runtime": {
                "backend": "torch",
                "device": "cpu",
                "sequence_chunk_size": 4,
                "sequence_chunk_overlap": 2,
                "sequence_optimizer": "adam",
                "sequence_seed_refine_iters": 1,
                "sequence_boundary_velocity_reference": True,
            },
        }
    )

    frame_call_idx = {"value": 0}
    recorded = {"sequence_kwargs": []}

    def fake_fit_stageii_frame_torch(**kwargs):
        call_idx = frame_call_idx["value"]
        frame_call_idx["value"] += 1
        markers_obs = torch.as_tensor(kwargs["marker_observations"], dtype=torch.float32)
        latent_pose = torch.zeros(1, kwargs["layout"].latent_dim, dtype=torch.float32)
        latent_pose[0, 0] = 10.0 + float(call_idx)
        transl = torch.full((1, 3), 20.0 + float(call_idx), dtype=torch.float32)
        return SimpleNamespace(
            latent_pose=latent_pose,
            fullpose=torch.zeros(1, 165, dtype=torch.float32),
            transl=transl,
            expression=None,
            predicted_markers=markers_obs.clone(),
            vertices=markers_obs.unsqueeze(0).clone(),
            joints=torch.zeros(1, 3, 3, dtype=torch.float32),
            loss_terms={"data": 0.0, "poseB": 0.0, "poseH": 0.0, "poseF": 0.0, "expr": 0.0, "velo": 0.0},
        )

    def fake_fit_stageii_sequence_torch(**kwargs):
        chunk_idx = len(recorded["sequence_kwargs"])
        recorded["sequence_kwargs"].append(
            {
                "velocity_reference": None
                if kwargs.get("velocity_reference") is None
                else torch.as_tensor(kwargs["velocity_reference"], dtype=torch.float32).clone(),
                "velocity_reference_index": kwargs.get("velocity_reference_index"),
            }
        )
        markers_obs = torch.as_tensor(kwargs["marker_observations"], dtype=torch.float32)
        transl = torch.as_tensor(kwargs["transl_init"], dtype=torch.float32).clone()
        latent_pose = torch.as_tensor(kwargs["latent_pose_init"], dtype=torch.float32).clone()
        if chunk_idx == 0:
            latent_pose[:, 0] = torch.tensor([100.0, 101.0, 102.0, 103.0], dtype=torch.float32)
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
                "veloT": torch.zeros(markers_obs.shape[0], dtype=torch.float32),
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

    assert stageii_data["fullpose"].shape == (6, 165)
    assert len(recorded["sequence_kwargs"]) == 2
    assert recorded["sequence_kwargs"][1]["velocity_reference_index"] == 2
    assert torch.allclose(
        recorded["sequence_kwargs"][1]["velocity_reference"][:, :1],
        torch.tensor([[103.0], [104.0], [105.0]], dtype=torch.float32),
        atol=0.0,
    )


def test_mosh_stageii_torch_sequence_solver_can_request_full_boundary_velocity_profiles(
    tmp_path,
    monkeypatch,
):
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
            [0.30, 0.05, -0.20],
            [-0.05, -0.10, 0.15],
        ],
        dtype=torch.float32,
    )
    markers = markers_latent[None, :, :] + marker_offsets[:, None, :].numpy()
    mocap_fname = tmp_path / "synthetic_sequence_full_velocity_profile.pkl"
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
                    "stageii_wt_velo": 1.0,
                    "stageii_wt_annealing": 0.0,
                },
            },
            "runtime": {
                "backend": "torch",
                "device": "cpu",
                "sequence_chunk_size": 4,
                "sequence_chunk_overlap": 2,
                "sequence_optimizer": "adam",
                "sequence_seed_refine_iters": 1,
                "sequence_boundary_velocity_reference": True,
                "sequence_boundary_velocity_reference_full_length": True,
                "sequence_boundary_transl_velocity_reference": True,
                "sequence_boundary_transl_velocity_reference_full_length": True,
                "sequence_transl_velocity": 7.0,
            },
        }
    )

    frame_call_idx = {"value": 0}
    recorded = {"sequence_kwargs": []}

    def fake_fit_stageii_frame_torch(**kwargs):
        call_idx = frame_call_idx["value"]
        frame_call_idx["value"] += 1
        markers_obs = torch.as_tensor(kwargs["marker_observations"], dtype=torch.float32)
        latent_pose = torch.zeros(1, kwargs["layout"].latent_dim, dtype=torch.float32)
        latent_pose[0, 0] = 10.0 + float(call_idx)
        transl = torch.full((1, 3), 20.0 + float(call_idx), dtype=torch.float32)
        return SimpleNamespace(
            latent_pose=latent_pose,
            fullpose=torch.zeros(1, 165, dtype=torch.float32),
            transl=transl,
            expression=None,
            predicted_markers=markers_obs.clone(),
            vertices=markers_obs.unsqueeze(0).clone(),
            joints=torch.zeros(1, 3, 3, dtype=torch.float32),
            loss_terms={"data": 0.0, "poseB": 0.0, "poseH": 0.0, "poseF": 0.0, "expr": 0.0, "velo": 0.0},
        )

    def fake_fit_stageii_sequence_torch(**kwargs):
        chunk_idx = len(recorded["sequence_kwargs"])
        recorded["sequence_kwargs"].append(
            {
                "velocity_reference": None
                if kwargs.get("velocity_reference") is None
                else torch.as_tensor(kwargs["velocity_reference"], dtype=torch.float32).clone(),
                "velocity_reference_index": kwargs.get("velocity_reference_index"),
                "transl_velocity_reference": None
                if kwargs.get("transl_velocity_reference") is None
                else torch.as_tensor(kwargs["transl_velocity_reference"], dtype=torch.float32).clone(),
                "transl_velocity_reference_index": kwargs.get("transl_velocity_reference_index"),
            }
        )
        markers_obs = torch.as_tensor(kwargs["marker_observations"], dtype=torch.float32)
        transl = torch.as_tensor(kwargs["transl_init"], dtype=torch.float32).clone()
        latent_pose = torch.as_tensor(kwargs["latent_pose_init"], dtype=torch.float32).clone()
        if chunk_idx == 0:
            latent_pose[:, 0] = torch.tensor([100.0, 101.0, 102.0, 103.0], dtype=torch.float32)
            transl[:] = torch.tensor(
                [
                    [200.0, 200.0, 200.0],
                    [201.0, 201.0, 201.0],
                    [202.0, 202.0, 202.0],
                    [203.0, 203.0, 203.0],
                ],
                dtype=torch.float32,
            )
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
                "veloT": torch.zeros(markers_obs.shape[0], dtype=torch.float32),
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

    assert stageii_data["fullpose"].shape == (6, 165)
    assert len(recorded["sequence_kwargs"]) == 2
    assert recorded["sequence_kwargs"][1]["velocity_reference_index"] is None
    assert torch.allclose(
        recorded["sequence_kwargs"][1]["velocity_reference"][:, :1],
        torch.tensor([[102.0], [103.0], [104.0], [105.0]], dtype=torch.float32),
        atol=0.0,
    )
    assert recorded["sequence_kwargs"][1]["transl_velocity_reference_index"] is None
    assert torch.allclose(
        recorded["sequence_kwargs"][1]["transl_velocity_reference"],
        torch.tensor(
            [
                [202.0, 202.0, 202.0],
                [203.0, 203.0, 203.0],
                [204.0, 204.0, 204.0],
                [205.0, 205.0, 205.0],
            ],
            dtype=torch.float32,
        ),
        atol=0.0,
    )
