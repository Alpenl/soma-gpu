import importlib.util
import inspect
import os.path as osp
import pickle
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch
from loguru import logger

from moshpp.marker_layout.labels_map import general_labels_map
from moshpp.models.smplx_torch_wrapper import SmplxTorchWrapper
from moshpp.optim.frame_fit_torch import (
    HandPcaSpec,
    TorchFrameFitOptions,
    TorchFrameFitWeights,
    build_stageii_evaluator,
    decode_stageii_latent_pose,
    fit_stageii_frame_torch,
    make_stageii_latent_layout,
)
try:
    from moshpp.optim.sequence_evaluator_torch import build_stageii_sequence_evaluator
    from moshpp.optim.sequence_fit_torch import (
        TorchSequenceFitOptions,
        TorchSequenceFitWeights,
        fit_stageii_sequence_torch,
    )
except ModuleNotFoundError:
    build_stageii_sequence_evaluator = None
    TorchSequenceFitOptions = None
    TorchSequenceFitWeights = None
    fit_stageii_sequence_torch = None
from moshpp.prior.gmm_prior_torch import prepare_gmm_prior
from moshpp.tools.c3d import Reader as C3DReader
from moshpp.transformed_lm_torch import MarkerAttachment, build_marker_attachment


def _ensure_legacy_pickle_compat():
    legacy_aliases = {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "unicode": str,
        "str": str,
    }
    for name, value in legacy_aliases.items():
        if not hasattr(np, name):
            setattr(np, name, value)
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec


def _load_pickle_compat(path):
    with open(path, "rb") as handle:
        try:
            return pickle.load(handle)
        except UnicodeDecodeError:
            handle.seek(0)
            return pickle.load(handle, encoding="latin1")


def _to_string_list(labels):
    if labels is None:
        return []
    if isinstance(labels, np.ndarray):
        labels = labels.tolist()
    return [label.decode() if isinstance(label, bytes) else str(label) for label in labels]


def _read_mocap_raw(mocap_fname):
    mocap_fname = str(mocap_fname)
    mocap_fname_lower = mocap_fname.lower()
    if mocap_fname_lower.endswith(".pkl"):
        data = _load_pickle_compat(mocap_fname)
        return {
            "markers": np.asarray(data["markers"]),
            "labels": _to_string_list(data.get("labels")),
            "labels_perframe": data.get("labels_perframe"),
            "frame_rate": float(data.get("frame_rate", 120.0)),
        }
    if mocap_fname_lower.endswith(".c3d") or mocap_fname_lower.endswith(".mcp"):
        with open(mocap_fname, "rb") as handle:
            reader = C3DReader(handle)
            labels = [str(label).strip() for label in reader.point_labels]
            frames = []
            for _, points, _ in reader.read_frames(copy=True):
                xyz = points[:, :3].astype(np.float32, copy=True)
                xyz[points[:, 3] < 0] = np.nan
                frames.append(xyz)
            return {
                "markers": np.asarray(frames, dtype=np.float32),
                "labels": labels,
                "labels_perframe": None,
                "frame_rate": float(reader.point_rate),
            }
    raise ValueError(f"Unsupported mocap format: {mocap_fname}")


@dataclass
class TorchMocapSession:
    markers: np.ndarray
    labels: list
    frame_rate: float
    labels_perframe: Optional[np.ndarray] = None

    def __len__(self):
        return self.markers.shape[0]

    def time_length(self):
        return float(len(self) / self.frame_rate)

    def markers_asdict(self):
        frames = []
        for frame_idx, markers in enumerate(self.markers):
            labels = self.labels
            if self.labels_perframe is not None:
                labels = _to_string_list(self.labels_perframe[frame_idx])
            frame_dict = {}
            for label, marker in zip(labels, markers):
                if np.any(np.isnan(marker)) or np.allclose(marker, 0.0):
                    continue
                frame_dict[label] = marker
            frames.append(frame_dict)
        return frames


def load_torch_mocap_session(
    mocap_fname,
    *,
    mocap_unit,
    mocap_rotate=None,
    labels_map=None,
    exclude_markers=None,
    only_markers=None,
):
    raw = _read_mocap_raw(mocap_fname)
    scale = {"mm": 1000.0, "cm": 100.0, "m": 1.0}[mocap_unit]
    markers = raw["markers"].astype(np.float32) / scale
    labels = [label.replace(" ", "") for label in raw["labels"]]
    labels = [label.split(":")[-1] for label in labels]
    if labels_map is not None:
        labels = [labels_map.get(label, label) for label in labels]

    mask = np.ones(len(labels), dtype=bool)
    if only_markers is not None:
        allowed = set(only_markers)
        mask &= np.array([label in allowed for label in labels], dtype=bool)
    else:
        mask &= np.array([not label.startswith("*") for label in labels], dtype=bool)
        if exclude_markers is not None:
            excluded = set(exclude_markers)
            mask &= np.array([label not in excluded for label in labels], dtype=bool)

    labels = [label for label, keep in zip(labels, mask) if keep]
    markers = markers[:, mask]

    if raw["labels_perframe"] is not None:
        labels_perframe = np.asarray(raw["labels_perframe"])[:, mask]
        labels_perframe = np.asarray(
            [
                [labels_map.get(str(label), str(label)) if labels_map is not None else str(label) for label in row]
                for row in labels_perframe
            ]
        )
    else:
        labels_perframe = None

    if mocap_rotate is not None:
        from human_body_prior.tools.rotation_tools import rotate_points_xyz

        markers = rotate_points_xyz(markers, mocap_rotate).reshape(markers.shape)

    return TorchMocapSession(
        markers=markers,
        labels=labels,
        labels_perframe=labels_perframe,
        frame_rate=raw["frame_rate"],
    )


def load_body_pose_prior_torch(pose_body_prior_fname, *, exclude_hands, device):
    if not osp.exists(pose_body_prior_fname):
        raise FileNotFoundError(f"pose_body_prior_fname does not exist: {pose_body_prior_fname}")
    with open(pose_body_prior_fname, "rb") as handle:
        gmm = pickle.load(handle, encoding="latin1")
    npose = 63 if exclude_hands else 69
    means = torch.as_tensor(gmm["means"][:, :npose], dtype=torch.float32, device=device)
    covars = torch.as_tensor(gmm["covars"][:, :npose, :npose], dtype=torch.float32, device=device)
    weights = torch.as_tensor(gmm["weights"], dtype=torch.float32, device=device)
    return prepare_gmm_prior(means, covars, weights)


def load_hand_pca_spec_torch(pose_hand_prior_fname, *, dof_per_hand, use_hands_mean, device):
    if not osp.exists(pose_hand_prior_fname):
        raise FileNotFoundError(f"pose_hand_prior_fname does not exist: {pose_hand_prior_fname}")
    hand_data = np.load(pose_hand_prior_fname)
    left_mean = hand_data["hands_meanl"] if use_hands_mean else np.zeros(45, dtype=np.float32)
    right_mean = hand_data["hands_meanr"] if use_hands_mean else np.zeros(45, dtype=np.float32)
    return HandPcaSpec(
        left_components=torch.as_tensor(hand_data["componentsl"][:dof_per_hand], dtype=torch.float32, device=device),
        right_components=torch.as_tensor(hand_data["componentsr"][:dof_per_hand], dtype=torch.float32, device=device),
        left_mean=torch.as_tensor(left_mean, dtype=torch.float32, device=device),
        right_mean=torch.as_tensor(right_mean, dtype=torch.float32, device=device),
    )


def make_body_model_factory_torch(cfg, *, device):
    model_type = cfg.surface_model.type
    model_fname = Path(str(cfg.surface_model.fname))
    runtime = getattr(cfg, "runtime", None)
    sequence_chunk_size = max(int(_runtime_get(runtime, "sequence_chunk_size", 1) or 1), 1)

    if model_type == "smplx":
        npz_fname = model_fname.with_suffix(".npz")
        if npz_fname.exists():
            from human_body_prior.body_model.body_model import BodyModel

            class HumanBodyPriorAdapter:
                def __init__(self, body_model):
                    self.body_model = body_model

                def __call__(self, **kwargs):
                    pose_hand = torch.cat((kwargs["left_hand_pose"], kwargs["right_hand_pose"]), dim=1)
                    pose_eye = torch.cat((kwargs["leye_pose"], kwargs["reye_pose"]), dim=1)
                    output = self.body_model(
                        root_orient=kwargs["global_orient"],
                        pose_body=kwargs["body_pose"],
                        pose_hand=pose_hand,
                        pose_jaw=kwargs["jaw_pose"],
                        pose_eye=pose_eye,
                        betas=kwargs["betas"],
                        trans=kwargs["transl"],
                        expression=kwargs.get("expression"),
                    )
                    return SimpleNamespace(vertices=output.v, joints=output.Jtr)

            def factory():
                body_model = BodyModel(
                    bm_fname=str(npz_fname),
                    num_betas=cfg.surface_model.num_betas,
                    num_expressions=cfg.surface_model.num_expressions,
                ).to(device)
                return HumanBodyPriorAdapter(body_model)

            return factory

    import smplx

    if model_fname.suffix == ".pkl" and importlib.util.find_spec("scipy") is None:
        npz_fname = model_fname.with_suffix(".npz")
        if npz_fname.exists():
            model_fname = npz_fname
    model_cls = {
        "smpl": smplx.SMPL,
        "smplh": smplx.SMPLH,
        "smplx": smplx.SMPLX,
    }.get(model_type)
    if model_cls is None:
        raise NotImplementedError(f"Unsupported surface_model.type for torch backend: {model_type}")

    ext = model_fname.suffix.lstrip(".")

    def factory():
        if ext == "pkl":
            _ensure_legacy_pickle_compat()
        kwargs = {
            "model_path": str(model_fname),
            "gender": cfg.surface_model.gender,
            "ext": ext,
            "batch_size": sequence_chunk_size,
            "num_betas": cfg.surface_model.num_betas,
            "dtype": torch.float32,
        }
        if model_type in {"smplh", "smplx"}:
            kwargs["use_pca"] = False
            kwargs["flat_hand_mean"] = not cfg.surface_model.use_hands_mean
        if model_type == "smplx":
            kwargs["num_expression_coeffs"] = cfg.surface_model.num_expressions
        model = model_cls(**kwargs)
        return model.to(device)

    return factory


def _runtime_device(cfg, explicit_device=None):
    if explicit_device is not None:
        return explicit_device
    runtime = getattr(cfg, "runtime", None)
    if runtime is not None and getattr(runtime, "device", None):
        return runtime.device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _runtime_get(runtime, key, default=None):
    if runtime is None:
        return default
    return getattr(runtime, key, default)


def _runtime_optional_positive_int(runtime, key):
    value = _runtime_get(runtime, key, None)
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Runtime {key} must be a positive integer when provided. Got {value!r}.") from exc
    if parsed <= 0:
        raise ValueError(f"Runtime {key} must be a positive integer when provided. Got {parsed}.")
    return parsed


def _runtime_stage_fit_options(cfg, runtime):
    valid_optimizers = {"lbfgs", "adam"}

    def _stage_optimizer(name):
        value = str(_runtime_get(runtime, f"{name}_optimizer", "lbfgs")).lower()
        if value not in valid_optimizers:
            raise ValueError(
                f"Unsupported runtime optimizer for {name}: {value}. Expected one of {sorted(valid_optimizers)}."
            )
        return value

    default_maxiter = int(cfg.opt_settings.maxiter)
    options = TorchFrameFitOptions(
        rigid_iters=int(_runtime_get(runtime, "rigid_iters", default_maxiter)),
        warmup_iters=int(_runtime_get(runtime, "warmup_iters", default_maxiter)),
        refine_iters=int(_runtime_get(runtime, "refine_iters", default_maxiter)),
        rigid_lr=float(_runtime_get(runtime, "rigid_lr", 1.0)),
        warmup_lr=float(_runtime_get(runtime, "warmup_lr", 1.0)),
        refine_lr=float(_runtime_get(runtime, "refine_lr", 1.0)),
        rigid_optimizer=_stage_optimizer("rigid"),
        warmup_optimizer=_stage_optimizer("warmup"),
        refine_optimizer=_stage_optimizer("refine"),
        history_size=int(_runtime_get(runtime, "lbfgs_history_size", 100)),
        tolerance_grad=float(_runtime_get(runtime, "lbfgs_tolerance_grad", 1e-7)),
        tolerance_change=float(_runtime_get(runtime, "lbfgs_tolerance_change", 1e-9)),
        max_eval=_runtime_get(runtime, "lbfgs_max_eval", None),
        rigid_max_eval=_runtime_get(runtime, "rigid_max_eval", None),
        warmup_max_eval=_runtime_get(runtime, "warmup_max_eval", None),
        refine_max_eval=_runtime_get(runtime, "refine_max_eval", None),
    )

    stage_max_evals = []
    for stage_max_eval in (options.rigid_max_eval, options.warmup_max_eval, options.refine_max_eval):
        if stage_max_eval is not None:
            stage_max_evals.append(int(stage_max_eval))
    if options.max_eval is None and stage_max_evals and len(set(stage_max_evals)) == 1:
        options.max_eval = stage_max_evals[0]
    return options


def _initial_translation(markers_obs, markers_latent, visible_mask=None):
    if visible_mask is not None:
        visible_mask = torch.as_tensor(visible_mask, dtype=torch.bool, device=markers_obs.device)
        if visible_mask.ndim != 1:
            raise ValueError(f"visible_mask must be 1D, got {tuple(visible_mask.shape)}")
        if int(visible_mask.sum().item()) == 0:
            raise ValueError("visible_mask must keep at least one marker.")
        markers_obs = markers_obs[visible_mask]
        markers_latent = markers_latent[visible_mask]
    return (markers_obs.mean(dim=0, keepdim=True) - markers_latent.mean(dim=0, keepdim=True)).to(torch.float32)


def _subset_attachment(attachment, marker_ids):
    return MarkerAttachment(
        closest=attachment.closest[marker_ids],
        coeffs=attachment.coeffs[marker_ids],
    )


def _sequence_chunk_ranges(total_frames, chunk_size, overlap):
    if total_frames <= 0:
        return []
    chunk_size = max(int(chunk_size), 1)
    overlap = max(int(overlap), 0)
    step = max(chunk_size - overlap, 1)
    ranges = []
    start = 0
    while start < total_frames:
        end = min(start + chunk_size, total_frames)
        ranges.append((start, end))
        if end >= total_frames:
            break
        start += step
    return ranges


def _runtime_sequence_chunk_stitch_mode(runtime):
    stitch_mode = _runtime_get(runtime, "sequence_chunk_stitch_mode", "keep_overlap_tail")
    if stitch_mode is None:
        stitch_mode = "keep_overlap_tail"
    stitch_mode = str(stitch_mode)
    valid_modes = {
        "keep_overlap_tail",
        "adaptive_transl_jump",
        "adaptive_transl_jump_pose_guard",
        "adaptive_transl_jump_pose_mesh_guard",
    }
    if stitch_mode not in valid_modes:
        raise ValueError(
            f"Runtime sequence_chunk_stitch_mode must be one of {sorted(valid_modes)}. Got {stitch_mode!r}."
        )
    return stitch_mode


def _select_chunk_keep_start(
    *,
    stitch_mode,
    overlap_count,
    previous_fullpose_tail,
    previous_transl_tail,
    previous_vertices_tail=None,
    current_fullpose,
    current_transl,
    current_vertices=None,
    return_diagnostics=False,
):
    overlap_count = max(int(overlap_count), 0)
    default_keep_start = overlap_count
    max_keep_start = 0
    default_pose_jump = None
    default_mesh_jump = None
    candidate_metrics = []

    def _finalize(selected_keep_start):
        selected_keep_start = int(selected_keep_start)
        if not return_diagnostics:
            return selected_keep_start
        selected_metric = None
        for row in candidate_metrics:
            row["selected"] = int(row["keep_start"]) == selected_keep_start
            if row["selected"]:
                selected_metric = row
        return selected_keep_start, {
            "stitch_mode": stitch_mode,
            "overlap_count": overlap_count,
            "max_keep_start": max_keep_start,
            "default_keep_start": default_keep_start,
            "selected_keep_start": selected_keep_start,
            "default_pose_jump": default_pose_jump,
            "default_mesh_jump": default_mesh_jump,
            "selected_transl_jump": None if selected_metric is None else selected_metric["transl_jump"],
            "selected_pose_jump": None if selected_metric is None else selected_metric["pose_jump"],
            "selected_mesh_jump": None if selected_metric is None else selected_metric["mesh_jump"],
            "candidate_metrics": candidate_metrics,
        }

    if overlap_count <= 0:
        return _finalize(0)
    if stitch_mode == "keep_overlap_tail":
        return _finalize(default_keep_start)
    if previous_transl_tail is None or current_transl is None:
        return _finalize(default_keep_start)

    previous_transl_tail = torch.as_tensor(previous_transl_tail, dtype=torch.float32)
    current_transl = torch.as_tensor(current_transl, dtype=torch.float32)
    if previous_transl_tail.ndim < 2 or current_transl.ndim < 2:
        return _finalize(default_keep_start)

    max_keep_start = min(overlap_count, previous_transl_tail.shape[0], current_transl.shape[0] - 1)
    if max_keep_start < 1:
        return _finalize(default_keep_start)

    previous_fullpose_tail = (
        None
        if previous_fullpose_tail is None
        else torch.as_tensor(previous_fullpose_tail, dtype=torch.float32)
    )
    current_fullpose = None if current_fullpose is None else torch.as_tensor(current_fullpose, dtype=torch.float32)
    previous_vertices_tail = (
        None
        if previous_vertices_tail is None
        else torch.as_tensor(previous_vertices_tail, dtype=torch.float32)
    )
    current_vertices = None if current_vertices is None else torch.as_tensor(current_vertices, dtype=torch.float32)
    default_pose_jump = None
    requires_pose_guard = stitch_mode in {"adaptive_transl_jump_pose_guard", "adaptive_transl_jump_pose_mesh_guard"}
    if requires_pose_guard:
        if (
            previous_fullpose_tail is None
            or current_fullpose is None
            or previous_fullpose_tail.ndim < 2
            or current_fullpose.ndim < 2
            or previous_fullpose_tail.shape[0] < default_keep_start
            or current_fullpose.shape[0] <= default_keep_start
        ):
            return _finalize(default_keep_start)
        default_pose_jump = float(
            torch.linalg.vector_norm(
                current_fullpose[default_keep_start] - previous_fullpose_tail[default_keep_start - 1]
            ).item()
        )

    default_mesh_jump = None
    if stitch_mode == "adaptive_transl_jump_pose_mesh_guard":
        if (
            previous_vertices_tail is None
            or current_vertices is None
            or previous_vertices_tail.ndim < 3
            or current_vertices.ndim < 3
            or previous_vertices_tail.shape[0] < default_keep_start
            or current_vertices.shape[0] <= default_keep_start
        ):
            return _finalize(default_keep_start)
        default_mesh_jump = float(
            torch.linalg.vector_norm(
                current_vertices[default_keep_start] - previous_vertices_tail[default_keep_start - 1]
            ).item()
        )

    best_keep_start = default_keep_start
    best_score = None
    for keep_start in range(1, max_keep_start + 1):
        transl_jump = float(
            torch.linalg.vector_norm(current_transl[keep_start] - previous_transl_tail[keep_start - 1]).item()
        )
        pose_jump = 0.0
        if (
            previous_fullpose_tail is not None
            and current_fullpose is not None
            and previous_fullpose_tail.ndim >= 2
            and current_fullpose.ndim >= 2
            and previous_fullpose_tail.shape[0] >= keep_start
            and current_fullpose.shape[0] > keep_start
        ):
            pose_jump = float(
                torch.linalg.vector_norm(current_fullpose[keep_start] - previous_fullpose_tail[keep_start - 1]).item()
            )
        mesh_jump = 0.0
        if (
            previous_vertices_tail is not None
            and current_vertices is not None
            and previous_vertices_tail.ndim >= 3
            and current_vertices.ndim >= 3
            and previous_vertices_tail.shape[0] >= keep_start
            and current_vertices.shape[0] > keep_start
        ):
            mesh_jump = float(
                torch.linalg.vector_norm(current_vertices[keep_start] - previous_vertices_tail[keep_start - 1]).item()
            )
        passed_pose_guard = default_pose_jump is None or pose_jump <= default_pose_jump
        passed_mesh_guard = default_mesh_jump is None or mesh_jump <= default_mesh_jump
        candidate_metrics.append(
            {
                "keep_start": int(keep_start),
                "transl_jump": transl_jump,
                "pose_jump": pose_jump,
                "mesh_jump": mesh_jump,
                "passed_pose_guard": passed_pose_guard,
                "passed_mesh_guard": passed_mesh_guard,
                "eligible": passed_pose_guard and passed_mesh_guard,
                "selected": False,
            }
        )
        if not passed_pose_guard or not passed_mesh_guard:
            continue
        score = (transl_jump, pose_jump, mesh_jump, -keep_start)
        if best_score is None or score < best_score:
            best_score = score
            best_keep_start = keep_start
    return _finalize(best_keep_start)


def _trim_perframe_tail(perframe_data, trim_count):
    trim_count = max(int(trim_count), 0)
    if trim_count <= 0:
        return
    for key, values in perframe_data.items():
        if key == "stageii_errs":
            for loss_history in values.values():
                del loss_history[-trim_count:]
            continue
        del values[-trim_count:]


def _build_chunk_observations(*, chunk_frames, observed_markers_dict, latent_labels, label_to_latent_id, device):
    chunk_markers = []
    chunk_visible = []
    for frame_idx in chunk_frames:
        frame_obs = observed_markers_dict[frame_idx]
        if len(frame_obs) == 0:
            raise ValueError(f"no available observed markers for frame {frame_idx}.")
        frame_markers = torch.zeros(len(latent_labels), 3, dtype=torch.float32, device=device)
        frame_visible = torch.zeros(len(latent_labels), dtype=torch.bool, device=device)
        for label, marker in frame_obs.items():
            latent_id = label_to_latent_id.get(label)
            if latent_id is None:
                continue
            frame_markers[latent_id] = torch.as_tensor(marker, dtype=torch.float32, device=device)
            frame_visible[latent_id] = True
        if not bool(frame_visible.any()):
            raise ValueError(f"no latent-aligned observed markers for frame {frame_idx}.")
        chunk_markers.append(frame_markers)
        chunk_visible.append(frame_visible)
    return torch.stack(chunk_markers, dim=0), torch.stack(chunk_visible, dim=0)


def _runtime_sequence_fit_options(cfg, runtime):
    if TorchSequenceFitOptions is None:
        raise RuntimeError("sequence fit options requested but torch sequence solver is unavailable")

    valid_optimizers = {"lbfgs", "adam"}
    default_optimizer = str(_runtime_get(runtime, "refine_optimizer", "adam")).lower()
    optimizer = str(_runtime_get(runtime, "sequence_optimizer", default_optimizer)).lower()
    if optimizer not in valid_optimizers:
        raise ValueError(f"Unsupported runtime optimizer for sequence: {optimizer}. Expected one of {sorted(valid_optimizers)}.")

    default_max_iters = int(_runtime_get(runtime, "refine_iters", int(cfg.opt_settings.maxiter)))
    default_lr = float(_runtime_get(runtime, "refine_lr", 1e-1))
    return TorchSequenceFitOptions(
        max_iters=int(_runtime_get(runtime, "sequence_max_iters", default_max_iters)),
        lr=float(_runtime_get(runtime, "sequence_lr", default_lr)),
        optimizer=optimizer,
        history_size=int(_runtime_get(runtime, "sequence_history_size", _runtime_get(runtime, "lbfgs_history_size", 100))),
        tolerance_grad=float(
            _runtime_get(runtime, "sequence_tolerance_grad", _runtime_get(runtime, "lbfgs_tolerance_grad", 1e-7))
        ),
        tolerance_change=float(
            _runtime_get(runtime, "sequence_tolerance_change", _runtime_get(runtime, "lbfgs_tolerance_change", 1e-9))
        ),
        max_eval=_runtime_get(runtime, "sequence_max_eval", _runtime_get(runtime, "lbfgs_max_eval", None)),
    )


def _runtime_sequence_seed_options(options, runtime):
    seed_iters = int(_runtime_get(runtime, "sequence_seed_refine_iters", 0) or 0)
    if seed_iters <= 0:
        return None

    valid_optimizers = {"lbfgs", "adam"}
    refine_optimizer = str(_runtime_get(runtime, "sequence_seed_refine_optimizer", options.optimizer)).lower()
    if refine_optimizer not in valid_optimizers:
        raise ValueError(
            f"Unsupported runtime optimizer for sequence seed: {refine_optimizer}. Expected one of {sorted(valid_optimizers)}."
        )
    refine_max_eval = _runtime_get(runtime, "sequence_seed_refine_max_eval", options.max_eval)
    return TorchFrameFitOptions(
        rigid_iters=0,
        warmup_iters=0,
        refine_iters=seed_iters,
        rigid_lr=1.0,
        warmup_lr=1.0,
        refine_lr=float(_runtime_get(runtime, "sequence_seed_refine_lr", options.lr)),
        rigid_optimizer="lbfgs",
        warmup_optimizer="lbfgs",
        refine_optimizer=refine_optimizer,
        history_size=options.history_size,
        tolerance_grad=options.tolerance_grad,
        tolerance_change=options.tolerance_change,
        max_eval=refine_max_eval,
        rigid_max_eval=None,
        warmup_max_eval=None,
        refine_max_eval=refine_max_eval,
    )


def _runtime_sequence_fit_weights(cfg, runtime, *, avg_visible_count, marker_count, anneal_factor):
    if TorchSequenceFitWeights is None:
        raise RuntimeError("sequence fit weights requested but torch sequence solver is unavailable")

    weight_cfg = cfg.opt_settings.weights
    return TorchSequenceFitWeights(
        data=float(weight_cfg.stageii_wt_data) * (46 / max(int(avg_visible_count), 1)),
        pose_body=float(weight_cfg.stageii_wt_poseB) * anneal_factor,
        pose_hand=float(weight_cfg.stageii_wt_poseH) * anneal_factor,
        pose_face=float(weight_cfg.stageii_wt_poseF) * anneal_factor,
        expr=float(weight_cfg.stageii_wt_expr),
        velocity=float(weight_cfg.stageii_wt_velo),
        transl_velocity=float(_runtime_get(runtime, "sequence_transl_velocity", 0.0)),
        boundary_transl_seam=float(_runtime_get(runtime, "sequence_boundary_transl_seam", 0.0)),
        temporal_accel=float(_runtime_get(runtime, "sequence_temporal_accel", 0.0)),
        delta_pose=float(_runtime_get(runtime, "sequence_delta_pose", 0.0)),
        delta_trans=float(_runtime_get(runtime, "sequence_delta_trans", 0.0)),
        delta_expr=float(_runtime_get(runtime, "sequence_delta_expr", 0.0)),
    )


def _loss_history_to_numpy(values):
    normalized = []
    for value in values:
        if torch.is_tensor(value):
            normalized.append(float(value.detach().item()))
        else:
            normalized.append(float(value))
    return np.asarray(normalized, dtype=np.float32)


def _build_sequence_seed_cache(
    *,
    selected_frames,
    sequence_markers_obs,
    sequence_visible,
    current_latent_pose,
    current_transl,
    current_expression,
    prev_latent_pose,
    body_model,
    wrapper,
    betas_tensor,
    marker_attachment,
    pose_prior,
    layout,
    hand_pca,
    optimize_fingers,
    optimize_face,
    cfg,
    markers_latent_tensor,
    evaluator,
    seed_options,
):
    if seed_options is None:
        return None

    seed_latent_pose = current_latent_pose.detach().clone()
    seed_transl = current_transl.detach().clone()
    seed_expression = current_expression.detach().clone() if current_expression is not None else None
    seed_prev_latent = prev_latent_pose.detach().clone() if prev_latent_pose is not None else None

    cached_latent_init = []
    cached_transl_init = []
    cached_expression_init = [] if optimize_face else None

    for frame_pos, _frame_idx in enumerate(selected_frames):
        frame_visible = sequence_visible[frame_pos]
        visible_count = int(frame_visible.sum().item())
        num_missing_markers = float(len(markers_latent_tensor) - visible_count)
        anneal_factor = 1.0
        if num_missing_markers > 0:
            anneal_factor += (num_missing_markers / len(markers_latent_tensor)) * cfg.opt_settings.weights.stageii_wt_annealing

        frame_weights = TorchFrameFitWeights(
            data=cfg.opt_settings.weights.stageii_wt_data * (46 / max(visible_count, 1)),
            pose_body=cfg.opt_settings.weights.stageii_wt_poseB * anneal_factor,
            pose_hand=cfg.opt_settings.weights.stageii_wt_poseH * anneal_factor,
            pose_face=cfg.opt_settings.weights.stageii_wt_poseF * anneal_factor,
            expr=cfg.opt_settings.weights.stageii_wt_expr,
            velocity=cfg.opt_settings.weights.stageii_wt_velo,
        )

        if frame_pos == 0:
            seed_transl = _initial_translation(
                sequence_markers_obs[frame_pos],
                markers_latent_tensor,
                visible_mask=frame_visible,
            )

        seed_result = fit_stageii_frame_torch(
            body_model=body_model,
            wrapper=wrapper,
            betas=betas_tensor,
            marker_attachment=marker_attachment,
            marker_observations=sequence_markers_obs[frame_pos],
            pose_prior=pose_prior,
            layout=layout,
            latent_pose_init=seed_latent_pose,
            transl_init=seed_transl,
            expression_init=seed_expression,
            hand_pca=hand_pca,
            optimize_fingers=optimize_fingers,
            optimize_face=optimize_face,
            optimize_toes=bool(cfg.moshpp.optimize_toes),
            velocity_reference=seed_prev_latent,
            weights=frame_weights,
            options=seed_options,
            rigid_init=False,
            warmup_pose_scales=(),
            evaluator=evaluator,
        )

        seed_latent_pose = seed_result.latent_pose.detach()
        seed_transl = seed_result.transl.detach()
        seed_expression = seed_result.expression.detach() if seed_result.expression is not None else None
        seed_prev_latent = seed_result.latent_pose.detach()

        cached_latent_init.append(seed_latent_pose[0])
        cached_transl_init.append(seed_transl[0])
        if optimize_face:
            if seed_expression is None:
                cached_expression_init.append(
                    torch.zeros(cfg.surface_model.num_expressions, dtype=torch.float32, device=seed_latent_pose.device)
                )
            else:
                cached_expression_init.append(seed_expression[0])

    return (
        torch.stack(cached_latent_init, dim=0),
        torch.stack(cached_transl_init, dim=0),
        torch.stack(cached_expression_init, dim=0) if optimize_face else None,
    )


def _splice_chunk_overlap_reference(
    base_reference,
    previous_tail,
    overlap_count,
    *,
    include_keep_seam=False,
    keep_seam_window=0,
):
    if base_reference is None or previous_tail is None or overlap_count <= 0:
        return base_reference
    reference = base_reference.detach().clone()
    reference[:overlap_count] = previous_tail[-overlap_count:].detach().clone()
    if include_keep_seam and overlap_count < reference.shape[0]:
        keep_seam_window = max(int(keep_seam_window or 1), 1)
        seam_window_end = min(overlap_count + keep_seam_window, reference.shape[0])
        seam_window = base_reference[overlap_count:seam_window_end].detach().clone()
        seam_window = seam_window - seam_window[:1].clone()
        reference[overlap_count:seam_window_end] = previous_tail[-1:].detach().clone() + seam_window
    return reference


def _build_chunk_velocity_reference(base_reference, previous_tail, overlap_count, *, keep_seam_window=0):
    if previous_tail is None:
        return None
    anchor = previous_tail[-1:].detach().clone()
    if base_reference is None or overlap_count <= 0:
        return anchor
    keep_seam_window = max(int(keep_seam_window or 1), 1)
    if overlap_count >= base_reference.shape[0]:
        return anchor
    seam_window_end = min(overlap_count + keep_seam_window, base_reference.shape[0])
    seam_positions = base_reference[overlap_count - 1 : seam_window_end].detach().clone()
    if seam_positions.shape[0] == 0:
        return anchor
    seam_positions = seam_positions - seam_positions[:1].clone()
    return anchor + seam_positions


def _build_chunk_full_velocity_reference(base_reference, previous_tail, overlap_count):
    if previous_tail is None:
        return None
    if base_reference is None or overlap_count <= 0:
        return previous_tail[-1:].detach().clone()
    if overlap_count > base_reference.shape[0]:
        raise ValueError(
            f"overlap_count={overlap_count} exceeds base reference length {base_reference.shape[0]}"
        )
    reference = base_reference.detach().clone()
    seam_anchor = previous_tail[-1:].detach().clone()
    seam_shift = seam_anchor - reference[overlap_count - 1 : overlap_count].clone()
    return reference + seam_shift


def _build_chunk_transl_velocity_reference(base_reference, previous_tail, overlap_count, *, keep_seam_window=0):
    return _build_chunk_velocity_reference(
        base_reference,
        previous_tail,
        overlap_count,
        keep_seam_window=keep_seam_window,
    )


def _build_chunk_zero_seam_transl_velocity_reference(base_reference, previous_tail, overlap_count, *, keep_seam_window=0):
    if previous_tail is None:
        return None
    anchor = previous_tail[-1:].detach().clone()
    if base_reference is None or overlap_count <= 0:
        return anchor
    keep_seam_window = max(int(keep_seam_window or 1), 1)
    if overlap_count >= base_reference.shape[0]:
        return anchor
    seam_window_end = min(overlap_count + keep_seam_window, base_reference.shape[0])
    seam_positions = base_reference[overlap_count:seam_window_end].detach().clone()
    if seam_positions.shape[0] == 0:
        return anchor
    seam_positions = seam_positions - seam_positions[:1].clone()
    return torch.cat((anchor, anchor + seam_positions), dim=0)


def _build_chunk_transl_boundary_reference(base_reference, previous_tail, overlap_count, *, keep_seam_window=0):
    return _build_chunk_zero_seam_transl_velocity_reference(
        base_reference,
        previous_tail,
        overlap_count,
        keep_seam_window=keep_seam_window,
    )


def _build_chunk_full_transl_velocity_reference(base_reference, previous_tail, overlap_count):
    return _build_chunk_full_velocity_reference(
        base_reference,
        previous_tail,
        overlap_count,
    )


def mosh_stageii_torch(
    mocap_fname: str,
    cfg,
    markers_latent,
    latent_labels,
    betas,
    marker_meta,
    v_template_fname=None,
    *,
    body_model_factory=None,
    pose_prior=None,
    hand_pca=None,
    device=None,
):
    del v_template_fname
    num_train_markers = 46
    if cfg.surface_model.type != "smplx":
        raise NotImplementedError("The torch backend currently supports surface_model.type='smplx' only.")

    device = _runtime_device(cfg, explicit_device=device)
    mocap = load_torch_mocap_session(
        mocap_fname,
        mocap_unit=cfg.mocap.unit,
        mocap_rotate=cfg.mocap.rotate,
        labels_map=general_labels_map,
        exclude_markers=getattr(cfg.mocap, "exclude_markers", None),
        only_markers=None,
    )
    observed_markers_dict = mocap.markers_asdict()

    optimize_fingers = bool(cfg.moshpp.optimize_fingers)
    optimize_face = bool(cfg.moshpp.optimize_face)

    if optimize_fingers and not np.any(["finger" in part for part in marker_meta["marker_type_mask"].keys()]):
        optimize_fingers = False
    if optimize_face and not np.any(["face" in part for part in marker_meta["marker_type_mask"].keys()]):
        optimize_face = False

    if body_model_factory is None:
        body_model_factory = make_body_model_factory_torch(cfg, device=device)
    if pose_prior is None:
        pose_prior = load_body_pose_prior_torch(
            cfg.moshpp.pose_body_prior_fname,
            exclude_hands=True,
            device=device,
        )
    if hand_pca is None and optimize_fingers:
        hand_pca = load_hand_pca_spec_torch(
            cfg.moshpp.pose_hand_prior_fname,
            dof_per_hand=cfg.surface_model.dof_per_hand,
            use_hands_mean=cfg.surface_model.use_hands_mean,
            device=device,
        )

    body_model = body_model_factory()
    layout = make_stageii_latent_layout(
        surface_model_type=cfg.surface_model.type,
        dof_per_hand=cfg.surface_model.dof_per_hand,
        optimize_fingers=optimize_fingers,
        optimize_face=optimize_face,
    )

    betas_tensor = torch.as_tensor(betas[: cfg.surface_model.num_betas], dtype=torch.float32, device=device).unsqueeze(0)
    markers_latent_tensor = torch.as_tensor(markers_latent, dtype=torch.float32, device=device)
    wrapper = SmplxTorchWrapper(body_model=body_model, surface_model_type=cfg.surface_model.type)
    canonical_fullpose = decode_stageii_latent_pose(
        torch.zeros(1, layout.latent_dim, dtype=torch.float32, device=device),
        layout,
        hand_pca=hand_pca,
    )
    canonical_output = wrapper(
        fullpose=canonical_fullpose,
        betas=betas_tensor,
        transl=torch.zeros(1, 3, dtype=torch.float32, device=device),
    )
    marker_attachment = build_marker_attachment(
        canonical_output.vertices[0].detach().cpu(),
        markers_latent_tensor.detach().cpu(),
        surface_model_type=cfg.surface_model.type,
    ).to(device=device, dtype=torch.float32)
    runtime = getattr(cfg, "runtime", None)
    perframe_data = {
        "markers_sim": [],
        "markers_obs": [],
        "labels_obs": [],
        "fullpose": [],
        "trans": [],
        "stageii_errs": {},
    }

    selected_frames = list(
        range(
        cfg.mocap.start_fidx,
        len(mocap) if cfg.mocap.end_fidx == -1 else cfg.mocap.end_fidx,
        cfg.mocap.ds_rate,
        )
    )

    current_latent_pose = torch.zeros(1, layout.latent_dim, dtype=torch.float32, device=device)
    current_transl = torch.zeros(1, 3, dtype=torch.float32, device=device)
    current_expression = (
        torch.zeros(1, cfg.surface_model.num_expressions, dtype=torch.float32, device=device)
        if optimize_face
        else None
    )
    prev_latent_pose = None
    prev_transl = None
    label_to_latent_id = {label: idx for idx, label in enumerate(latent_labels)}
    sequence_chunk_size = max(int(_runtime_get(runtime, "sequence_chunk_size", 1) or 1), 1)
    sequence_chunk_overlap = max(int(_runtime_get(runtime, "sequence_chunk_overlap", 0) or 0), 0)
    sequence_chunk_stitch_mode = _runtime_sequence_chunk_stitch_mode(runtime)
    sequence_boundary_velocity_reference = bool(_runtime_get(runtime, "sequence_boundary_velocity_reference", False))
    sequence_boundary_velocity_reference_full_length = bool(
        _runtime_get(runtime, "sequence_boundary_velocity_reference_full_length", False)
    )
    sequence_boundary_transl_velocity_reference = bool(
        _runtime_get(runtime, "sequence_boundary_transl_velocity_reference", False)
    )
    sequence_boundary_transl_velocity_reference_full_length = bool(
        _runtime_get(runtime, "sequence_boundary_transl_velocity_reference_full_length", False)
    )
    sequence_boundary_transl_velocity_reference_zero_seam = bool(
        _runtime_get(runtime, "sequence_boundary_transl_velocity_reference_zero_seam", False)
    )
    sequence_boundary_transl_velocity_reference_window = _runtime_optional_positive_int(
        runtime,
        "sequence_boundary_transl_velocity_reference_window",
    )
    sequence_boundary_transl_delta_reference_window = _runtime_optional_positive_int(
        runtime,
        "sequence_boundary_transl_delta_reference_window",
    )
    sequence_boundary_transl_seam_window = _runtime_optional_positive_int(
        runtime,
        "sequence_boundary_transl_seam_window",
    )
    sequence_boundary_data_window = _runtime_optional_positive_int(
        runtime,
        "sequence_boundary_data_window",
    )
    sequence_boundary_data_scale = _runtime_get(runtime, "sequence_boundary_data_scale", None)
    if sequence_boundary_data_scale is not None:
        try:
            sequence_boundary_data_scale = float(sequence_boundary_data_scale)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Runtime sequence_boundary_data_scale must be a non-negative float when provided. Got {sequence_boundary_data_scale!r}."
            ) from exc
        if sequence_boundary_data_scale < 0.0:
            raise ValueError(
                f"Runtime sequence_boundary_data_scale must be a non-negative float when provided. Got {sequence_boundary_data_scale}."
            )
    compile_evaluator = bool(_runtime_get(runtime, "compile_evaluator", False))
    compile_mode = str(_runtime_get(runtime, "compile_mode", "default"))
    compile_fullgraph = bool(_runtime_get(runtime, "compile_fullgraph", False))

    frame_evaluator = build_stageii_evaluator(
        wrapper=wrapper,
        layout=layout,
        hand_pca=hand_pca,
        pose_prior=pose_prior,
        optimize_fingers=optimize_fingers,
        optimize_face=optimize_face,
        compile_module=compile_evaluator,
        compile_mode=compile_mode,
        compile_fullgraph=compile_fullgraph,
    )
    sequence_evaluator = None
    if sequence_chunk_size > 1:
        if build_stageii_sequence_evaluator is None:
            raise RuntimeError("sequence_chunk_size requested but build_stageii_sequence_evaluator is unavailable")
        sequence_evaluator = build_stageii_sequence_evaluator(
            wrapper=wrapper,
            layout=layout,
            hand_pca=hand_pca,
            pose_prior=pose_prior,
            optimize_fingers=optimize_fingers,
            optimize_face=optimize_face,
            compile_module=compile_evaluator,
            compile_mode=compile_mode,
            compile_fullgraph=compile_fullgraph,
        )

    if sequence_chunk_size > 1:
        if fit_stageii_sequence_torch is None:
            raise RuntimeError("sequence_chunk_size requested but fit_stageii_sequence_torch is unavailable")

        sequence_options = _runtime_sequence_fit_options(cfg, runtime)
        sequence_seed_options = _runtime_sequence_seed_options(sequence_options, runtime)
        sequence_seed_cache = None
        previous_chunk_latent_tail = None
        previous_chunk_fullpose_tail = None
        previous_chunk_transl_tail = None
        previous_chunk_vertices_tail = None
        previous_chunk_expression_tail = None
        sequence_chunk_keep_starts = []
        sequence_chunk_stitch_diagnostics = []
        if sequence_seed_options is not None:
            try:
                sequence_markers_obs, sequence_visible = _build_chunk_observations(
                    chunk_frames=selected_frames,
                    observed_markers_dict=observed_markers_dict,
                    latent_labels=latent_labels,
                    label_to_latent_id=label_to_latent_id,
                    device=device,
                )
                sequence_seed_cache = _build_sequence_seed_cache(
                    selected_frames=selected_frames,
                    sequence_markers_obs=sequence_markers_obs,
                    sequence_visible=sequence_visible,
                    current_latent_pose=current_latent_pose,
                    current_transl=current_transl,
                    current_expression=current_expression,
                    prev_latent_pose=prev_latent_pose,
                    body_model=body_model,
                    wrapper=wrapper,
                    betas_tensor=betas_tensor,
                    marker_attachment=marker_attachment,
                    pose_prior=pose_prior,
                    layout=layout,
                    hand_pca=hand_pca,
                    optimize_fingers=optimize_fingers,
                    optimize_face=optimize_face,
                    cfg=cfg,
                    markers_latent_tensor=markers_latent_tensor,
                    evaluator=frame_evaluator,
                    seed_options=sequence_seed_options,
                )
            except ValueError as exc:
                logger.error(str(exc))
                sequence_seed_cache = None

        for chunk_idx, (row_start, row_end) in enumerate(
            _sequence_chunk_ranges(len(selected_frames), sequence_chunk_size, sequence_chunk_overlap)
        ):
            chunk_frames = selected_frames[row_start:row_end]
            try:
                chunk_markers_obs, chunk_visible = _build_chunk_observations(
                    chunk_frames=chunk_frames,
                    observed_markers_dict=observed_markers_dict,
                    latent_labels=latent_labels,
                    label_to_latent_id=label_to_latent_id,
                    device=device,
                )
            except ValueError as exc:
                logger.error(str(exc))
                continue

            visible_counts = chunk_visible.sum(dim=1).to(dtype=torch.float32)
            avg_visible_count = max(int(round(float(visible_counts.mean().item()))), 1)
            avg_missing_markers = float(len(markers_latent) - visible_counts.mean().item())
            anneal_factor = 1.0
            if avg_missing_markers > 0:
                anneal_factor += (avg_missing_markers / len(markers_latent)) * cfg.opt_settings.weights.stageii_wt_annealing

            sequence_weights = _runtime_sequence_fit_weights(
                cfg,
                runtime,
                avg_visible_count=avg_visible_count,
                marker_count=len(markers_latent),
                anneal_factor=anneal_factor,
            )
            chunk_overlap_count = 0 if chunk_idx == 0 else min(sequence_chunk_overlap, row_end - row_start)
            chunk_length = row_end - row_start

            if sequence_seed_cache is None:
                chunk_latent_init = []
                chunk_transl_init = []
                chunk_expression_init = [] if optimize_face else None
                for local_idx in range(len(chunk_frames)):
                    chunk_latent_init.append(current_latent_pose[0])
                    chunk_transl_init.append(
                        _initial_translation(
                            chunk_markers_obs[local_idx],
                            markers_latent_tensor,
                            visible_mask=chunk_visible[local_idx],
                        )[0]
                    )
                    if optimize_face:
                        expression_init = current_expression
                        if expression_init is None:
                            expression_init = torch.zeros(1, cfg.surface_model.num_expressions, dtype=torch.float32, device=device)
                        chunk_expression_init.append(expression_init[0])
                chunk_latent_init = torch.stack(chunk_latent_init, dim=0)
                chunk_transl_init = torch.stack(chunk_transl_init, dim=0)
                chunk_expression_init = torch.stack(chunk_expression_init, dim=0) if optimize_face else None
            else:
                cached_latent_init, cached_transl_init, cached_expression_init = sequence_seed_cache
                chunk_latent_init = cached_latent_init[row_start:row_end].detach().clone()
                chunk_transl_init = cached_transl_init[row_start:row_end].detach().clone()
                chunk_expression_init = (
                    cached_expression_init[row_start:row_end].detach().clone()
                    if optimize_face and cached_expression_init is not None
                    else None
                )

            velocity_reference = None
            velocity_reference_index = None
            if (
                sequence_boundary_velocity_reference
                and prev_latent_pose is not None
                and chunk_overlap_count < chunk_length
            ):
                if sequence_boundary_velocity_reference_full_length and previous_chunk_latent_tail is not None:
                    velocity_reference = _build_chunk_full_velocity_reference(
                        chunk_latent_init,
                        previous_chunk_latent_tail,
                        chunk_overlap_count,
                    )
                else:
                    velocity_reference_index = chunk_overlap_count
                    velocity_reference = prev_latent_pose
                    keep_seam_window = min(chunk_overlap_count, chunk_length - chunk_overlap_count)
                    if previous_chunk_latent_tail is not None and keep_seam_window > 1:
                        velocity_reference = _build_chunk_velocity_reference(
                            chunk_latent_init,
                            previous_chunk_latent_tail,
                            chunk_overlap_count,
                            keep_seam_window=keep_seam_window,
                        )

            transl_velocity_reference = None
            transl_velocity_reference_index = None
            transl_boundary_reference = None
            transl_boundary_reference_index = None
            if (
                sequence_boundary_transl_velocity_reference
                and prev_transl is not None
                and chunk_overlap_count < chunk_length
            ):
                if sequence_boundary_transl_velocity_reference_full_length and previous_chunk_transl_tail is not None:
                    transl_velocity_reference = _build_chunk_full_transl_velocity_reference(
                        chunk_transl_init,
                        previous_chunk_transl_tail,
                        chunk_overlap_count,
                    )
                else:
                    transl_velocity_reference_index = chunk_overlap_count
                    transl_velocity_reference = prev_transl
                    keep_seam_window = sequence_boundary_transl_velocity_reference_window
                    if keep_seam_window is None:
                        keep_seam_window = min(chunk_overlap_count, chunk_length - chunk_overlap_count)
                    if previous_chunk_transl_tail is not None and (
                        sequence_boundary_transl_velocity_reference_window is not None or keep_seam_window > 1
                    ):
                        if sequence_boundary_transl_velocity_reference_zero_seam:
                            transl_velocity_reference = _build_chunk_zero_seam_transl_velocity_reference(
                                chunk_transl_init,
                                previous_chunk_transl_tail,
                                chunk_overlap_count,
                                keep_seam_window=keep_seam_window,
                            )
                        else:
                            transl_velocity_reference = _build_chunk_transl_velocity_reference(
                                chunk_transl_init,
                                previous_chunk_transl_tail,
                                chunk_overlap_count,
                                keep_seam_window=keep_seam_window,
                            )
            if (
                float(getattr(sequence_weights, "boundary_transl_seam", 0.0)) != 0.0
                and prev_transl is not None
                and chunk_overlap_count < chunk_length
            ):
                transl_boundary_reference_index = chunk_overlap_count
                if previous_chunk_transl_tail is not None and sequence_boundary_transl_seam_window is not None:
                    transl_boundary_reference = _build_chunk_transl_boundary_reference(
                        chunk_transl_init,
                        previous_chunk_transl_tail,
                        chunk_overlap_count,
                        keep_seam_window=sequence_boundary_transl_seam_window,
                    )
                else:
                    transl_boundary_reference = (
                        previous_chunk_transl_tail[-1:].detach().clone()
                        if previous_chunk_transl_tail is not None
                        else prev_transl
                    )

            chunk_latent_reference = chunk_latent_init
            chunk_transl_reference = chunk_transl_init
            chunk_expression_reference = chunk_expression_init
            chunk_marker_data_weights = None
            if (
                sequence_boundary_data_scale is not None
                and sequence_boundary_data_scale != 1.0
                and chunk_overlap_count > 0
                and chunk_overlap_count < chunk_length
            ):
                keep_seam_window = sequence_boundary_data_window or 1
                window_end = min(chunk_overlap_count + keep_seam_window, chunk_length)
                if window_end > chunk_overlap_count:
                    chunk_marker_data_weights = torch.ones(
                        (chunk_length, chunk_markers_obs.shape[1]),
                        dtype=torch.float32,
                        device=device,
                    )
                    chunk_marker_data_weights[chunk_overlap_count:window_end] = sequence_boundary_data_scale
            if chunk_overlap_count > 0:
                if float(getattr(sequence_weights, "delta_pose", 0.0)) != 0.0 and previous_chunk_latent_tail is not None:
                    chunk_latent_reference = _splice_chunk_overlap_reference(
                        chunk_latent_init,
                        previous_chunk_latent_tail,
                        chunk_overlap_count,
                    )
                if float(getattr(sequence_weights, "delta_trans", 0.0)) != 0.0 and previous_chunk_transl_tail is not None:
                    keep_seam_window = sequence_boundary_transl_delta_reference_window
                    if keep_seam_window is None:
                        keep_seam_window = chunk_overlap_count
                    chunk_transl_reference = _splice_chunk_overlap_reference(
                        chunk_transl_init,
                        previous_chunk_transl_tail,
                        chunk_overlap_count,
                        include_keep_seam=True,
                        keep_seam_window=keep_seam_window,
                    )
                if (
                    optimize_face
                    and float(getattr(sequence_weights, "delta_expr", 0.0)) != 0.0
                    and chunk_expression_init is not None
                    and previous_chunk_expression_tail is not None
                ):
                    chunk_expression_reference = _splice_chunk_overlap_reference(
                        chunk_expression_init,
                        previous_chunk_expression_tail,
                        chunk_overlap_count,
                    )

            result = fit_stageii_sequence_torch(
                body_model=body_model,
                wrapper=wrapper,
                betas=betas_tensor,
                marker_attachment=marker_attachment,
                marker_observations=chunk_markers_obs,
                pose_prior=pose_prior,
                layout=layout,
                latent_pose_init=chunk_latent_init,
                transl_init=chunk_transl_init,
                expression_init=chunk_expression_init,
                latent_pose_reference=chunk_latent_reference,
                transl_reference=chunk_transl_reference,
                expression_reference=chunk_expression_reference,
                hand_pca=hand_pca,
                optimize_fingers=optimize_fingers,
                optimize_face=optimize_face,
                optimize_toes=bool(cfg.moshpp.optimize_toes),
                velocity_reference=velocity_reference,
                velocity_reference_index=velocity_reference_index,
                transl_velocity_reference=transl_velocity_reference,
                transl_velocity_reference_index=transl_velocity_reference_index,
                transl_boundary_reference=transl_boundary_reference,
                transl_boundary_reference_index=transl_boundary_reference_index,
                visible_mask=chunk_visible,
                marker_data_weights=chunk_marker_data_weights,
                weights=sequence_weights,
                options=sequence_options,
                evaluator=sequence_evaluator,
            )

            current_latent_pose = torch.as_tensor(result.latent_pose[-1:]).detach()
            current_transl = torch.as_tensor(result.transl[-1:]).detach()
            current_expression = torch.as_tensor(result.expression[-1:]).detach() if result.expression is not None else None
            prev_latent_pose = torch.as_tensor(result.latent_pose[-1:]).detach()
            prev_transl = torch.as_tensor(result.transl[-1:]).detach()

            keep_start = 0
            overlap_count = min(sequence_chunk_overlap, row_end - row_start)
            stitch_diagnostics = {
                "chunk_index": int(chunk_idx),
                "row_start": int(row_start),
                "row_end": int(row_end),
                "overlap_count": int(overlap_count),
                "default_keep_start": 0,
                "selected_keep_start": 0,
                "trim_count": 0,
                "candidate_metrics": [],
            }
            if chunk_idx > 0:
                keep_start, stitch_diagnostics = _select_chunk_keep_start(
                    stitch_mode=sequence_chunk_stitch_mode,
                    overlap_count=overlap_count,
                    previous_fullpose_tail=previous_chunk_fullpose_tail,
                    previous_transl_tail=previous_chunk_transl_tail,
                    previous_vertices_tail=previous_chunk_vertices_tail,
                    current_fullpose=result.fullpose,
                    current_transl=result.transl,
                    current_vertices=result.vertices,
                    return_diagnostics=True,
                )
                stitch_diagnostics.update(
                    {
                        "chunk_index": int(chunk_idx),
                        "row_start": int(row_start),
                        "row_end": int(row_end),
                        "overlap_count": int(overlap_count),
                        "trim_count": int(overlap_count - keep_start),
                    }
                )
                _trim_perframe_tail(perframe_data, overlap_count - keep_start)
            sequence_chunk_keep_starts.append(int(keep_start))
            sequence_chunk_stitch_diagnostics.append(stitch_diagnostics)
            for local_idx in range(keep_start, row_end - row_start):
                visible_labels = [latent_labels[idx] for idx in torch.nonzero(chunk_visible[local_idx], as_tuple=False).flatten().tolist()]
                for key, value in result.loss_terms.items():
                    if torch.is_tensor(value):
                        perframe_data["stageii_errs"].setdefault(key, []).append(float(value[local_idx].item()))
                    else:
                        perframe_data["stageii_errs"].setdefault(key, []).append(float(value))
                perframe_data["markers_sim"].append(result.predicted_markers[local_idx].detach().cpu().numpy().copy())
                perframe_data["markers_obs"].append(chunk_markers_obs[local_idx].detach().cpu().numpy().copy())
                perframe_data["labels_obs"].append(visible_labels)
                perframe_data["fullpose"].append(result.fullpose[local_idx].detach().cpu().numpy().copy())
                perframe_data["trans"].append(result.transl[local_idx].detach().cpu().numpy().copy())
            if sequence_chunk_overlap > 0:
                tail_count = min(sequence_chunk_overlap, result.fullpose.shape[0])
                previous_chunk_latent_tail = torch.as_tensor(result.latent_pose[-tail_count:]).detach()
                previous_chunk_fullpose_tail = torch.as_tensor(result.fullpose[-tail_count:]).detach()
                previous_chunk_transl_tail = torch.as_tensor(result.transl[-tail_count:]).detach()
                previous_chunk_vertices_tail = torch.as_tensor(result.vertices[-tail_count:]).detach()
                previous_chunk_expression_tail = (
                    torch.as_tensor(result.expression[-tail_count:]).detach()
                    if result.expression is not None
                    else None
                )
    else:
        for fidx, t in enumerate(selected_frames):
            frame_obs = observed_markers_dict[t]
            if len(frame_obs) == 0:
                logger.error(f"no available observed markers for frame {t}. skipping the frame.")
                continue

            visible_ids = [lid for lid, label in enumerate(latent_labels) if label in frame_obs]
            visible_labels = [latent_labels[lid] for lid in visible_ids]
            markers_obs = torch.stack(
                [torch.as_tensor(frame_obs[label], dtype=torch.float32, device=device) for label in visible_labels]
            )
            attachment_subset = _subset_attachment(marker_attachment, visible_ids)

            num_missing_markers = float(len(markers_latent) - len(visible_ids))
            anneal_factor = 1.0
            if num_missing_markers > 0:
                anneal_factor += (num_missing_markers / len(markers_latent)) * cfg.opt_settings.weights.stageii_wt_annealing

            weights = TorchFrameFitWeights(
                data=cfg.opt_settings.weights.stageii_wt_data * (num_train_markers / max(markers_obs.shape[0], 1)),
                pose_body=cfg.opt_settings.weights.stageii_wt_poseB * anneal_factor,
                pose_hand=cfg.opt_settings.weights.stageii_wt_poseH * anneal_factor,
                pose_face=cfg.opt_settings.weights.stageii_wt_poseF * anneal_factor,
                expr=cfg.opt_settings.weights.stageii_wt_expr,
                velocity=cfg.opt_settings.weights.stageii_wt_velo,
            )
            options = _runtime_stage_fit_options(cfg, runtime)

            if fidx == 0:
                current_transl = _initial_translation(markers_obs, markers_latent_tensor[visible_ids])
                warmup_scales = (10.0, 5.0, 1.0)
                rigid_init = True
            else:
                warmup_scales = (1.0,)
                rigid_init = False

            result = fit_stageii_frame_torch(
                body_model=body_model,
                betas=betas_tensor,
                marker_attachment=attachment_subset,
                marker_observations=markers_obs,
                pose_prior=pose_prior,
                layout=layout,
                latent_pose_init=current_latent_pose,
                transl_init=current_transl,
                expression_init=current_expression,
                hand_pca=hand_pca,
                optimize_fingers=optimize_fingers,
                optimize_face=optimize_face,
                optimize_toes=bool(cfg.moshpp.optimize_toes),
                velocity_reference=prev_latent_pose,
                weights=weights,
                options=options,
                rigid_init=rigid_init,
                warmup_pose_scales=warmup_scales,
                evaluator=frame_evaluator,
            )

            current_latent_pose = result.latent_pose.detach()
            current_transl = result.transl.detach()
            current_expression = result.expression.detach() if result.expression is not None else None
            prev_latent_pose = result.latent_pose.detach()

            for key, value in result.loss_terms.items():
                perframe_data["stageii_errs"].setdefault(key, []).append(value)

            perframe_data["markers_sim"].append(result.predicted_markers.cpu().numpy().copy())
            perframe_data["markers_obs"].append(markers_obs.cpu().numpy().copy())
            perframe_data["labels_obs"].append(visible_labels)
            perframe_data["fullpose"].append(result.fullpose.cpu().numpy()[0].copy())
            perframe_data["trans"].append(result.transl.cpu().numpy()[0].copy())

    stageii_debug_details = {
        "stageii_errs": {key: _loss_history_to_numpy(values) for key, values in perframe_data.pop("stageii_errs").items()},
        "markers_sim": perframe_data.pop("markers_sim"),
        "markers_obs": perframe_data.pop("markers_obs"),
        "labels_obs": perframe_data.pop("labels_obs"),
        "markers_orig": mocap.markers[list(selected_frames)],
        "labels_orig": mocap.labels,
        "mocap_fname": mocap_fname,
        "mocap_frame_rate": mocap.frame_rate,
        "mocap_time_length": mocap.time_length(),
    }
    if sequence_chunk_size > 1:
        stageii_debug_details["sequence_chunk_stitch_mode"] = sequence_chunk_stitch_mode
        stageii_debug_details["sequence_chunk_keep_starts"] = sequence_chunk_keep_starts
        stageii_debug_details["sequence_chunk_stitch_diagnostics"] = sequence_chunk_stitch_diagnostics
    stageii_data = {key: np.array(values) for key, values in perframe_data.items()}
    stageii_data["stageii_debug_details"] = stageii_debug_details
    return stageii_data
