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
    decode_stageii_latent_pose,
    fit_stageii_frame_torch,
    make_stageii_latent_layout,
)
try:
    from moshpp.optim.sequence_fit_torch import (
        TorchSequenceFitOptions,
        TorchSequenceFitWeights,
        fit_stageii_sequence_torch,
    )
except ModuleNotFoundError:
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
    optimizer = str(_runtime_get(runtime, "sequence_optimizer", "adam")).lower()
    if optimizer not in valid_optimizers:
        raise ValueError(f"Unsupported runtime optimizer for sequence: {optimizer}. Expected one of {sorted(valid_optimizers)}.")

    default_max_iters = int(_runtime_get(runtime, "sequence_max_iters", int(cfg.opt_settings.maxiter)))
    default_lr = float(_runtime_get(runtime, "sequence_lr", 1e-1))
    return TorchSequenceFitOptions(
        max_iters=default_max_iters,
        lr=default_lr,
        optimizer=optimizer,
        history_size=int(_runtime_get(runtime, "sequence_history_size", 100)),
        tolerance_grad=float(_runtime_get(runtime, "sequence_tolerance_grad", 1e-7)),
        tolerance_change=float(_runtime_get(runtime, "sequence_tolerance_change", 1e-9)),
        max_eval=_runtime_get(runtime, "sequence_max_eval", None),
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
        temporal_accel=float(_runtime_get(runtime, "sequence_temporal_accel", 0.0)),
        delta_pose=float(_runtime_get(runtime, "sequence_delta_pose", 0.0)),
        delta_trans=float(_runtime_get(runtime, "sequence_delta_trans", 0.0)),
        delta_expr=float(_runtime_get(runtime, "sequence_delta_expr", 0.0)),
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
    runtime = getattr(cfg, "runtime", None)
    label_to_latent_id = {label: idx for idx, label in enumerate(latent_labels)}
    sequence_chunk_size = max(int(_runtime_get(runtime, "sequence_chunk_size", 1) or 1), 1)
    sequence_chunk_overlap = max(int(_runtime_get(runtime, "sequence_chunk_overlap", 0) or 0), 0)

    if sequence_chunk_size > 1:
        if fit_stageii_sequence_torch is None:
            raise RuntimeError("sequence_chunk_size requested but fit_stageii_sequence_torch is unavailable")

        sequence_options = _runtime_sequence_fit_options(cfg, runtime)

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

            result = fit_stageii_sequence_torch(
                body_model=body_model,
                wrapper=wrapper,
                betas=betas_tensor,
                marker_attachment=marker_attachment,
                marker_observations=chunk_markers_obs,
                pose_prior=pose_prior,
                layout=layout,
                latent_pose_init=torch.stack(chunk_latent_init, dim=0),
                transl_init=torch.stack(chunk_transl_init, dim=0),
                expression_init=torch.stack(chunk_expression_init, dim=0) if optimize_face else None,
                hand_pca=hand_pca,
                optimize_fingers=optimize_fingers,
                optimize_face=optimize_face,
                optimize_toes=bool(cfg.moshpp.optimize_toes),
                visible_mask=chunk_visible,
                weights=sequence_weights,
                options=sequence_options,
            )

            current_latent_pose = torch.as_tensor(result.latent_pose[-1:]).detach()
            current_transl = torch.as_tensor(result.transl[-1:]).detach()
            current_expression = torch.as_tensor(result.expression[-1:]).detach() if result.expression is not None else None
            prev_latent_pose = torch.as_tensor(result.latent_pose[-1:]).detach()

            keep_start = 0 if chunk_idx == 0 else min(sequence_chunk_overlap, row_end - row_start)
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
            options = TorchFrameFitOptions(
                rigid_iters=cfg.opt_settings.maxiter,
                warmup_iters=cfg.opt_settings.maxiter,
                refine_iters=cfg.opt_settings.maxiter,
            )

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
        "stageii_errs": {key: np.array(values) for key, values in perframe_data.pop("stageii_errs").items()},
        "markers_sim": perframe_data.pop("markers_sim"),
        "markers_obs": perframe_data.pop("markers_obs"),
        "labels_obs": perframe_data.pop("labels_obs"),
        "markers_orig": mocap.markers[list(selected_frames)],
        "labels_orig": mocap.labels,
        "mocap_fname": mocap_fname,
        "mocap_frame_rate": mocap.frame_rate,
        "mocap_time_length": mocap.time_length(),
    }
    stageii_data = {key: np.array(values) for key, values in perframe_data.items()}
    stageii_data["stageii_debug_details"] = stageii_debug_details
    return stageii_data
