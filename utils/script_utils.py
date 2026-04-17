import os.path as osp
import pickle
from glob import glob
from pathlib import Path


def resolve_support_base_dir(work_base_dir, support_base_dir=None):
    if support_base_dir:
        return support_base_dir
    return osp.join(work_base_dir, "support_files")


def default_stageii_output_paths(stageii_pkl_path):
    if stageii_pkl_path.endswith("_stageii.pkl"):
        stem = stageii_pkl_path[:-4]
    else:
        stem = osp.splitext(stageii_pkl_path)[0]
    return stem + ".obj", stem + ".pc2"


def default_stageii_artifact_paths(stageii_pkl_path, *, video_suffix="_stageii.mp4"):
    obj_out, pc2_out = default_stageii_output_paths(stageii_pkl_path)
    if stageii_pkl_path.endswith("_stageii.pkl"):
        video_out = stageii_pkl_path[: -len("_stageii.pkl")] + video_suffix
    else:
        video_out = osp.splitext(stageii_pkl_path)[0] + video_suffix
    return obj_out, pc2_out, video_out


def list_stageii_pickles(input_dir, suffix="_stageii.pkl"):
    pattern = osp.join(input_dir, "*" + suffix)
    return sorted(glob(pattern))


def codec_for_video_path(video_path):
    if video_path.lower().endswith(".mp4"):
        return "mp4v"
    return "XVID"


def load_pickle_compat(path):
    with Path(path).open("rb") as handle:
        try:
            return pickle.load(handle)
        except UnicodeDecodeError:
            handle.seek(0)
            return pickle.load(handle, encoding="latin1")


def _matches_fname_filter(path, fname_filter):
    if not fname_filter:
        return True
    path = str(path)
    return any(token in path for token in fname_filter)


def discover_stageii_pickles(work_base_dir, dataset, *, fname_filter=None, suffix="_stageii.pkl"):
    dataset_root = Path(work_base_dir) / dataset
    if not dataset_root.exists():
        return []
    return [
        str(path)
        for path in sorted(dataset_root.rglob("*" + suffix))
        if _matches_fname_filter(path, fname_filter)
    ]


def _stageii_surface_model_cfg(stageii_pkl):
    stageii_data = load_pickle_compat(stageii_pkl)
    cfg = stageii_data.get("stageii_debug_details", {}).get("cfg")
    if cfg is None:
        raise KeyError(f"{stageii_pkl} does not include stageii_debug_details.cfg")
    try:
        return cfg["surface_model"]
    except Exception as exc:
        raise KeyError(f"{stageii_pkl} does not include cfg.surface_model") from exc


def _support_model_path_candidates(surface_model_cfg, support_base_dir):
    if not support_base_dir:
        return []
    model_type = surface_model_cfg.get("type")
    gender = surface_model_cfg.get("gender")
    if not model_type or not gender:
        return []

    base_dir = Path(support_base_dir) / str(model_type) / str(gender)
    original_model_path = str(surface_model_cfg.get("fname", ""))
    original_suffix = Path(original_model_path).suffix.lower()
    suffixes = []
    if original_suffix in {".npz", ".pkl"}:
        suffixes.append(original_suffix)
    for suffix in (".npz", ".pkl"):
        if suffix not in suffixes:
            suffixes.append(suffix)
    return [str(base_dir / ("model" + suffix)) for suffix in suffixes]


def resolve_stageii_model_path(stageii_pkl, *, support_base_dir=None):
    surface_model_cfg = _stageii_surface_model_cfg(stageii_pkl)
    model_path = surface_model_cfg.get("fname")
    if model_path is None:
        raise KeyError(f"{stageii_pkl} does not include cfg.surface_model.fname")
    model_path = str(model_path)
    candidates = _support_model_path_candidates(surface_model_cfg, support_base_dir)
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    if Path(model_path).exists():
        return model_path
    if candidates:
        return candidates[0]
    if not model_path:
        raise ValueError(f"{stageii_pkl} resolved an empty cfg.surface_model.fname")
    return model_path
