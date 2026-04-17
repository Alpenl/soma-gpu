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


def _string_flag_is_true(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _default_mocap_path_part(mocap_path, index_from_end, *, label):
    parts = mocap_path.as_posix().split("/")
    try:
        return parts[index_from_end]
    except IndexError as exc:
        raise ValueError(
            f"cannot infer {label} from --mocap-fname={mocap_path}; "
            "provide a dataset/session-style path or override the relevant mocap.* cfg explicitly"
        ) from exc


def planned_stageii_output_path_from_overrides(overrides):
    explicit_stageii_path = overrides.get("dirs.stageii_fname")
    if explicit_stageii_path:
        return Path(str(explicit_stageii_path))

    mocap_path = Path(str(overrides["mocap.fname"]))
    work_base_dir = Path(str(overrides["dirs.work_base_dir"]))
    ds_name = str(
        overrides.get(
            "mocap.ds_name",
            _default_mocap_path_part(mocap_path, -3, label="mocap.ds_name"),
        )
    )
    session_name = str(
        overrides.get(
            "mocap.session_name",
            _default_mocap_path_part(mocap_path, -2, label="mocap.session_name"),
        )
    )
    basename = str(overrides.get("mocap.basename", mocap_path.stem))

    session_subject_subfolders = overrides.get("dirs.session_subject_subfolders")
    if session_subject_subfolders is not None:
        relative_dir = Path(str(session_subject_subfolders))
    elif _string_flag_is_true(overrides.get("mocap.multi_subject")):
        subject_name = overrides.get("mocap.subject_name")
        if not subject_name:
            raise ValueError(
                "cannot infer multi-subject stageii output path without mocap.subject_name; "
                "set mocap.subject_name or dirs.session_subject_subfolders explicitly"
            )
        relative_dir = Path(session_name) / str(subject_name)
    else:
        relative_dir = Path(session_name)

    return work_base_dir / ds_name / relative_dir / f"{basename}_stageii.pkl"


def batch_output_dir_for_input(input_path, *, output_dir=None, input_root=None):
    if output_dir is None:
        return None
    output_dir = Path(output_dir)
    if input_root is None:
        return str(output_dir)

    input_parent = Path(input_path).parent
    input_root = Path(input_root)
    try:
        relative_parent = input_parent.relative_to(input_root)
    except ValueError as exc:
        raise ValueError(f"{input_path} is not under input_root={input_root}") from exc
    return str(output_dir / relative_parent)


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
    return discover_stageii_pickles_in_dir(
        dataset_root, fname_filter=fname_filter, suffix=suffix
    )


def discover_stageii_pickles_in_dir(input_dir, *, fname_filter=None, suffix="_stageii.pkl"):
    input_root = Path(input_dir)
    if not input_root.exists():
        return []
    return [
        str(path)
        for path in sorted(input_root.rglob("*" + suffix))
        if _matches_fname_filter(path, fname_filter)
    ]


def format_stageii_match_error(search_root, *, fname_filter=None, suffix="_stageii.pkl"):
    message = f"No *{suffix} files matched under {search_root}"
    if fname_filter:
        message += " with fname_filter={}".format(", ".join(str(token) for token in fname_filter))
    return message


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
