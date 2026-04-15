import os.path as osp
from glob import glob


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


def list_stageii_pickles(input_dir, suffix="_stageii.pkl"):
    pattern = osp.join(input_dir, "*" + suffix)
    return sorted(glob(pattern))


def codec_for_video_path(video_path):
    if video_path.lower().endswith(".mp4"):
        return "mp4v"
    return "XVID"
