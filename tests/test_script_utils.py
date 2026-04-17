import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.script_utils import (
    discover_stageii_pickles,
    codec_for_video_path,
    default_stageii_artifact_paths,
    default_stageii_output_paths,
    list_stageii_pickles,
    planned_stageii_output_path_from_overrides,
    resolve_support_base_dir,
    resolve_stageii_model_path,
)


def test_resolve_support_base_dir_uses_default_under_work_base_dir():
    assert resolve_support_base_dir("/tmp/soma-work") == os.path.join(
        "/tmp/soma-work", "support_files"
    )


def test_resolve_support_base_dir_prefers_explicit_override():
    assert resolve_support_base_dir("/tmp/soma-work", "/data/support") == "/data/support"


def test_default_stageii_output_paths_replace_stageii_suffix():
    obj_out, pc2_out = default_stageii_output_paths("/tmp/demo_stageii.pkl")

    assert obj_out == "/tmp/demo_stageii.obj"
    assert pc2_out == "/tmp/demo_stageii.pc2"


def test_default_stageii_output_paths_fall_back_to_plain_stem():
    obj_out, pc2_out = default_stageii_output_paths("/tmp/demo.pkl")

    assert obj_out == "/tmp/demo.obj"
    assert pc2_out == "/tmp/demo.pc2"


def test_default_stageii_artifact_paths_replace_stageii_suffix():
    obj_out, pc2_out, video_out = default_stageii_artifact_paths("/tmp/demo_stageii.pkl")

    assert obj_out == "/tmp/demo_stageii.obj"
    assert pc2_out == "/tmp/demo_stageii.pc2"
    assert video_out == "/tmp/demo_stageii.mp4"


def test_default_stageii_artifact_paths_fall_back_to_plain_stem():
    obj_out, pc2_out, video_out = default_stageii_artifact_paths("/tmp/demo.pkl")

    assert obj_out == "/tmp/demo.obj"
    assert pc2_out == "/tmp/demo.pc2"
    assert video_out == "/tmp/demo_stageii.mp4"


def test_planned_stageii_output_path_from_overrides_infers_dataset_and_session_from_mocap_path(tmp_path):
    overrides = {
        "mocap.fname": str(tmp_path / "input" / "wolf001" / "capture.mcp"),
        "dirs.work_base_dir": str(tmp_path / "work"),
        "mocap.basename": "manual_candidate",
    }

    assert planned_stageii_output_path_from_overrides(overrides) == (
        tmp_path / "work" / "input" / "wolf001" / "manual_candidate_stageii.pkl"
    )


def test_planned_stageii_output_path_from_overrides_prefers_explicit_stageii_path(tmp_path):
    explicit_stageii_path = tmp_path / "explicit" / "candidate_stageii.pkl"

    assert planned_stageii_output_path_from_overrides(
        {
            "dirs.stageii_fname": explicit_stageii_path,
        }
    ) == explicit_stageii_path


def test_planned_stageii_output_path_from_overrides_errors_for_multi_subject_without_subject_name(
    tmp_path,
):
    overrides = {
        "mocap.fname": str(tmp_path / "input" / "wolf001" / "capture.mcp"),
        "dirs.work_base_dir": str(tmp_path / "work"),
        "mocap.multi_subject": "true",
    }

    with pytest.raises(
        ValueError,
        match="cannot infer multi-subject stageii output path without mocap.subject_name",
    ):
        planned_stageii_output_path_from_overrides(overrides)


def test_list_stageii_pickles_returns_sorted_matches(tmp_path):
    (tmp_path / "b_stageii.pkl").write_text("")
    (tmp_path / "ignore.pkl").write_text("")
    (tmp_path / "a_stageii.pkl").write_text("")

    assert list_stageii_pickles(str(tmp_path)) == [
        str(tmp_path / "a_stageii.pkl"),
        str(tmp_path / "b_stageii.pkl"),
    ]


def test_discover_stageii_pickles_recurses_and_applies_fname_filter(tmp_path):
    dataset_root = tmp_path / "demo_ds"
    keep_path = dataset_root / "subject01" / "swing_stageii.pkl"
    drop_path = dataset_root / "subject01" / "serve_stageii.pkl"
    keep_path.parent.mkdir(parents=True, exist_ok=True)
    keep_path.write_text("")
    drop_path.write_text("")

    assert discover_stageii_pickles(str(tmp_path), "demo_ds", fname_filter=["swing"]) == [
        str(keep_path)
    ]


def test_resolve_stageii_model_path_prefers_support_base_dir_assets(tmp_path):
    support_dir = tmp_path / "support"
    relocated_model_path = support_dir / "smplx" / "female" / "model.npz"
    relocated_model_path.parent.mkdir(parents=True, exist_ok=True)
    relocated_model_path.write_bytes(b"npz")
    stageii_path = tmp_path / "demo_stageii.pkl"
    stageii_path.write_bytes(
        __import__("pickle").dumps(
            {
                "stageii_debug_details": {
                    "cfg": {
                        "surface_model": {
                            "type": "smplx",
                            "gender": "female",
                            "fname": "/old-machine/support_files/smplx/female/model.pkl",
                        }
                    }
                }
            }
        )
    )

    assert resolve_stageii_model_path(
        str(stageii_path), support_base_dir=str(support_dir)
    ) == str(relocated_model_path)


def test_codec_for_video_path_uses_mp4v_for_mp4():
    assert codec_for_video_path("demo.mp4") == "mp4v"


def test_codec_for_video_path_uses_xvid_for_avi():
    assert codec_for_video_path("demo.avi") == "XVID"
