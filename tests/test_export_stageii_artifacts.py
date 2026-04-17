import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import export_stageii_artifacts
from utils.script_utils import default_stageii_artifact_paths

SUPPORT_ROOT = ROOT / "support_files"


def _write_stageii_pickle(path, *, model_path, surface_model_type="smplx", gender="male"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        pickle.dumps(
            {
                "stageii_debug_details": {
                    "cfg": {
                        "surface_model": {
                            "type": surface_model_type,
                            "fname": str(model_path),
                            "gender": gender,
                        }
                    }
                }
            }
        )
    )


def test_export_stageii_artifacts_uses_default_output_paths_and_shared_render_inputs(monkeypatch, tmp_path):
    input_path = tmp_path / "tiny_stageii.pkl"
    input_path.write_bytes(
        pickle.dumps(
            {
                "fullpose": np.zeros((2, 165), dtype=np.float32),
                "betas": np.zeros(400, dtype=np.float32),
                "trans": np.zeros((2, 3), dtype=np.float32),
            }
        )
    )

    preloaded_model = type("PreloadedModel", (), {"faces": np.array([[0, 1, 2]], dtype=np.int32)})()
    predecoded_vertices = np.zeros((2, 3, 3), dtype=np.float32)
    captured = {
        "load_model_calls": [],
        "load_vertices_calls": [],
        "mesh_calls": [],
        "video_calls": [],
    }

    monkeypatch.setattr(
        export_stageii_artifacts.render_video,
        "load_render_model",
        lambda model_path: captured["load_model_calls"].append(model_path) or preloaded_model,
    )
    monkeypatch.setattr(
        export_stageii_artifacts.render_video,
        "load_vertices",
        lambda pkl_path, model: captured["load_vertices_calls"].append((pkl_path, model)) or predecoded_vertices,
    )

    def fake_export_stageii_meshes(*, input_pkl, model_path=None, model=None, vertices=None, obj_out=None, pc2_out=None):
        captured["mesh_calls"].append((input_pkl, model, vertices, obj_out, pc2_out))
        Path(obj_out).write_text("obj")
        Path(pc2_out).write_bytes(b"pc2")
        return str(obj_out), str(pc2_out)

    monkeypatch.setattr(export_stageii_artifacts.save_smplx_verts, "export_stageii_meshes", fake_export_stageii_meshes)
    monkeypatch.setattr(
        export_stageii_artifacts.render_video,
        "render_vertices_to_video",
        lambda *, vertices, faces, output_path, **kwargs: (
            captured["video_calls"].append((vertices, faces, output_path, kwargs)),
            Path(output_path).write_bytes(b"mp4"),
            str(output_path),
        )[-1],
    )

    result = export_stageii_artifacts.export_stageii_artifacts(
        input_pkl=input_path,
        model_path=SUPPORT_ROOT / "smplx" / "male" / "model.npz",
        fps=5,
        width=128,
        height=128,
        arch="cpu",
        camera_z=2.25,
    )

    expected_obj, expected_pc2, expected_video = default_stageii_artifact_paths(str(input_path))
    assert result == {
        "obj_path": expected_obj,
        "pc2_path": expected_pc2,
        "video_path": expected_video,
    }
    assert captured["load_model_calls"] == [SUPPORT_ROOT / "smplx" / "male" / "model.npz"]
    assert captured["load_vertices_calls"] == [(input_path, preloaded_model)]
    assert captured["mesh_calls"] == [
        (input_path, preloaded_model, predecoded_vertices, expected_obj, expected_pc2)
    ]
    assert len(captured["video_calls"]) == 1
    assert captured["video_calls"][0][0] is predecoded_vertices
    assert np.array_equal(captured["video_calls"][0][1], preloaded_model.faces)
    assert captured["video_calls"][0][2] == expected_video
    assert captured["video_calls"][0][3]["camera_z"] == 2.25
    assert Path(expected_obj).exists()
    assert Path(expected_pc2).exists()
    assert Path(expected_video).exists()


def test_export_stageii_artifacts_reuses_preloaded_model_and_vertices_without_loading_again(monkeypatch, tmp_path):
    input_path = tmp_path / "tiny_stageii.pkl"
    input_path.write_bytes(b"placeholder")

    preloaded_model = type("PreloadedModel", (), {"faces": np.array([[0, 1, 2]], dtype=np.int32)})()
    predecoded_vertices = np.ones((2, 3, 3), dtype=np.float32)
    captured = {}

    monkeypatch.setattr(
        export_stageii_artifacts.render_video,
        "load_render_model",
        lambda model_path: (_ for _ in ()).throw(AssertionError("should not reload model")),
    )
    monkeypatch.setattr(
        export_stageii_artifacts.render_video,
        "load_vertices",
        lambda pkl_path, model: (_ for _ in ()).throw(AssertionError("should not reload vertices")),
    )
    monkeypatch.setattr(
        export_stageii_artifacts.save_smplx_verts,
        "export_stageii_meshes",
        lambda *, input_pkl, model_path=None, model=None, vertices=None, obj_out=None, pc2_out=None: (
            captured.__setitem__("mesh", (model, vertices)),
            Path(obj_out).write_text("obj"),
            Path(pc2_out).write_bytes(b"pc2"),
            (str(obj_out), str(pc2_out)),
        )[-1],
    )
    monkeypatch.setattr(
        export_stageii_artifacts.render_video,
        "render_vertices_to_video",
        lambda *, vertices, faces, output_path, **kwargs: (
            captured.__setitem__("video", (vertices, faces)),
            Path(output_path).write_bytes(b"mp4"),
            str(output_path),
        )[-1],
    )

    result = export_stageii_artifacts.export_stageii_artifacts(
        input_pkl=input_path,
        model=preloaded_model,
        vertices=predecoded_vertices,
        obj_out=tmp_path / "bundle.obj",
        pc2_out=tmp_path / "bundle.pc2",
        video_out=tmp_path / "bundle.mp4",
        arch="cpu",
    )

    assert result == {
        "obj_path": str(tmp_path / "bundle.obj"),
        "pc2_path": str(tmp_path / "bundle.pc2"),
        "video_path": str(tmp_path / "bundle.mp4"),
    }
    assert captured["mesh"] == (preloaded_model, predecoded_vertices)
    assert captured["video"] == (predecoded_vertices, preloaded_model.faces)


def test_export_stageii_artifacts_batch_resolves_model_path_per_stageii(monkeypatch, tmp_path):
    support_dir = tmp_path / "support"
    male_model = support_dir / "smplx" / "male" / "model.npz"
    female_model = support_dir / "smplx" / "female" / "model.pkl"
    male_model.parent.mkdir(parents=True, exist_ok=True)
    female_model.parent.mkdir(parents=True, exist_ok=True)
    male_model.write_bytes(b"npz")
    female_model.write_bytes(b"pkl")

    swing_stageii = tmp_path / "exports" / "subject01" / "swing_stageii.pkl"
    serve_stageii = tmp_path / "exports" / "subject02" / "serve_stageii.pkl"
    _write_stageii_pickle(
        swing_stageii,
        model_path="/old-machine/support_files/smplx/male/model.pkl",
        gender="male",
    )
    _write_stageii_pickle(
        serve_stageii,
        model_path="/old-machine/support_files/smplx/female/model.npz",
        gender="female",
    )

    calls = []

    def fake_export_stageii_artifacts(**kwargs):
        calls.append(kwargs)
        return {
            "obj_path": str(Path(kwargs["input_pkl"]).with_suffix(".obj")),
            "pc2_path": str(Path(kwargs["input_pkl"]).with_suffix(".pc2")),
            "video_path": str(Path(kwargs["input_pkl"]).with_suffix(".mp4")),
        }

    monkeypatch.setattr(
        export_stageii_artifacts,
        "export_stageii_artifacts",
        fake_export_stageii_artifacts,
    )

    results = export_stageii_artifacts.export_stageii_artifacts_batch(
        input_pkls=[swing_stageii, serve_stageii],
        support_base_dir=support_dir,
        fps=15,
        width=160,
        height=120,
        arch="cpu",
    )

    assert len(results) == 2
    assert [call["input_pkl"] for call in calls] == [str(swing_stageii), str(serve_stageii)]
    assert [call["model_path"] for call in calls] == [str(male_model), str(female_model)]
    assert all(call["fps"] == 15 for call in calls)
    assert all(call["width"] == 160 for call in calls)
    assert all(call["height"] == 120 for call in calls)
    assert all(call["arch"] == "cpu" for call in calls)


def test_export_stageii_artifacts_main_supports_recursive_input_dir(monkeypatch, tmp_path):
    support_dir = tmp_path / "support"
    matching_stageii = tmp_path / "exports" / "subject01" / "swing_stageii.pkl"
    ignored_stageii = tmp_path / "exports" / "subject01" / "serve_stageii.pkl"
    _write_stageii_pickle(
        matching_stageii,
        model_path="/old-machine/support_files/smplx/male/model.pkl",
        gender="male",
    )
    _write_stageii_pickle(
        ignored_stageii,
        model_path="/old-machine/support_files/smplx/male/model.pkl",
        gender="male",
    )

    captured = {}

    def fake_export_stageii_artifacts_batch(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(
        export_stageii_artifacts,
        "export_stageii_artifacts_batch",
        fake_export_stageii_artifacts_batch,
    )

    export_stageii_artifacts.main(
        [
            "--input-dir",
            str(tmp_path / "exports"),
            "--support-base-dir",
            str(support_dir),
            "--fname-filter",
            "swing",
            "--fps",
            "15",
        ]
    )

    assert captured["input_pkls"] == [str(matching_stageii)]
    assert captured["support_base_dir"] == str(support_dir)
    assert captured["fps"] == 15


def test_export_stageii_artifacts_main_errors_when_input_dir_matches_no_stageii_pickles(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        export_stageii_artifacts.main(
            [
                "--input-dir",
                str(tmp_path / "exports"),
                "--fname-filter",
                "swing",
            ]
        )

    assert excinfo.value.code == 2


def test_export_stageii_artifacts_main_errors_when_single_input_cannot_resolve_model_path(
    tmp_path,
):
    input_path = tmp_path / "broken_stageii.pkl"
    input_path.write_bytes(pickle.dumps({"fullpose": [], "betas": [], "trans": []}))

    with pytest.raises(SystemExit) as excinfo:
        export_stageii_artifacts.main(
            [
                "--input-pkl",
                str(input_path),
            ]
        )

    assert excinfo.value.code == 2
