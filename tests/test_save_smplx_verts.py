import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import save_smplx_verts
from utils.mesh_io import load_obj_mesh, readPC2
from utils.script_utils import default_stageii_output_paths

SUPPORT_ROOT = ROOT / "support_files"


def _write_stageii_pickle(path, *, model_path, surface_model_type="smplx", gender="male"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        pickle.dumps(
            {
                "fullpose": np.zeros((2, 165), dtype=np.float32),
                "betas": np.zeros(400, dtype=np.float32),
                "trans": np.zeros((2, 3), dtype=np.float32),
                "stageii_debug_details": {
                    "cfg": {
                        "surface_model": {
                            "type": surface_model_type,
                            "fname": str(model_path),
                            "gender": gender,
                        }
                    }
                },
            }
        )
    )


def test_export_stageii_meshes_supports_legacy_stageii_inputs_and_model_pkl_path(tmp_path):
    input_path = tmp_path / "legacy_stageii.pkl"
    input_path.write_bytes(
        pickle.dumps(
            {
                "ps": {"fitting_model": "smplh"},
                "pose_est_fullposes": np.zeros((2, 156), dtype=np.float32),
                "shape_est_betas": np.zeros(400, dtype=np.float32),
                "pose_est_trans": np.zeros((2, 3), dtype=np.float32),
            }
        )
    )

    obj_out = tmp_path / "legacy.obj"
    pc2_out = tmp_path / "legacy.pc2"
    result = save_smplx_verts.export_stageii_meshes(
        input_pkl=input_path,
        model_path=SUPPORT_ROOT / "smplx" / "male" / "model.pkl",
        obj_out=obj_out,
        pc2_out=pc2_out,
    )

    assert result == (str(obj_out), str(pc2_out))
    assert obj_out.exists()
    assert pc2_out.exists()

    vertices, faces = load_obj_mesh(str(obj_out))
    pc2_data = readPC2(str(pc2_out))

    assert vertices.shape[0] > 0
    assert faces.shape[1] == 3
    assert pc2_data["nSamples"] == 2
    assert pc2_data["V"].shape[2] == 3
    assert np.isfinite(pc2_data["V"]).all()


def test_export_stageii_meshes_uses_default_output_paths_when_omitted(tmp_path):
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

    result = save_smplx_verts.export_stageii_meshes(
        input_pkl=input_path,
        model_path=SUPPORT_ROOT / "smplx" / "male" / "model.npz",
    )

    expected_obj, expected_pc2 = default_stageii_output_paths(str(input_path))
    assert result == (expected_obj, expected_pc2)
    assert Path(expected_obj).exists()
    assert Path(expected_pc2).exists()


def test_export_stageii_meshes_reuses_preloaded_model_without_loading_again(monkeypatch, tmp_path):
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
    captured = {}

    monkeypatch.setattr(
        save_smplx_verts.render_video,
        "load_render_model",
        lambda model_path: (_ for _ in ()).throw(AssertionError("should not reload model")),
    )
    monkeypatch.setattr(
        save_smplx_verts,
        "load_smpl_vertices",
        lambda pkl_path, model: (
            captured.__setitem__("model", model),
            np.zeros((2, 3, 3), dtype=np.float32),
        )[1],
    )
    monkeypatch.setattr(
        save_smplx_verts,
        "save_obj_mesh",
        lambda mesh_path, verts, faces: Path(mesh_path).write_text("obj"),
    )
    monkeypatch.setattr(
        save_smplx_verts,
        "writePC2",
        lambda pc2_path, vertices: Path(pc2_path).write_bytes(b"pc2"),
    )

    obj_out = tmp_path / "preloaded.obj"
    pc2_out = tmp_path / "preloaded.pc2"
    result = save_smplx_verts.export_stageii_meshes(
        input_pkl=input_path,
        model=preloaded_model,
        obj_out=obj_out,
        pc2_out=pc2_out,
    )

    assert result == (str(obj_out), str(pc2_out))
    assert captured["model"] is preloaded_model
    assert obj_out.exists()
    assert pc2_out.exists()


def test_export_stageii_meshes_reuses_predecoded_vertices_without_loading_again(monkeypatch, tmp_path):
    input_path = tmp_path / "tiny_stageii.pkl"
    input_path.write_bytes(b"placeholder")

    preloaded_model = type("PreloadedModel", (), {"faces": np.array([[0, 1, 2]], dtype=np.int32)})()
    predecoded_vertices = np.ones((2, 3, 3), dtype=np.float32)
    captured = {}

    monkeypatch.setattr(
        save_smplx_verts,
        "load_smpl_vertices",
        lambda pkl_path, model: (_ for _ in ()).throw(AssertionError("should not decode vertices")),
    )
    monkeypatch.setattr(
        save_smplx_verts,
        "save_obj_mesh",
        lambda mesh_path, verts, faces: (
            captured.__setitem__("obj", (verts, faces)),
            Path(mesh_path).write_text("obj"),
        )[-1],
    )
    monkeypatch.setattr(
        save_smplx_verts,
        "writePC2",
        lambda pc2_path, vertices: (
            captured.__setitem__("pc2", vertices),
            Path(pc2_path).write_bytes(b"pc2"),
        )[-1],
    )

    result = save_smplx_verts.export_stageii_meshes(
        input_pkl=input_path,
        model=preloaded_model,
        vertices=predecoded_vertices,
        obj_out=tmp_path / "predecoded.obj",
        pc2_out=tmp_path / "predecoded.pc2",
    )

    assert result == (str(tmp_path / "predecoded.obj"), str(tmp_path / "predecoded.pc2"))
    assert np.array_equal(captured["obj"][0], predecoded_vertices[0])
    assert np.array_equal(captured["obj"][1], preloaded_model.faces)
    assert captured["pc2"] is predecoded_vertices


def test_save_smplx_verts_main_resolves_model_path_from_stageii_and_support_base_dir(
    monkeypatch, tmp_path
):
    support_dir = tmp_path / "support"
    support_model = support_dir / "smplx" / "male" / "model.npz"
    support_model.parent.mkdir(parents=True, exist_ok=True)
    support_model.write_bytes(b"npz")

    input_path = tmp_path / "tiny_stageii.pkl"
    _write_stageii_pickle(
        input_path,
        model_path="/old-machine/support_files/smplx/male/model.pkl",
        gender="male",
    )

    captured = {}

    monkeypatch.setattr(
        save_smplx_verts,
        "export_stageii_meshes",
        lambda **kwargs: captured.update(kwargs) or ("out.obj", "out.pc2"),
    )

    save_smplx_verts.main(
        [
            "--input-pkl",
            str(input_path),
            "--support-base-dir",
            str(support_dir),
        ]
    )

    assert captured["input_pkl"] == str(input_path)
    assert captured["model_path"] == str(support_model)


def test_save_smplx_verts_main_errors_when_model_path_cannot_be_resolved(tmp_path):
    input_path = tmp_path / "broken_stageii.pkl"
    input_path.write_bytes(
        pickle.dumps(
            {
                "fullpose": np.zeros((2, 165), dtype=np.float32),
                "betas": np.zeros(400, dtype=np.float32),
                "trans": np.zeros((2, 3), dtype=np.float32),
            }
        )
    )

    with pytest.raises(SystemExit) as excinfo:
        save_smplx_verts.main(
            [
                "--input-pkl",
                str(input_path),
            ]
        )

    assert excinfo.value.code == 2


def test_save_smplx_verts_main_errors_when_model_load_fails(monkeypatch, tmp_path):
    input_path = tmp_path / "broken_stageii.pkl"
    _write_stageii_pickle(
        input_path,
        model_path="/missing/support_files/smplx/male/model.npz",
        gender="male",
    )

    monkeypatch.setattr(
        save_smplx_verts.render_video,
        "load_render_model",
        lambda model_path: (_ for _ in ()).throw(FileNotFoundError("missing model asset")),
    )

    with pytest.raises(SystemExit) as excinfo:
        save_smplx_verts.main(
            [
                "--input-pkl",
                str(input_path),
            ]
        )

    assert excinfo.value.code == 2


def test_export_stageii_meshes_batch_reuses_loaded_model_and_mirrors_output_dir(
    monkeypatch, tmp_path
):
    support_dir = tmp_path / "support"
    shared_model = support_dir / "smplx" / "male" / "model.npz"
    shared_model.parent.mkdir(parents=True, exist_ok=True)
    shared_model.write_bytes(b"npz")

    input_root = tmp_path / "exports"
    swing_stageii = input_root / "subject01" / "swing_stageii.pkl"
    serve_stageii = input_root / "subject02" / "serve_stageii.pkl"
    _write_stageii_pickle(
        swing_stageii,
        model_path="/old-machine/support_files/smplx/male/model.pkl",
        gender="male",
    )
    _write_stageii_pickle(
        serve_stageii,
        model_path="/old-machine/support_files/smplx/male/model.pkl",
        gender="male",
    )

    load_calls = []
    loaded_model = object()
    export_calls = []

    monkeypatch.setattr(
        save_smplx_verts.render_video,
        "load_render_model",
        lambda model_path: load_calls.append(model_path) or loaded_model,
    )
    monkeypatch.setattr(
        save_smplx_verts,
        "export_stageii_meshes",
        lambda **kwargs: (
            export_calls.append(kwargs),
            (str(kwargs["obj_out"]), str(kwargs["pc2_out"])),
        )[-1],
    )

    output_dir = tmp_path / "mesh-only"
    results = save_smplx_verts.export_stageii_meshes_batch(
        input_pkls=[swing_stageii, serve_stageii],
        support_base_dir=support_dir,
        output_dir=output_dir,
        input_root=input_root,
    )

    expected_swing_obj = output_dir / "subject01" / "swing_stageii.obj"
    expected_swing_pc2 = output_dir / "subject01" / "swing_stageii.pc2"
    expected_serve_obj = output_dir / "subject02" / "serve_stageii.obj"
    expected_serve_pc2 = output_dir / "subject02" / "serve_stageii.pc2"

    assert load_calls == [str(shared_model)]
    assert [call["input_pkl"] for call in export_calls] == [str(swing_stageii), str(serve_stageii)]
    assert [call["model_path"] for call in export_calls] == [str(shared_model), str(shared_model)]
    assert [call["model"] for call in export_calls] == [loaded_model, loaded_model]
    assert [call["obj_out"] for call in export_calls] == [
        str(expected_swing_obj),
        str(expected_serve_obj),
    ]
    assert [call["pc2_out"] for call in export_calls] == [
        str(expected_swing_pc2),
        str(expected_serve_pc2),
    ]
    assert results == [
        {"obj_path": str(expected_swing_obj), "pc2_path": str(expected_swing_pc2)},
        {"obj_path": str(expected_serve_obj), "pc2_path": str(expected_serve_pc2)},
    ]


def test_save_smplx_verts_main_uses_output_dir_for_single_input(monkeypatch, tmp_path):
    support_dir = tmp_path / "support"
    support_model = support_dir / "smplx" / "male" / "model.npz"
    support_model.parent.mkdir(parents=True, exist_ok=True)
    support_model.write_bytes(b"npz")

    input_path = tmp_path / "tiny_stageii.pkl"
    _write_stageii_pickle(
        input_path,
        model_path="/old-machine/support_files/smplx/male/model.pkl",
        gender="male",
    )

    captured = {}

    monkeypatch.setattr(
        save_smplx_verts,
        "export_stageii_meshes",
        lambda **kwargs: captured.update(kwargs) or ("out.obj", "out.pc2"),
    )

    output_dir = tmp_path / "mesh-only"
    save_smplx_verts.main(
        [
            "--input-pkl",
            str(input_path),
            "--support-base-dir",
            str(support_dir),
            "--output-dir",
            str(output_dir),
        ]
    )

    expected_obj, expected_pc2 = default_stageii_output_paths(str(input_path))
    assert captured["obj_out"] == str(output_dir / Path(expected_obj).name)
    assert captured["pc2_out"] == str(output_dir / Path(expected_pc2).name)


def test_save_smplx_verts_main_dispatches_batch_export(monkeypatch, tmp_path):
    support_dir = tmp_path / "support"
    support_model = support_dir / "smplx" / "male" / "model.npz"
    support_model.parent.mkdir(parents=True, exist_ok=True)
    support_model.write_bytes(b"npz")

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

    monkeypatch.setattr(
        save_smplx_verts,
        "export_stageii_meshes_batch",
        lambda **kwargs: captured.update(kwargs) or [],
    )

    save_smplx_verts.main(
        [
            "--input-dir",
            str(tmp_path / "exports"),
            "--support-base-dir",
            str(support_dir),
            "--output-dir",
            str(tmp_path / "mesh-only"),
            "--fname-filter",
            "swing",
        ]
    )

    assert captured["input_pkls"] == [str(matching_stageii)]
    assert captured["support_base_dir"] == str(support_dir)
    assert captured["output_dir"] == str(tmp_path / "mesh-only")
    assert captured["input_root"] == str(tmp_path / "exports")


def test_save_smplx_verts_main_errors_when_input_dir_matches_no_stageii_pickles(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        save_smplx_verts.main(
            [
                "--input-dir",
                str(tmp_path / "exports"),
                "--fname-filter",
                "swing",
            ]
        )

    assert excinfo.value.code == 2
