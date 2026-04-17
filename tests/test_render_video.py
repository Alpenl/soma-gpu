import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import render_video

SUPPORT_ROOT = ROOT / "support_files"


def test_load_render_model_supports_real_smplx_npz_path():
    assert hasattr(render_video, "load_render_model")

    model = render_video.load_render_model(SUPPORT_ROOT / "smplx" / "male" / "model.npz")

    assert isinstance(model.faces, np.ndarray)
    assert model.faces.shape[1] == 3
    assert model.v_template.shape[0] > 0


def test_load_render_model_supports_real_smplx_pkl_path():
    assert hasattr(render_video, "load_render_model")

    model = render_video.load_render_model(SUPPORT_ROOT / "smplx" / "male" / "model.pkl")

    assert isinstance(model.faces, np.ndarray)
    assert model.faces.shape[1] == 3
    assert model.v_template.shape[0] > 0


def test_load_vertices_supports_stageii_pickle_with_loaded_npz_model(tmp_path):
    assert hasattr(render_video, "load_render_model")

    stageii_pkl = tmp_path / "synthetic_stageii.pkl"
    stageii_pkl.write_bytes(
        __import__("pickle").dumps(
            {
                "fullpose": np.zeros((2, 165), dtype=np.float32),
                "betas": np.zeros(400, dtype=np.float32),
                "trans": np.zeros((2, 3), dtype=np.float32),
            }
        )
    )

    model = render_video.load_render_model(SUPPORT_ROOT / "smplx" / "male" / "model.npz")
    vertices = render_video.load_vertices(stageii_pkl, model)

    assert vertices.shape[0] == 2
    assert vertices.shape[2] == 3
    assert np.isfinite(vertices).all()


def test_build_preview_jobs_supports_single_input_and_direct_output_mp4(tmp_path):
    assert hasattr(render_video, "build_preview_jobs")
    parser = render_video.build_parser()
    input_path = tmp_path / "subject_stageii.pkl"
    output_path = tmp_path / "subject_preview.mp4"
    args = parser.parse_args(
        [
            "--input-path",
            str(input_path),
            "--output-path",
            str(output_path),
            "--model-path",
            str(ROOT / "support_files" / "smplx" / "male" / "model.npz"),
        ]
    )

    jobs = render_video.build_preview_jobs(args)

    assert jobs == [(str(input_path), str(output_path))]


def test_resolve_camera_config_uses_frontal_preset_and_allows_overrides():
    assert hasattr(render_video, "resolve_camera_config")

    camera = render_video.resolve_camera_config(
        SimpleNamespace(
            camera_preset="frontal",
            camera_x=None,
            camera_y=None,
            camera_z=2.25,
            lookat_x=None,
            lookat_y=None,
            lookat_z=None,
            up_x=None,
            up_y=None,
            up_z=None,
        )
    )

    assert camera.camera_x == 0.0
    assert camera.camera_y == -3.0
    assert camera.camera_z == 2.25
    assert camera.lookat_x == 0.0
    assert camera.lookat_y == 0.0
    assert camera.lookat_z == 1.0
    assert camera.up_x == 0.0
    assert camera.up_y == 0.0
    assert camera.up_z == 1.0


def test_render_stageii_preview_provides_stable_single_file_entrypoint(monkeypatch, tmp_path):
    assert hasattr(render_video, "render_stageii_preview")
    captured = {}
    input_path = tmp_path / "single_stageii.pkl"
    output_path = tmp_path / "single_preview.mp4"

    def fake_render_preview_jobs(args):
        captured["args"] = args

    monkeypatch.setattr(render_video, "render_preview_jobs", fake_render_preview_jobs)

    result = render_video.render_stageii_preview(
        input_path=input_path,
        output_path=output_path,
        model_path=SUPPORT_ROOT / "smplx" / "male" / "model.npz",
    )

    assert result == str(output_path)
    assert captured["args"].input_path == str(input_path)
    assert captured["args"].output_path == str(output_path)
    assert captured["args"].camera_preset == "frontal"


def test_render_stageii_preview_infers_output_path_when_omitted(monkeypatch, tmp_path):
    captured = {}
    input_path = tmp_path / "single_stageii.pkl"

    def fake_render_preview_jobs(args):
        captured["args"] = args

    monkeypatch.setattr(render_video, "render_preview_jobs", fake_render_preview_jobs)

    result = render_video.render_stageii_preview(
        input_path=input_path,
        model_path=SUPPORT_ROOT / "smplx" / "male" / "model.npz",
    )

    expected_output = str(tmp_path / "single_stageii.mp4")
    assert result == expected_output
    assert captured["args"].output_path == expected_output
