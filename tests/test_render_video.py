import pickle
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

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


def test_load_vertices_supports_public_legacy_stageii_sample():
    assert hasattr(render_video, "load_render_model")

    model = render_video.load_render_model(SUPPORT_ROOT / "smplx" / "male" / "model.npz")
    vertices = render_video.load_vertices(ROOT / "support_data" / "tests" / "mosh_stageii.pkl", model)

    assert vertices.shape[0] > 0
    assert vertices.shape[2] == 3
    assert np.isfinite(vertices).all()


def test_load_vertices_defaults_missing_expression_to_neutral_instead_of_beta_slice(tmp_path):
    stageii_pkl = tmp_path / "synthetic_stageii.pkl"
    betas = np.zeros((2, 400), dtype=np.float32)
    betas[:, 300:310] = 5.0
    stageii_pkl.write_bytes(
        pickle.dumps(
            {
                "fullpose": np.zeros((2, 165), dtype=np.float32),
                "betas": betas,
                "trans": np.zeros((2, 3), dtype=np.float32),
            }
        )
    )

    captured = {}

    class RecordingModel:
        def __call__(self, **kwargs):
            import torch

            captured.update({name: value.detach().clone() for name, value in kwargs.items()})
            num_frames = kwargs["global_orient"].shape[0]
            return SimpleNamespace(
                vertices=torch.zeros(num_frames, 3, 3, dtype=torch.float32),
                joints=torch.zeros(num_frames, 3, 3, dtype=torch.float32),
            )

    render_video.load_vertices(stageii_pkl, RecordingModel())

    assert np.allclose(captured["expression"].cpu().numpy(), 0.0)


def test_load_vertices_neutral_face_zeros_jaw_eyes_and_expression(tmp_path):
    stageii_pkl = tmp_path / "synthetic_stageii.pkl"
    fullpose = np.zeros((2, 165), dtype=np.float32)
    fullpose[:, 66:75] = np.arange(1, 10, dtype=np.float32)
    stageii_pkl.write_bytes(
        pickle.dumps(
            {
                "fullpose": fullpose,
                "betas": np.zeros((2, 10), dtype=np.float32),
                "trans": np.zeros((2, 3), dtype=np.float32),
                "expression": np.full((2, 10), 3.0, dtype=np.float32),
            }
        )
    )

    captured = {}

    class RecordingModel:
        def __call__(self, **kwargs):
            import torch

            captured.update({name: value.detach().clone() for name, value in kwargs.items()})
            num_frames = kwargs["global_orient"].shape[0]
            return SimpleNamespace(
                vertices=torch.zeros(num_frames, 3, 3, dtype=torch.float32),
                joints=torch.zeros(num_frames, 3, 3, dtype=torch.float32),
            )

    render_video.load_vertices(stageii_pkl, RecordingModel(), neutral_face=True)

    assert np.allclose(captured["jaw_pose"].cpu().numpy(), 0.0)
    assert np.allclose(captured["leye_pose"].cpu().numpy(), 0.0)
    assert np.allclose(captured["reye_pose"].cpu().numpy(), 0.0)
    assert np.allclose(captured["expression"].cpu().numpy(), 0.0)


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


def test_build_parser_defaults_to_a5_delivery_render_settings():
    parser = render_video.build_parser()
    args = parser.parse_args(
        [
            "--input-path",
            "demo_stageii.pkl",
            "--model-path",
            str(ROOT / "support_files" / "smplx" / "male" / "model.npz"),
        ]
    )

    assert args.camera_preset == "fixed-front"
    assert args.width == 1024
    assert args.height == 1024
    assert args.supersample == 2
    assert args.ffmpeg_crf == 16
    assert args.ffmpeg_preset == "slow"
    assert args.neutral_face is True


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


def test_resolve_camera_config_uses_fixed_front_preset():
    camera = render_video.resolve_camera_config(
        SimpleNamespace(
            camera_preset="fixed-front",
            camera_x=None,
            camera_y=None,
            camera_z=None,
            lookat_x=None,
            lookat_y=None,
            lookat_z=None,
            up_x=None,
            up_y=None,
            up_z=None,
        )
    )

    assert camera.camera_x == -3.0
    assert camera.camera_y == 0.0
    assert camera.camera_z == 1.0
    assert camera.lookat_x == 0.0
    assert camera.lookat_y == 0.0
    assert camera.lookat_z == 1.0
    assert camera.up_x == 0.0
    assert camera.up_y == 0.0
    assert camera.up_z == 1.0


def test_resolve_camera_config_estimates_subject_frontal_from_stageii_pose():
    assert hasattr(render_video, "resolve_camera_config")

    fullpose = np.zeros((1, 165), dtype=np.float32)
    fullpose[0, 1] = -np.pi / 2.0
    stageii_inputs = {
        "fullpose": fullpose,
        "betas": np.zeros((1, 10), dtype=np.float32),
        "trans": np.array([[0.5, -0.25, 0.85]], dtype=np.float32),
        "expression": None,
    }

    camera = render_video.resolve_camera_config(
        SimpleNamespace(
            camera_preset="subject-frontal",
            camera_x=None,
            camera_y=None,
            camera_z=None,
            lookat_x=None,
            lookat_y=None,
            lookat_z=None,
            up_x=None,
            up_y=None,
            up_z=None,
        ),
        stageii_inputs=stageii_inputs,
    )

    assert camera.camera_x == pytest.approx(-2.5)
    assert camera.camera_y == pytest.approx(-0.25)
    assert camera.camera_z == pytest.approx(1.0)
    assert camera.lookat_x == pytest.approx(0.5)
    assert camera.lookat_y == pytest.approx(-0.25)
    assert camera.lookat_z == pytest.approx(1.0)
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
    assert captured["args"].camera_preset == "fixed-front"
    assert captured["args"].width == 1024
    assert captured["args"].height == 1024
    assert captured["args"].supersample == 2
    assert captured["args"].ffmpeg_crf == 16
    assert captured["args"].ffmpeg_preset == "slow"
    assert captured["args"].neutral_face is True


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


def test_render_stageii_preview_forwards_face_zeroing_args(monkeypatch, tmp_path):
    captured = {}

    def fake_render_preview_jobs(args):
        captured["args"] = args

    monkeypatch.setattr(render_video, "render_preview_jobs", fake_render_preview_jobs)

    render_video.render_stageii_preview(
        input_path=tmp_path / "single_stageii.pkl",
        model_path=SUPPORT_ROOT / "smplx" / "male" / "model.npz",
        neutral_face=True,
        zero_jaw=True,
        zero_expression=True,
    )

    assert captured["args"].neutral_face is True
    assert captured["args"].zero_jaw is True
    assert captured["args"].zero_expression is True


def test_render_stageii_preview_writes_mp4_without_taichi_scene_deprecation(tmp_path):
    pytest.importorskip("cv2")
    pytest.importorskip("taichi")

    input_path = tmp_path / "tiny_stageii.pkl"
    output_path = tmp_path / "tiny_stageii.mp4"
    input_path.write_bytes(
        pickle.dumps(
            {
                "fullpose": np.zeros((2, 165), dtype=np.float32),
                "betas": np.zeros(400, dtype=np.float32),
                "trans": np.zeros((2, 3), dtype=np.float32),
            }
        )
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = render_video.render_stageii_preview(
            input_path=input_path,
            output_path=output_path,
            model_path=SUPPORT_ROOT / "smplx" / "male" / "model.npz",
            width=128,
            height=128,
            fps=5,
            arch="cpu",
            force=True,
        )

    assert result == str(output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    assert not any("Instantiating ti.ui.Scene directly is deprecated" in str(warning.message) for warning in caught)


def test_render_vertices_to_video_writes_mp4_without_taichi_scene_deprecation(tmp_path):
    pytest.importorskip("cv2")
    pytest.importorskip("taichi")

    model = render_video.load_render_model(SUPPORT_ROOT / "smplx" / "male" / "model.npz")
    vertices = render_video.load_vertices(ROOT / "support_data" / "tests" / "mosh_stageii.pkl", model)[:2]
    output_path = tmp_path / "tiny_vertices.mp4"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = render_video.render_vertices_to_video(
            vertices=vertices,
            faces=model.faces,
            output_path=output_path,
            width=128,
            height=128,
            fps=5,
            arch="cpu",
        )

    assert result == str(output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    assert not any("Instantiating ti.ui.Scene directly is deprecated" in str(warning.message) for warning in caught)
