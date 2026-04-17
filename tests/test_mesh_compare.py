import importlib
import importlib.util
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.mesh_io import writePC2


def _load_mesh_compare_module():
    spec = importlib.util.find_spec("utils.mesh_compare")
    assert spec is not None, "utils.mesh_compare should provide the reusable mesh comparison entrypoint"
    return importlib.import_module("utils.mesh_compare")


def test_load_mesh_sequence_reads_stageii_pkl_and_uses_runtime_chunk_config(tmp_path, monkeypatch):
    mesh_compare = _load_mesh_compare_module()
    sample_path = tmp_path / "demo_stageii.pkl"
    sample_path.write_bytes(
        pickle.dumps(
            {
                "fullpose": np.zeros((5, 165), dtype=np.float32),
                "betas": np.zeros(10, dtype=np.float32),
                "trans": np.zeros((5, 3), dtype=np.float32),
                "markers_latent": np.zeros((1, 3), dtype=np.float32),
                "latent_labels": ["A"],
                "stageii_debug_details": {
                    "cfg": {
                        "surface_model": {
                            "type": "smplx",
                            "gender": "male",
                            "fname": "/old-machine/support_files/smplx/male/model.npz",
                        },
                        "runtime": {
                            "sequence_chunk_size": 32,
                            "sequence_chunk_overlap": 4,
                        },
                    }
                },
            }
        )
    )
    expected_vertices = np.arange(45, dtype=np.float32).reshape(5, 3, 3)
    fake_model = SimpleNamespace(name="model")
    captured = {}

    def fake_resolve(stageii_pkl, *, support_base_dir=None):
        captured["resolve_call"] = (str(stageii_pkl), support_base_dir)
        return "/resolved/model.npz"

    def fake_load_render_model(model_path):
        captured["model_path"] = model_path
        return fake_model

    def fake_load_vertices(stageii_pkl, model):
        captured["vertices_call"] = (str(stageii_pkl), model)
        return expected_vertices

    monkeypatch.setattr(mesh_compare, "resolve_stageii_model_path", fake_resolve)
    monkeypatch.setattr(mesh_compare.render_video, "load_render_model", fake_load_render_model)
    monkeypatch.setattr(mesh_compare.render_video, "load_vertices", fake_load_vertices)

    loaded = mesh_compare.load_mesh_sequence(sample_path, support_base_dir="/support-files")

    assert loaded.source_format == "stageii_pkl"
    assert loaded.chunk_size == 32
    assert loaded.chunk_overlap == 4
    assert loaded.vertices is expected_vertices
    assert captured["resolve_call"] == (str(sample_path), "/support-files")
    assert captured["model_path"] == "/resolved/model.npz"
    assert captured["vertices_call"] == (str(sample_path), fake_model)


def test_load_mesh_sequence_reads_pc2_and_accepts_explicit_chunk_config(tmp_path):
    mesh_compare = _load_mesh_compare_module()
    pc2_path = tmp_path / "demo.pc2"
    expected_vertices = np.asarray(
        [
            [[0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0]],
            [[2.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    writePC2(str(pc2_path), expected_vertices)

    loaded = mesh_compare.load_mesh_sequence(
        pc2_path,
        chunk_size=16,
        chunk_overlap=2,
    )

    assert loaded.source_format == "pc2"
    assert loaded.chunk_size == 16
    assert loaded.chunk_overlap == 2
    np.testing.assert_allclose(loaded.vertices, expected_vertices)


def test_compare_mesh_sequences_reports_mesh_accel_seam_and_frame_delta_summaries(tmp_path):
    mesh_compare = _load_mesh_compare_module()
    reference_path = tmp_path / "reference.pc2"
    candidate_path = tmp_path / "candidate.pc2"
    reference_vertices = np.asarray(
        [
            [[0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0]],
            [[2.0, 0.0, 0.0]],
            [[10.0, 0.0, 0.0]],
            [[11.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    candidate_vertices = np.asarray(
        [
            [[0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0]],
            [[2.0, 0.0, 0.0]],
            [[9.0, 0.0, 0.0]],
            [[10.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    writePC2(str(reference_path), reference_vertices)
    writePC2(str(candidate_path), candidate_vertices)

    report = mesh_compare.compare_mesh_sequences(
        reference_path,
        candidate_path,
        chunk_size=3,
        chunk_overlap=1,
    )

    assert report["reference"]["frames"] == 5
    assert report["reference"]["vertices_per_frame"] == 1
    assert report["reference"]["mesh_accel_l2"]["max"] == pytest.approx(7.0)
    assert report["reference"]["mesh_seam_jump_l2"]["mean"] == pytest.approx(8.0)
    assert report["candidate"]["mesh_accel_l2"]["max"] == pytest.approx(6.0)
    assert report["candidate"]["mesh_seam_jump_l2"]["mean"] == pytest.approx(7.0)
    assert report["frame_delta_l2"]["count"] == 5
    assert report["frame_delta_l2"]["mean"] == pytest.approx(0.4)
    assert report["frame_delta_l2"]["max"] == pytest.approx(1.0)


def test_compare_mesh_sequences_rejects_frame_shape_mismatch(tmp_path):
    mesh_compare = _load_mesh_compare_module()
    reference_path = tmp_path / "reference.pc2"
    candidate_path = tmp_path / "candidate.pc2"
    writePC2(
        str(reference_path),
        np.asarray(
            [
                [[0.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0]],
                [[2.0, 0.0, 0.0]],
            ],
            dtype=np.float32,
        ),
    )
    writePC2(
        str(candidate_path),
        np.asarray(
            [
                [[0.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0]],
            ],
            dtype=np.float32,
        ),
    )

    with pytest.raises(ValueError, match="same shape"):
        mesh_compare.compare_mesh_sequences(reference_path, candidate_path)


def test_mesh_compare_main_writes_json_report(tmp_path, monkeypatch):
    mesh_compare = _load_mesh_compare_module()
    output_path = tmp_path / "report.json"
    expected_report = {
        "reference": {"mesh_accel_l2": {"mean": 1.0}},
        "candidate": {"mesh_accel_l2": {"mean": 0.5}},
        "frame_delta_l2": {"mean": 0.25},
    }

    monkeypatch.setattr(mesh_compare, "compare_mesh_sequences", lambda *args, **kwargs: expected_report)

    result = mesh_compare.main(
        [
            "--reference",
            "reference.pc2",
            "--candidate",
            "candidate.pc2",
            "--output",
            str(output_path),
        ]
    )

    assert result == expected_report
    assert json.loads(output_path.read_text()) == expected_report
