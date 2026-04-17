import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import save_smplx_verts
from utils.mesh_io import load_obj_mesh, readPC2
from utils.script_utils import default_stageii_output_paths

SUPPORT_ROOT = ROOT / "support_files"


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
