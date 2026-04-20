WORLD_FRONTAL_CAMERA_PRESET = "frontal"
SUBJECT_FRONTAL_CAMERA_PRESET = "subject-frontal"
FIXED_FRONT_CAMERA_PRESET = "fixed-front"
DEFAULT_CAMERA_PRESET = FIXED_FRONT_CAMERA_PRESET

SUBJECT_CAMERA_DISTANCE = 3.0
SUBJECT_CAMERA_LOOKAT_HEIGHT = 0.15

CAMERA_PRESETS = {
    WORLD_FRONTAL_CAMERA_PRESET: {
        "camera_x": 0.0,
        "camera_y": -3.0,
        "camera_z": 1.0,
        "lookat_x": 0.0,
        "lookat_y": 0.0,
        "lookat_z": 1.0,
        "up_x": 0.0,
        "up_y": 0.0,
        "up_z": 1.0,
    },
    FIXED_FRONT_CAMERA_PRESET: {
        "camera_x": -3.0,
        "camera_y": 0.0,
        "camera_z": 1.0,
        "lookat_x": 0.0,
        "lookat_y": 0.0,
        "lookat_z": 1.0,
        "up_x": 0.0,
        "up_y": 0.0,
        "up_z": 1.0,
    },
}

DEFAULT_VIDEO_FPS = 30
DEFAULT_VIDEO_WIDTH = 1024
DEFAULT_VIDEO_HEIGHT = 1024
DEFAULT_VIDEO_SUPERSAMPLE = 2
DEFAULT_VIDEO_FFMPEG_CRF = 16
DEFAULT_VIDEO_FFMPEG_PRESET = "slow"
DEFAULT_VIDEO_ARCH = "gpu"
DEFAULT_RENDER_NEUTRAL_FACE = True
DEFAULT_RENDER_ZERO_JAW = False
DEFAULT_RENDER_ZERO_EXPRESSION = False

DEFAULT_MOSH_FIT_CFG = {
    "surface_model.dof_per_hand": 24,
    "moshpp.optimize_fingers": True,
    "moshpp.optimize_face": False,
    "moshpp.optimize_toes": False,
    "moshpp.optimize_betas": True,
    "moshpp.optimize_dynamics": False,
    "moshpp.stagei_frame_picker.type": "random",
    "moshpp.stagei_frame_picker.seed": 100,
    "moshpp.stagei_frame_picker.num_frames": 12,
    "opt_settings.maxiter": 100,
    "opt_settings.weights.stageii_wt_data": 400,
    "opt_settings.weights.stageii_wt_velo": 2.5,
    "opt_settings.weights.stageii_wt_expr": 1.0,
    "opt_settings.weights.stageii_wt_poseB": 1.6,
    "opt_settings.weights.stageii_wt_poseH": 1.0,
    "opt_settings.weights.stageii_wt_poseF": 1.0,
}

# GPU per-frame solver config (quality-matched to CPU dogleg)
DEFAULT_GPU_PERFRAME_RUNTIME = {
    "backend": "torch",
    "device": "cuda",
    "sequence_chunk_size": 1,
    "refine_lr": 0.05,
    "sequence_seed_refine_iters": 5,
}
