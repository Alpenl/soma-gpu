import os.path as osp
import numpy as np
from soma.tools.run_soma_multiple import run_soma_on_multiple_settings

soma_expr_id = 'V48_02_SuperSet'  # the checkpoint name, don't change
soma_data_id = 'OC_05_G_03_real_000_synt_100'

# data-specific settings
soma_mocap_target_ds_name = '20240205'
# subject_name = 'linxu'
# where the mocap data is stored
mocap_base_dir = "/home/user416/data/tennis_motion/mocap_raw"

soma_work_base_dir = '/home/user416/project/soma'
support_base_dir = osp.join(soma_work_base_dir, 'support_files')

fname_filter = ['zhou']

run_soma_on_multiple_settings(
    soma_expr_ids=[
        soma_expr_id,
    ],
    soma_mocap_target_ds_names=[
        soma_mocap_target_ds_name,
    ],
    soma_data_ids=[
        soma_data_id,
    ],
    soma_cfg={
        'soma.batch_size': 256,
        'dirs.support_base_dir':support_base_dir,
        'mocap.unit': 'mm',
        'save_c3d': True,
        'keep_nan_points': True,
        'remove_zero_trajectories': True
    },
    parallel_cfg={
        # 'max_num_jobs': 10,# comment to run on whole dataset
        'randomly_run_jobs': True,
    },
    run_tasks=[
        'soma',
    ],
    mocap_base_dir = mocap_base_dir,
    soma_work_base_dir=soma_work_base_dir,
    mocap_ext='.c3d',
    fname_filter=fname_filter,
)




mocap_dir = osp.join(mocap_base_dir, '..',
                         'soma_labeled_mocap_tracklet',
                         soma_mocap_target_ds_name)
# stagei_mocap_fnames = gen_stagei_mocap_fnames_random(mocap_dir, subject_name, ext='.pkl')

run_soma_on_multiple_settings(
    soma_expr_ids=[
        soma_expr_id,
    ],
    soma_mocap_target_ds_names=[
        soma_mocap_target_ds_name,
    ],
    soma_data_ids=
    [soma_data_id,],
    mosh_cfg={
        'moshpp.verbosity': 1,  # set to two to visualize the process in psbody.mesh.mesh_viewer
        # 'moshpp.stagei_frame_picker.stagei_mocap_fnames': stagei_mocap_fnames,
        'moshpp.stagei_frame_picker.type': 'random',

        'dirs.support_base_dir': support_base_dir,

        # 'mocap.end_fidx': 600  # comment in real runs
    },
    mocap_base_dir=mocap_base_dir,
    run_tasks=['mosh'],
    fname_filter=fname_filter,
    #         fast_dev_run=True,
    mocap_ext='.c3d',
    soma_work_base_dir=soma_work_base_dir,
    parallel_cfg={
        # 'max_num_jobs': 1,  # comment to run on all mocaps
        'randomly_run_jobs': True,
    },

)
