import os
import numpy as np
from tqdm import tqdm
from moshpp.tools.mocap_interface import read_mocap, write_mocap_c3d


if __name__ == '__main__':
    base_dir = "/home/user416/data/tennis_motion/mocap_raw/20240205/linxu"
    fnames = [k.replace('.npy', '') for k in os.listdir(base_dir) if k.endswith('.npy') and '_racket' not in k]
    for fname in tqdm(fnames):

        save_path = os.path.join(base_dir, f'{fname}.c3d')
        if os.path.exists(save_path):
            print(f"Skipping {save_path}")
            continue
        markers = np.load(os.path.join(base_dir, f"{fname}.npy")).astype(float)
        num_frames, num_markers, _ = markers.shape
        write_mocap_c3d(markers=markers, labels=[str(k) for k in range(num_markers)],
                        out_mocap_fname=save_path, frame_rate=30)