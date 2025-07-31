import numpy as np
import os
import os.path as osp
import torch

src = "/datasets/iphone/paper-windmill/extra/segment_dynamic_dep=sensor_bootstapir_tap.npz"
track = np.load(src, allow_pickle=True)
print(track.files)
print(track["tracks"].shape)