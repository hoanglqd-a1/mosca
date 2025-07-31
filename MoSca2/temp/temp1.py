import numpy as np
import cv2
import os
import os.path as osp
import shutil
import torch

src_dir = "/datasets/iphone/paper-windmill/logs/iphone_fit_colfree_native_add3_20250622_021432/"
epi_track_mask_dir = "track_identification.npz"

epi_track_mask = np.load(osp.join(src_dir, epi_track_mask_dir))
static_epi_mask = epi_track_mask["static_track_mask"]
print(static_epi_mask.shape)
print(static_epi_mask)
print((static_epi_mask == True).sum())

test = "track_test"
for th in os.listdir(osp.join(src_dir, test)):
    track = torch.load(osp.join(src_dir, test, th))
    print(track.shape)
    print((track == True).sum())