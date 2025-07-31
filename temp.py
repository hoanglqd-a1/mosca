import os
import numpy as np
import cv2
import os.path as osp
from glob import glob
import shutil

src_dir = "/datasets/iphone/paper-windmill/"
segment_dir = "dynamic_mask"

total_mask = []
for file in sorted(glob(osp.join(src_dir, segment_dir, "*.jpg"))):
    mask = cv2.imread(file, cv2.IMREAD_GRAYSCALE) > 127
    total_mask.append(np.float32(mask))

np.save(osp.join(src_dir, "segment_mask.npy"), total_mask)