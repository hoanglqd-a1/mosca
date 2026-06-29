import numpy as np
import imageio.v2 as imageio
import os


path = "/datasets/iphone/space-out-1/extra/segment_mask.npy"
mask = np.load(path)

mask_list = []
for i in range(mask.shape[0]):
    mask_list.append(mask[i].astype(np.uint8) * 255)


imageio.mimwrite(os.path.join("/datasets/iphone/space-out-1/extra/segment_mask.gif"), mask_list, fps=20)