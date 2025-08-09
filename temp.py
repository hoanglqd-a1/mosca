import os
import numpy as np
import os.path as osp
from glob import glob
import shutil
import torch
import imageio
from sam2.sam2_video_predictor import SAM2VideoPredictor
import matplotlib.pyplot as plt
from PIL import Image

src_dir = "/datasets/iphone/block/epi/error"

def load_epi_error(save_dir):
    fns = [f for f in os.listdir(save_dir) if f.endswith(".npy")]
    fns.sort()
    epi_error = []
    for fn in fns:
        epi_error.append(np.load(osp.join(save_dir, fn)))
    epi_error = np.stack(epi_error, 0)
    epi_error = torch.tensor(epi_error).float()
    return epi_error

epi = load_epi_error(src_dir)
epi_mask = epi > 1e-2
resampling_mask_dilate_ksize = 7
sample_mask = (
    torch.nn.functional.max_pool2d(
        epi_mask[:, None].float(),
        kernel_size=resampling_mask_dilate_ksize,
        stride=1,
        padding=(resampling_mask_dilate_ksize - 1) // 2,
    )[:, 0]
    > 0.5
)

imageio.mimsave(
    "epi_resample_mask.gif",
    sample_mask.cpu().numpy().astype(np.uint8) * 255,
)

