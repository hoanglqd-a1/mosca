import numpy as np
import os
import os.path as osp
import torch

src = "/datasets/iphone/paper-windmill/logs/iphone_fit_colfree_native_add3_20250625_102123/bundle"
bundle_path = "bundle_cams.pth"
bundle = torch.load(osp.join(src, bundle_path))
print("Bundle keys:", bundle.keys())
print(bundle["_rel_focal"])