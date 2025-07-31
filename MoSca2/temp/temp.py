import cv2
import numpy as np
import os
import shutil
import torch 
import imageio
import os.path as osp

img_dir = "/datasets/iphone/paper-windmill/images"
log_path = "/datasets/iphone/paper-windmill/logs/iphone_fit_colfree_native_add3_20250619_153210"
if os.path.exists(osp.join(log_path, "temp")):
    shutil.rmtree(osp.join(log_path, "temp"))
os.mkdir(osp.join(log_path, "temp"))
photo_data = np.load(osp.join(log_path, "photo_warmup_rendered.npz"))
device = torch.device("cuda:0")
img_fns = [
                f
                for f in os.listdir(img_dir)
                if f.endswith(".jpg") or f.endswith(".png")
            ]
img_fns.sort()
img_names = [osp.splitext(f)[0] for f in img_fns]
images = [imageio.imread(osp.join(img_dir, img_fn)) for img_fn in img_fns]
images = torch.Tensor(np.stack(images)) / 255.0  # T,H,W,3
images = images[..., :3]
rgb = images.to(device)

rgb_rendered = torch.tensor(photo_data["rgb"]).to(device).permute(0, 2, 3, 1)
render_error = abs(rgb - rgb_rendered).max(dim=-1).values
render_error_mask = render_error > 0.01
render_error_mask_viz = (render_error_mask[..., None] * rgb).cpu().numpy()
imageio.mimsave(
    osp.join(log_path, "temp", f"render_error_mask_th={0.01:.3f}.gif"),
    (render_error_mask_viz * 255).astype(np.uint8),
)

cv2.imwrite(osp.join(osp.join(log_path, "temp"), f"{0}.png"), (photo_data["rgb"].transpose(0, 2, 3, 1)[0] * 255).astype(np.uint8))
cv2.imwrite(osp.join(osp.join(log_path, "temp"), f"input_{0}.png"), (rgb[0] * 255).cpu().numpy().astype(np.uint8))