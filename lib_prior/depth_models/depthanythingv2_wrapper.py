import torch
import numpy as np
import os, os.path as osp
from tqdm import tqdm
import cv2
import sys
sys.path.append(osp.abspath(osp.dirname(__file__)))
from depth_anything_v2.depth_anything_v2.dpt import DepthAnythingV2


from depth_utils import viz_depth_list

@torch.no_grad()
def __call_model__(model, raw_image):
    """Call a depth model with multiple fallbacks and return a torch tensor depth map.

    Attempts to be compatible with several model APIs:
    - model.inference({"input": rgb}) -> (pred_depth, ...)
    - model.predict(rgb) -> numpy or torch
    - model(rgb) / model.forward(rgb) -> torch or numpy

    rgb_tensor: torch.Tensor on CUDA or CPU with shape (1, C, H, W), dtype float
    Returns: pred_depth torch.Tensor (H, W) on same device as rgb_tensor
    """
    pred = model.infer_image(raw_image)
    # Normalize pred to a single HxW torch tensor
    if isinstance(pred, tuple) or isinstance(pred, list):
        pred = pred[0]

    # If dict
    if isinstance(pred, dict):
        if "depth" in pred:
            pred = pred["depth"]
        else:
            vals = [v for v in pred.values()]
            pred = vals[0]

    # Convert numpy to tensor
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    # If tensor maybe batch/channel dims
    if torch.is_tensor(pred):
        # remove batch and channel if present
        if pred.dim() == 4:
            # (B,1,H,W) or (B,C,H,W)
            pred = pred.squeeze(0)
        if pred.dim() == 3:
            # (C,H,W) or (1,H,W)
            # if first dim is 1 or 3, assume (1,H,W) or (3,H,W)
            if pred.shape[0] == 1:
                pred = pred.squeeze(0)
            elif pred.shape[0] == 3:
                # unlikely for depth, but take first channel
                pred = pred[0]
        pred = pred.to(dtype=torch.float32)
        return pred

    raise RuntimeError("Model returned unsupported depth type")


@torch.no_grad()
def __process_frame__(img, out_fn, model, input_size=(480, 640), mean=None, std=None, fxfycxcy_pixel=None, canonical_focal=1000.0, default_fov_deg=53.13):
    """Preprocess a single image, call the model and save metric depth to out_fn (.npz).

    - img: numpy array (H,W,3) BGR (cv2) or RGB. We will assume BGR by default as in other wrappers.
    - model: loaded model object (must be on proper device)
    - input_size: desired model input (H,W)
    - mean/std: optional normalization vectors (if None use ImageNet-like defaults)
    - fxfycxcy_pixel: optional intrinsic [fx,fy,cx,cy] to convert canonical depth to metric
    - canonical_focal: focal used by models that output canonical-scale depths (e.g. 1000)

    Returns: depth numpy array (H,W) float32 saved to out_fn
    """
    # The DepthAnythingV2 model expects a raw image numpy array (H, W, 3).
    # Ensure input is a uint8/float32 numpy array in HWC format and pass it directly to model.infer_image.
    rgb_origin = img.copy()
    if rgb_origin.ndim != 3 or rgb_origin.shape[2] != 3:
        raise ValueError(f"Expected input image with shape (H,W,3), got {rgb_origin.shape}")

    # Ensure dtype is uint8 or float32; model.infer_image usually accepts either (values in 0..255)
    if rgb_origin.dtype != np.uint8 and rgb_origin.dtype != np.float32:
        rgb_origin = rgb_origin.astype(np.uint8)

    # Call the model with the raw image (no resizing/normalization/padding) as requested.
    pred_depth = __call_model__(model, rgb_origin)
    print(pred_depth.min(), pred_depth.max())

    # pred_depth is a torch tensor with same H,W as the input image (or compatible). Convert to torch tensor
    if not torch.is_tensor(pred_depth):
        pred_depth = torch.from_numpy(np.asarray(pred_depth))

    # If shapes mismatch, attempt to resize predicted depth to original image size
    h, w = rgb_origin.shape[:2]
    if pred_depth.dim() == 2 and (pred_depth.shape[0] != h or pred_depth.shape[1] != w):
        pred_depth = torch.nn.functional.interpolate(pred_depth[None, None, ...], size=(h, w), mode='bilinear', align_corners=False).squeeze()

    # Compute intrinsic if needed (used to convert canonical depth to metric)
    if fxfycxcy_pixel is None:
        f = min(h, w) / 2 / np.tan(np.radians(default_fov_deg / 2))
        intrinsic = [f, f, w / 2, h / 2]
    else:
        intrinsic = list(fxfycxcy_pixel)

    # Convert canonical depth to metric depth using focal scaling if applicable
    fx = intrinsic[0]
    metric_depth = pred_depth.to(dtype=torch.float32) * (fx / float(canonical_focal))
    metric_depth = torch.clamp(metric_depth, 0.0, 1000.0)

    dep = metric_depth.cpu().numpy().astype(np.float32)
    np.savez_compressed(out_fn, dep=dep)
    return dep


def get_depthanythingv2_model(device='cuda'):
    """Load a DepthAnything-v2 style model. This function tries a few common loading mechanisms.

    Returns a model on `device` in eval mode.
    """
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    encoder = 'vitl' # or 'vits', 'vitb', 'vitg'

    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(f'../../weights/depth_anything_v2_{encoder}.pth', map_location='cpu'))
    model = model.to(device).eval()
    return model

def depthanythingv2_process_folder(
    model,
    img_list,
    fn_list,
    dst,
    input_size=(480,640),
    mean=None,
    std=None,
    fxfycxcy_pixel=None,
    canonical_focal=1000.0,
    default_fov_deg=53.13,
    invalid_mask_list=None,
):
    print('DepthAnythingV2 processing...')
    assert len(img_list) == len(fn_list)
    os.makedirs(dst, exist_ok=True)
    dep_list = []
    for i in tqdm(range(len(fn_list))):
        fn = fn_list[i]
        img = img_list[i]
        save_fn = osp.basename(fn).replace('.jpg', '.npz').replace('.png', '.npz')
        out_fn = os.path.join(dst, save_fn)
        dep = __process_frame__(
            img, out_fn, model, input_size=input_size, mean=mean, std=std, fxfycxcy_pixel=fxfycxcy_pixel, canonical_focal=canonical_focal, default_fov_deg=default_fov_deg
        )
        if invalid_mask_list is not None:
            dep[invalid_mask_list[i] > 0] = 0
        dep_list.append(dep)
    
    # render depth map video
    viz_depth_list(dep_list, dst + '.mp4')

    return


if __name__ == '__main__':
    device = 'cuda'
    # example usage: load a model and run on images in a folder
    src = '/datasets/nerfds_2/temp/images'
    model = get_depthanythingv2_model(device=device)

    fns = os.listdir(src)
    fns.sort()
    img_list, fn_list = [], []
    for fn in fns:
        if fn.endswith('.jpg') or fn.endswith('.png'):
            img_list.append(cv2.imread(os.path.join(src, fn))[..., ::-1]) # read as BGR and convert to RGB
            fn_list.append(fn)

    torch.cuda.reset_peak_memory_stats()
    depthanythingv2_process_folder(model, img_list, fn_list, dst=osp.join('/datasets/nerfds_2/temp/debug', 'depth_da2'))

    # depth_dir = '/datasets/nerfds_2/temp/debug/depth_da2/'
    # save_dir = '/datasets/nerfds_2/temp/debug/'
    # debug_dep_list = []
    # for fn in sorted(os.listdir(depth_dir)):
    #     if fn.endswith('.npz'):
    #         data = np.load(os.path.join(depth_dir, fn))
    #         dep = data['dep']
    #         print(f"Depth range for {fn}: min={dep.min():.3f}, max={dep.max():.3f}")
            
    #         min_dep = dep.min()
    #         max_dep = dep.max()
    #         dep = (dep - min_dep) / (max_dep - min_dep + 1e-8) # normalize to [0,1] and invert for better visualization
    #         debug_dep_list.append(dep)

    # viz_depth_list(debug_dep_list, osp.join(save_dir, 'debug_depth.mp4'))