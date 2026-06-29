# refine depth using normal map
import torch
from PIL import Image
import numpy as np
import os, os.path as osp
from tqdm import tqdm
import cv2
from matplotlib import cm
import imageio
from matplotlib import pyplot as plt
import sys
import torch.nn.functional as F
from typing import Optional
from lib_prior.depth_models.depth_utils import viz_depth_list

def smooth_depth(depth, kernel_size=5):
    kernel = cv2.getGaussianKernel(kernel_size, 0)
    kernel = kernel @ kernel.T  # 2D Gaussian kernel
    kernel = torch.from_numpy(kernel).float().to(depth.device)
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # shape (1,1,k,k)
    padding = kernel_size // 2
    depth = depth.unsqueeze(0).unsqueeze(0)  # shape (1,1,H,W)
    smoothed_depth = F.conv2d(depth, kernel, padding=padding)
    return smoothed_depth.squeeze(0).squeeze(0)

def gradient_x(img):
    return img[:, :, :, :-1] - img[:, :, :, 1:]

def gradient_y(img):
    return img[:, :, :-1, :] - img[:, :, 1:, :]

def edge_aware_smoothness_loss(depth, image):
    """
    depth: [B,1,H,W]
    image: [B,3,H,W]
    """

    depth_dx = gradient_x(depth)
    depth_dy = gradient_y(depth)

    image_dx = gradient_x(image)
    image_dy = gradient_y(image)

    # average RGB gradients
    weight_x = torch.exp(-torch.mean(torch.abs(image_dx), dim=1, keepdim=True))
    weight_y = torch.exp(-torch.mean(torch.abs(image_dy), dim=1, keepdim=True))

    smoothness_x = depth_dx.abs() * weight_x
    smoothness_y = depth_dy.abs() * weight_y

    loss = smoothness_x.mean() + smoothness_y.mean()

    return loss

def detect_depth_occlusion_boundaries(depth_map, threshold=10, ksize=5):
    error = cv2.Laplacian(depth_map, cv2.CV_64F, ksize=ksize)
    error = np.abs(error)
    _, occlusion_boundaries = cv2.threshold(error, threshold, 255, cv2.THRESH_BINARY)
    return occlusion_boundaries.astype(np.uint8), error

def laplacian_filter_depth(depths, threshold_ratio=0.5, ksize=5, open_ksize=3):
    # filter the depth changing boundary, they are not reliable
    dep_boundary_errors, dep_valid_masks = [], []
    ellip_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (open_ksize, open_ksize)
    )
    for dep in depths:
        # detect the edge boundary of depth
        dep = dep.astype(np.float32)
        # ! to handle different scale, the threshold should be adaptive
        threshold = np.median(dep) * threshold_ratio
        mask, error = detect_depth_occlusion_boundaries(dep, threshold, ksize)
        mask = mask > 0.5
        mask = ~mask  # valid mask
        # ! do a morph operator to remove outliers
        mask_opened = cv2.morphologyEx(
            mask.astype(np.uint8), cv2.MORPH_OPEN, ellip_kernel
        )
        mask_opened = mask_opened > 0
        # mask_opened = mask
        dep_valid_masks.append(mask_opened)
        dep_boundary_errors.append(error)
    dep_valid_masks = np.stack(dep_valid_masks, axis=0)
    dep_boundary_errors = np.stack(dep_boundary_errors, axis=0)
    return dep_valid_masks, dep_boundary_errors

def depth_to_normal_map(depth: torch.Tensor, fov_deg: float, cx: Optional[float] = None, cy: Optional[float] = None, eps: float = 1e-8):
    """
    Compute a normal map from a single-channel depth map using a pinhole camera
    model with the given vertical field-of-view in degrees.

    Args:
        depth: (H, W) torch tensor of depth values in the camera Z direction.
        fov_deg: vertical field of view in degrees (uses image height to compute
                 focal length in pixels: f = H / (2 * tan(fov/2))).
        cx, cy: optional principal point coordinates in pixels. If None, assumed
                at (W/2, H/2).
        eps: small epsilon to avoid division by zero.

    Returns:
        normal_map: (H, W, 3) float tensor with unit-length normals in camera
                    coordinates. Invalid pixels (where depth<=0 or NaN) will be
                    zeroed.
        valid_mask: (H, W) bool tensor indicating which normals are valid.
    """
    assert depth.ndim == 2, "depth must be HxW"
    device = depth.device
    H, W = depth.shape
    # compute focal length from vertical fov (assume fx = fy)
    fov_rad = float(fov_deg) * np.pi / 180.0
    # focal in pixels using image height
    f = float(H) / (2.0 * np.tan(fov_rad / 2.0) + 1e-12)

    if cx is None:
        cx = (W - 1) / 2.0
    if cy is None:
        cy = (H - 1) / 2.0

    # pixel coordinate grid
    xs = torch.arange(0, W, device=device, dtype=depth.dtype).view(1, W).expand(H, W)
    ys = torch.arange(0, H, device=device, dtype=depth.dtype).view(H, 1).expand(H, W)

    # backproject to camera coords
    z = depth
    valid = (z > 0) & torch.isfinite(z)
    x = (xs - cx) / f * z
    y = (ys - cy) / f * z

    pts = torch.stack([x, y, z], dim=-1)  # H,W,3

    # central differences for interior pixels
    p_x = pts[1:-1, 2:, :] - pts[1:-1, :-2, :]  # H-2, W-2, 3
    p_y = pts[2:, 1:-1, :] - pts[:-2, 1:-1, :]  # H-2, W-2, 3

    # cross product p_x x p_y gives normal (right-hand rule). We want normals
    # oriented towards the camera; will fix orientation below.
    n = torch.cross(p_x, p_y, dim=-1)
    n_norm = torch.norm(n, dim=-1, keepdim=True)
    n = n / (n_norm + eps)

    # assemble full normal map, pad borders with zeros
    normal_map = torch.zeros((H, W, 3), device=device, dtype=depth.dtype)
    normal_map[1:-1, 1:-1, :] = n

    # valid mask requires center and neighbor depths to be valid
    valid_c = valid[1:-1, 1:-1]
    valid_px = valid[1:-1, 2:]
    valid_mx = valid[1:-1, :-2]
    valid_py = valid[2:, 1:-1]
    valid_my = valid[:-2, 1:-1]
    valid_center = valid_c & valid_px & valid_mx & valid_py & valid_my

    mask = torch.zeros((H, W), device=device, dtype=torch.bool)
    mask[1:-1, 1:-1] = valid_center

    # orient normals to face camera: in camera coords, camera looks along +z,
    # so normals toward the camera should have negative z component (pointing
    # back to camera). Flip those with positive z.
    nz = normal_map[..., 2]
    flip = nz > 0
    normal_map[flip] = -normal_map[flip]

    # zero invalid normals
    normal_map[~mask] = 0.0

    return normal_map, mask

def depth_to_normal_plane_fit(
    depth,
    fov_deg,
    window_size=5,
    cx=None,
    cy=None,
    eps=1e-6,
):
    """
    Compute normals from depth using local least-squares plane fitting.

    Args:
        depth: (H,W)
        fov_deg: vertical field of view
        window_size: odd number (3,5,7,...)

    Returns:
        normal_map: (H,W,3)
        valid_mask: (H,W)
    """

    assert depth.ndim == 2
    assert window_size % 2 == 1

    device = depth.device
    dtype = depth.dtype

    H, W = depth.shape

    # Camera intrinsics
    fov_rad = np.deg2rad(fov_deg)
    f = H / (2.0 * np.tan(fov_rad / 2.0))

    if cx is None:
        cx = (W - 1) / 2.0

    if cy is None:
        cy = (H - 1) / 2.0

    # Unproject depth
    xs = torch.arange(W, device=device, dtype=dtype)
    ys = torch.arange(H, device=device, dtype=dtype)

    xs = xs[None, :].expand(H, W)
    ys = ys[:, None].expand(H, W)

    z = depth

    x = (xs - cx) / f * z
    y = (ys - cy) / f * z

    valid = torch.isfinite(z) & (z > 0)

    #
    # Extract local windows
    #
    pad = window_size // 2
    K = window_size * window_size

    xyz = torch.stack([x, y, z], dim=0)  # (3,H,W)

    patches = F.unfold(
        xyz.unsqueeze(0),
        kernel_size=window_size,
        padding=pad
    ) # (1,3*K,N)
    N = H * W

    patches = patches.view(3, K, N).permute(2, 1, 0) # (N,K,3)

    # Validity patches
    valid_patches = F.unfold(
        valid.float()[None, None],
        kernel_size=window_size,
        padding=pad
    )

    valid_patches = valid_patches.squeeze(0).T # (N,K)

    # Build weighted least squares
    X = patches[..., 0]
    Y = patches[..., 1]
    Z = patches[..., 2]

    Wmask = valid_patches

    #
    # Design matrix:
    #
    # z = a*x + b*y + c
    #
    A = torch.stack([
        X,
        Y,
        torch.ones_like(X)
    ], dim=-1)

    # (N,K,3)

    Aw = A * Wmask[..., None]
    Zw = Z * Wmask

    # Normal equations
    ATA = torch.matmul(
        Aw.transpose(1, 2),
        Aw
    )

    ATZ = torch.matmul(
        Aw.transpose(1, 2),
        Zw.unsqueeze(-1)
    )

    # Regularization
    eye = torch.eye(
        3,
        device=device,
        dtype=dtype
    )

    ATA = ATA + eps * eye[None]

    coeff = torch.linalg.solve(
        ATA,
        ATZ
    ).squeeze(-1)

    a = coeff[:, 0]
    b = coeff[:, 1]

    # Plane normal
    normals = torch.stack([
        -a,
        -b,
        torch.ones_like(a)
    ], dim=-1)

    normals = normals / (
        torch.norm(normals, dim=-1, keepdim=True)
        + eps
    )

    #
    # Orient toward camera
    #
    flip = normals[:, 2] > 0
    normals[flip] *= -1

    #
    # Validity
    #
    min_points = K // 2

    valid_center = (
        valid_patches.sum(dim=1) > min_points
    )

    normal_map = normals.view(H, W, 3)
    valid_mask = valid_center.view(H, W)

    normal_map[~valid_mask] = 0
    normal_map[~valid_mask] /= (torch.norm(normal_map[~valid_mask], dim=-1, keepdim=True) + eps)

    return normal_map, valid_mask

def refine_depth_with_normal(depth, depth_confidence, normal, image, fov_deg, cx=None, cy=None, eps=1e-8):
    """
    Refine a depth map using a normal map by enforcing local consistency between depth gradients and normals.

    Args:
        depth: (T, H, W) torch tensor of depth values in the camera Z direction.
        depth_confidence: (T, H, W) torch tensor of confidence values for each depth pixel (0 to 1).
        normal: (T, H, W, 3) torch tensor of unit-length normals in camera coordinates.
        fov_deg: vertical field of view in degrees (uses image height to compute focal length).
        cx, cy: optional principal point coordinates in pixels. If None, assumed at (W/2, H/2).
        eps: small epsilon to avoid division by zero.
    Returns:
        refined_depth: (T, H, W) torch tensor of refined depth values.
    """
    device = "cuda"
    refined_depth = depth.clone().to(device)
    refined_depth.requires_grad = True
    depth_confidence = depth_confidence.clone().to(device)
    normal = normal.to(device)
    image = image.to(device)

    dep_mask, _ = laplacian_filter_depth(depth.cpu().numpy(), threshold_ratio=1.0, ksize=5, open_ksize=3)

    dep_mask = torch.from_numpy(dep_mask).to(device)
    high_confidence_mask = (depth_confidence > 0.9)

    high_conf_percent = high_confidence_mask.float().mean().item() * 100
    print(f"High confidence pixels: {high_conf_percent:.2f}%")
    if high_conf_percent < 10:
        print("Warning: low percentage of high confidence pixels, optimization may not be effective")

    epoch = 100
    optimizer = torch.optim.Adam([refined_depth], lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch)
    max_frame = 500

    original_depth = depth.clone().detach().to(device)
    for it in tqdm(range(epoch)):
        optimizer.zero_grad()
        pred_normal, valid_mask = [], []
        
        random_indices = torch.randperm(refined_depth.shape[0])[:max_frame]
        batch_refined_depth = refined_depth[random_indices]  # randomly sample frames for efficiency
        batch_depth = original_depth[random_indices]
        batch_dep_mask = dep_mask[random_indices]
        batch_normal = normal[random_indices]

        for t in range(batch_depth.shape[0]):
            # n, m = depth_to_normal_map(refined_depth[t], fov_deg, cx, cy, eps)
            n, m = depth_to_normal_map(batch_refined_depth[t], fov_deg, cx=cx, cy=cy, eps=eps)
            pred_normal.append(n)
            valid_mask.append(m)
        pred_normal = torch.stack(pred_normal, dim=0)  # T,H,W,3
        valid_mask = torch.stack(valid_mask, dim=0)  # T,H,W

        assert pred_normal.shape == batch_normal.shape, f"Pred normal shape {pred_normal.shape} does not match input normal shape {batch_normal.shape}"
        depth_loss = F.mse_loss(batch_refined_depth[batch_dep_mask], batch_depth[batch_dep_mask])
        normal_loss = torch.mean(1 + torch.einsum("thwc, thwc -> thw", pred_normal, batch_normal)[valid_mask])

        smoothness_loss = edge_aware_smoothness_loss(refined_depth[:, None, :, :], image.permute(0, 3, 1, 2))

        loss = 0.0 * depth_loss + normal_loss + smoothness_loss * 0.5
        loss.backward()

        refined_depth.grad[high_confidence_mask | ~dep_mask] = 0.0 # only update low confidence pixels

        optimizer.step()
        scheduler.step()

        if it % 10 == 0 or it == epoch - 1:
            print(f"Iter {it}: Loss={loss.item():.4f}, Depth Loss={depth_loss.item():.4f}, Normal Loss={normal_loss.item():.4f}, Smoothness Loss={smoothness_loss.item():.4f}")

    return refined_depth.detach().cpu()

if __name__ == "__main__":
    workdir = "/datasets/nerfds_4/sieve"
    image_dir = osp.join(workdir, "images")
    depth_dir = osp.join(workdir, "metric3d_depth")
    normal_dir = osp.join(workdir, "normals/normals_npy")

    default_fov_deg = 49.66475

    depth_list = []
    depth_confidence_list = []
    normal_list = []
    image_list = []
    compute_normal_from_depth_list = []
    smooth_depth_list = []
    for fn in sorted(os.listdir(depth_dir)):
        if fn.endswith(".npz"):
            depth = torch.from_numpy(np.load(osp.join(depth_dir, fn))['dep'])
            depth_confidence = torch.from_numpy(np.load(osp.join(depth_dir, fn))['confidence'])
            depth_list.append(depth)
            depth_confidence_list.append(depth_confidence)
            smooth_depth_list.append(smooth_depth(depth))
            # compute_normal_from_depth_list.append(depth_to_normal_plane_fit(smoothed_depth, default_fov_deg))

    # imageio.mimsave(osp.join(workdir, "depth_confidence.mp4"), [(cm.viridis(dc.cpu().numpy())[:, :, :3] * 255).astype(np.uint8) for dc in depth_confidence_list])
    # imageio.mimsave(osp.join(workdir, "depth_conf_mask.mp4"), [(dc.cpu().numpy() > 0.9).astype(np.uint8) * 255 for dc in depth_confidence_list])

    viz_depth_list(smooth_depth_list, osp.join(workdir, "smoothed_depth.mp4"))
    exit()
    # imageio.mimsave(osp.join(workdir, "normal_from_depth.mp4"), [(-cnf[0] + 1) / 2 for cnf in compute_normal_from_depth_list])

    for fn in sorted(os.listdir(normal_dir)):
        if fn.endswith(".npy"):
            normal = torch.from_numpy(np.load(osp.join(normal_dir, fn)).transpose(1, 2, 0)).to(torch.float32)
            normal_list.append(normal)

    for fn in sorted(os.listdir(image_dir)):
        if fn.endswith(".jpg") or fn.endswith(".png"):
            image = torch.from_numpy(imageio.imread(osp.join(image_dir, fn))).float() / 255.0
            image_list.append(image)
    
    refined_depth = refine_depth_with_normal(
        torch.stack(depth_list, dim=0),
        torch.stack(depth_confidence_list, dim=0),
        torch.stack(normal_list, dim=0),
        torch.stack(image_list, dim=0),
        default_fov_deg,
    )

    refined_depth = refined_depth.cpu().numpy()
    viz_depth_list(refined_depth, osp.join(workdir, "refined_depth.mp4"))

    refined_normal_map_list = []
    for t in range(refined_depth.shape[0]):
        refined_normal_map, _ = depth_to_normal_map(torch.from_numpy(refined_depth[t]), default_fov_deg)
        refined_normal_map_list.append(refined_normal_map.cpu().numpy())
    refined_normal_map_list = np.stack(refined_normal_map_list, axis=0)
    imageio.mimsave(osp.join(workdir, "refined_normal_from_depth.mp4"), [(-rn + 1) / 2 for rn in refined_normal_map_list])