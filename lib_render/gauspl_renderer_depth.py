#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
import numpy as np
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from utils.rigid_utils import from_homogenous, to_homogenous
from utils.graphics_utils import normal_from_depth_image
from pytorch3d.transforms import matrix_to_quaternion

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack((w, x, y, z), dim=-1)


def render_normal(viewpoint_cam, depth, alpha):
    intrinsic_matrix, extrinsic_matrix = viewpoint_cam.get_calib_matrix_nerf()

    normal_ref = normal_from_depth_image(depth, intrinsic_matrix.to(depth.device), extrinsic_matrix.to(depth.device))
    normal_ref = normal_ref
    normal_ref = normal_ref.permute(2,0,1)

    return normal_ref

def render(
    xyz, 
    rotation, 
    scale,
    opacity,
    shs,
    H,
    W,
    # Multiple way to specify camera
    CAM_K=None,
    fx=None,
    fy=None,
    cx=None,
    cy=None,

    d_reflvec, 
    iteration, 
    opt,
    scaling_modifier=1.0, 
    override_color=None,
    brdf=None,
    diffuse=None,
    specular=None,
    roughness=None,
    normal=None,
    view_pos=None,
    bg_color=[1.0, 1.0, 1.0], 
    ):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(xyz, dtype=xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    device = xyz.device

    # Set up rasterization configuration
    if CAM_K is not None:
        fx, fy, cx, cy = CAM_K[0, 0], CAM_K[1, 1], CAM_K[0, 2], CAM_K[1, 2]
    else:
        assert fx is not None, "fx is not provided"
        if fy is None:
            fy = fx
        if cx is None:
            cx = W // 2
        if cy is None:
            cy = H // 2

    # * Specially handle the non-centered camera, using first padding and finally crop
    # ! fix this bug on 2023.10.28, use abs!!
    if abs(H // 2 - cy) > 1.0 or abs(W // 2 - cx) > 1.0:
        center_handling_flag = True
        left_w, right_w = cx, W - cx
        top_h, bottom_h = cy, H - cy
        new_W = int(2 * max(left_w, right_w))
        new_H = int(2 * max(top_h, bottom_h))
    else:
        center_handling_flag = False
        new_W, new_H = W, H

    # ! 2023.10.27 Fix this bug, change the order, should use the new_W, new_H to compute FoV
    # Set up rasterization configuration
    FoVx = focal2fov(fx, new_W)
    FoVy = focal2fov(fy, new_H)
    tanfovx = math.tan(FoVx * 0.5)
    tanfovy = math.tan(FoVy * 0.5)

    viewmatrix = torch.from_numpy(
        getWorld2View2(np.eye(3), np.zeros(3)).transpose(0, 1)
    ).to(device)
    projection_matrix = (
        getProjectionMatrix(znear=0.0001, zfar=1.0, fovX=FoVx, fovY=FoVy)
        .transpose(0, 1)
        .to(device)
    )
    full_proj_transform = (
        viewmatrix.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))
    ).squeeze(0)
    camera_center = viewmatrix.inverse()[3, :3]

    raster_settings = GaussianRasterizationSettings(
        image_height=int(new_H),
        image_width=int(new_W),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewmatrix,
        projmatrix=full_proj_transform,
        sh_degree=3,
        campos=camera_center,
        prefiltered=False,
        debug=True,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = xyz
    means2D = screenspace_points

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    scales = scale
    rotations = matrix_to_quaternion(rotation)

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if iteration >= opt.warm_up + 3000:
        gb_pos = xyz
        view_pos = view_pos.repeat(xyz.shape[0], 1) # (N, 3) 
        d_viewdir_normalized = torch.nn.functional.normalize(view_pos - gb_pos)
    
    if colors_precomp is None:
        if iteration >= opt.warm_up2:
            color = brdf.shade(xyz[None, None, ...].detach(), normal[None, None, ...], d_reflvec[None, None, ...], diffuse[None, None, ...], specular[None, None, ...], roughness[None, None, ...], view_pos[None, None, ...])
            colors_precomp = color.squeeze() 
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, _ = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rendered_image = rendered_image.clamp(0, 1)
    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : (radii > 0).nonzero(),
        "radii": radii,
        }

    if iteration >= opt.warm_up + 3000:
        p_hom = torch.cat([xyz, torch.ones_like(xyz[...,:1])], -1).unsqueeze(-1)
        p_view = torch.matmul(viewpoint_camera.world_view_transform.transpose(0,1), p_hom)
        p_view = p_view[...,:3,:]
        depth = p_view.squeeze()[...,2:3]
        depth = depth.repeat(1,3)
        render_extras = {"depth": depth}
        normal_normed = 0.5*normal + 0.5  
        render_extras.update({"normal": normal_normed})
    
        out_extras = {}
        for k in render_extras.keys():
            if render_extras[k] is None: continue
            image = rasterizer(
                means3D = means3D,
                means2D = means2D,
                shs = None,
                colors_precomp = render_extras[k],
                opacities = opacity,
                scales = scales,
                rotations = rotations,
                cov3D_precomp = cov3D_precomp)[0]
            out_extras[k] = image
        for k in ["normal"]:
            if k in out_extras.keys():
                out_extras[k] = (out_extras[k] - 0.5) * 2. 
                torch.nn.functional.normalize(out_extras[k], p=2, dim=0)
        
        raster_settings_alpha = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=torch.tensor([0,0,0], dtype=torch.float32, device="cuda"),
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=3,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=True,
        )
        rasterizer_alpha = GaussianRasterizer(raster_settings=raster_settings_alpha)
        alpha = torch.ones_like(means3D) 
        out_extras["alpha"] =  rasterizer_alpha(
            means3D = means3D,
            means2D = means2D,
            shs = None,
            colors_precomp = alpha,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)[0]
        
        out_extras["normal_ref"] = render_normal(viewpoint_cam=viewpoint_camera, depth=out_extras['depth'][0], alpha=out_extras['alpha'][0])
        out.update(out_extras)
    return out

def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))

def getWorld2View2(R, t, translate=np.array([0.0, 0.0, 0.0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty
