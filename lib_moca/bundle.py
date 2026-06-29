# Single File
from matplotlib import pyplot as plt
import torch, numpy as np
import os, sys, os.path as osp
from tqdm import tqdm
import logging, imageio
from pytorch3d.ops import knn_points
from matplotlib import cm
import cv2
from tqdm import tqdm

sys.path.append(osp.dirname(osp.abspath(__file__)))


from camera import MonocularCameras
from viz_helper import make_video_from_pattern, viz_global_ba
from robust_utils import positive_th_gaussian_decay


def compute_static_ba(
    s2d,
    log_dir,
    s_track,
    s_track_valid_mask,
    cams: MonocularCameras,
    max_t_per_step=10000,
    total_steps=2000,  # 6000
    switch_to_ind_step=1000,  # this is also the scheduler start!
    max_num_of_tracks=10000,
    depth_correction_after_step=1000,
    # lr and lambda
    lr_cam_q=0.0003,
    lr_cam_t=0.0003,
    lr_cam_f=0.0003,
    lr_dep_s=0.001,
    lr_dep_c=0.001,
    lambda_flow=1.0,
    lambda_depth=0.1,
    lambda_small_correction=0.01,
    # camera pose smoothness
    lambda_cam_smooth_trans=0.0,
    lambda_cam_smooth_rot=0.0,
    # viz
    viz_verbose_n=300,
    viz_fig_n=300,
    viz_denser_range=[],  # [[0, 10]],  # [[0, 40], [1000, 1040]],
    viz_denser_interval=1,
    save_more_flag=False,
    viz_video_rgb=None,
    # robustify
    huber_delta=-1,
    # ! all these robust weights are computed on the fly, which assume that the initializaiton is good
    depth_decay_th=2.0,
    depth_decay_sigma=1.0,
    std_decay_th=0.2,
    std_decay_sigma=0.2,
    #
    optimizer_class=torch.optim.Adam,
    s_optimal=None,
    a=None,
    b=None,
):
    if s_optimal is None:
        s_optimal = 1.0

    viz_dir = osp.join(log_dir, "static_ba_viz")
    os.makedirs(viz_dir, exist_ok=True)

    # prepare dense track
    # s_track = s2d.track[:, s2d.static_track_mask, :2].clone()
    s_track = s_track[..., :2].clone()
    s_track_valid_mask = s_track_valid_mask.clone()
    device = s_track.device
    if max_num_of_tracks < s_track.shape[1]:
        logging.info(
            f"Track is too dense {s_track.shape[1]}, radom sample to {max_num_of_tracks}"
        )
        choice = torch.randperm(s_track.shape[1])[:max_num_of_tracks]
        s_track = s_track[:, choice]
        s_track_valid_mask = s_track_valid_mask[:, choice]

    homo_list, dep_list, rgb_list = prepare_track_homo_dep_rgb_buffers(
        s2d, s_track, s_track_valid_mask, torch.arange(s2d.T).to(device)
    )

    if viz_video_rgb is not None:
        logging.info(f"Viz BA points on each frame...")
        viz_frames = viz_ba_point(viz_video_rgb, s_track, s_track_valid_mask)
        imageio.mimsave(osp.join(log_dir, "BA_points.mp4"), viz_frames)

    # * start solve global init of the camera
    logging.info(
        f"Static Scaffold BA: Depth correction after {depth_correction_after_step}; Lr Scheduling and Ind after {switch_to_ind_step} steps (total {total_steps})"
    )
    if a is None and b is None:
        param_scale = torch.ones(cams.T).to(device)
        param_scale.requires_grad_(True)
        param_dep_corr = torch.zeros(cams.T).to(device)
        param_dep_corr.requires_grad_(True)
    else:
        param_scale = a.clone().to(device)
        param_dep_corr = b.clone().to(device)
        param_scale.requires_grad_(True)
        param_dep_corr.requires_grad_(True)
        depth_correction_after_step = 0  # if already have a,b, then directly optimize the depth correction
        logging.info(f"Initialize depth scale with a range [{param_scale.min().item():.4f}, {param_scale.max().item():.4f}] and depth correction with a range [{param_dep_corr.min().item():.4f}, {param_dep_corr.max().item():.4f}]")

    optim_list = cams.get_optimizable_list(lr_f=lr_cam_f, lr_q=lr_cam_q, lr_t=lr_cam_t)
    if lr_dep_s > 0:
        optim_list.append(
            {"params": [param_scale], "lr": lr_dep_s, "name": "cam_scale"}
        )
    if lr_dep_c > 0:
        optim_list.append(
            {"params": [param_dep_corr], "lr": lr_dep_c, "name": "dep_correction"}
        )
    optimizer = optimizer_class(optim_list)
    scheduler = None
    s_track_valid_mask_w = s_track_valid_mask.float()
    s_track_valid_mask_w = s_track_valid_mask_w / s_track_valid_mask_w.sum(0)

    if huber_delta > 0:
        logging.info(f"Use Huber Loss with delta={huber_delta}")
        huber_loss = torch.nn.HuberLoss(reduction="none", delta=huber_delta)

    loss_list, std_list, fovx_list, fovy_list = [], [], [], []
    flow_loss_list, dep_loss_list, dep_corr_loss_list = [], [], []
    cam_rot_loss_list, cam_trans_loss_list = [], []

    logging.info(f"Start Static BA with {cams.T} frames and {dep_list.shape[1]} points")
    dep_list_mask = torch.ones(dep_list.shape[1]).bool().to(device) # N
    original_dep_list = dep_list.clone()

    with torch.no_grad():
        point_ref = get_world_points(homo_list, dep_list, cams)
        mask = s_track_valid_mask.float()
        count = mask.sum(dim=0)
        mean = (point_ref * mask[:, :, None]).sum(0) / count[:, None].clamp(min=1)
        point_diff_var = ((point_ref - mean[None]).norm(dim=-1) ** 2 * mask).sum(0) / count.clamp(min=1)
        for i in range(10):
            th = point_diff_var.quantile(0.1 * (i + 1))
            print(f"Quantile {0.1 * (i + 1):.1f}: {th}")
        point_threshold = point_diff_var.quantile(1.0)
        dep_list_mask = point_diff_var < point_threshold
        print(f"Keep {dep_list_mask.sum().item()}/{dep_list_mask.shape[0]} depth points")

        s_track_valid_mask = s_track_valid_mask[:, dep_list_mask]
        dep_list = dep_list[:, dep_list_mask]
        homo_list = homo_list[:, dep_list_mask]
        s_track_valid_mask_w = s_track_valid_mask.float()
        s_track_valid_mask_w = s_track_valid_mask_w / s_track_valid_mask_w.sum(0)
        rgb_list = rgb_list[:, dep_list_mask]

    for step in tqdm(range(total_steps)):
        if step == switch_to_ind_step:
            logging.info(
                "Switch to Independent Camera Optimization and Start Scheduling"
            )
            cams.disable_delta()
            optim_list = cams.get_optimizable_list(
                lr_f=lr_cam_f, lr_q=lr_cam_q, lr_t=lr_cam_t
            )
            if lr_dep_s > 0:
                optim_list.append(
                    {"params": [param_scale], "lr": lr_dep_s, "name": "cam_scale"}
                )
            if lr_dep_c > 0:
                optim_list.append(
                    {
                        "params": [param_dep_corr],
                        "lr": lr_dep_c,
                        "name": "dep_correction",
                    }
                )
            optimizer = optimizer_class(optim_list)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                total_steps - switch_to_ind_step,
                eta_min=min(lr_cam_q, lr_cam_t) / 100.0,
            )

        optimizer.zero_grad()

        ########################
        dep_scale = param_scale.abs()
        # dep_scale = dep_scale / dep_scale.mean()

        scaled_depth_list = dep_list * dep_scale[:, None]
        if step >= depth_correction_after_step:
            scaled_depth_list = scaled_depth_list + param_dep_corr[:, None]
            dep_corr_loss = abs(param_dep_corr).mean()
        else:
            dep_corr_loss = torch.zeros_like(scaled_depth_list.mean()).to(device)
        point_ref = get_world_points(homo_list, scaled_depth_list, cams)  # T,N,3

        # transform to each frame!
        if cams.T > max_t_per_step:
            tgt_inds = torch.randperm(cams.T)[:max_t_per_step].to(device)
        else:
            tgt_inds = torch.arange(cams.T).to(device)
        R_cw, t_cw = cams.Rt_cw_list()
        R_cw, t_cw = R_cw[tgt_inds], t_cw[tgt_inds]

        point_ref_to_every_frame = (
            torch.einsum("tij,snj->stni", R_cw, point_ref) + t_cw[None, :, None]
        )  # Src,Tgt,N,3
        uv_src_to_every_frame = cams.project(point_ref_to_every_frame)  # Src,Tgt,N,3
  
        # * robusitify the loss by down weight some curves
        with torch.no_grad():
            projection_singular_mask = abs(point_ref_to_every_frame[..., -1]) < 1e-5
            # no matter where the src is, it should be mapped to every frame with the gt tracking
            cross_time_mask = (
                s_track_valid_mask[:, None] * s_track_valid_mask[None, tgt_inds]
            ).float()
            cross_time_mask = (
                cross_time_mask * (~projection_singular_mask).float()
            )  # Src,Tgt,N

            depth_robust_w = positive_th_gaussian_decay(
                abs(scaled_depth_list), depth_decay_th, depth_decay_sigma
            )

            point_ref_mean = (point_ref * s_track_valid_mask_w[:, :, None]).sum(0)
            point_ref_std = (point_ref - point_ref_mean[None]).norm(dim=-1, p=2)
            point_ref_std_robust_w = (point_ref_std * s_track_valid_mask_w).sum(0)
            point_ref_std_robust_w = positive_th_gaussian_decay(
                point_ref_std_robust_w, std_decay_th, std_decay_sigma
            )

            robust_w = depth_robust_w * point_ref_std_robust_w[None]
            cross_robust_time_mask = robust_w[:, None] * robust_w[None, tgt_inds]
            cross_time_mask = cross_time_mask * cross_robust_time_mask.detach()

        uv_target = homo_list[None, tgt_inds].expand(
            len(uv_src_to_every_frame), -1, -1, -1
        )
        uv_loss_i = (uv_src_to_every_frame - uv_target).norm(dim=-1)

        if huber_delta > 0:
            uv_loss_i = huber_loss(uv_loss_i, torch.zeros_like(uv_loss_i))
        uv_loss = (uv_loss_i * cross_time_mask).sum() / (cross_time_mask.sum() + 1e-6)

        # compute depth loss
        dep_target = scaled_depth_list[None, tgt_inds].expand(
            len(uv_src_to_every_frame), -1, -1
        )
        warped_depth = point_ref_to_every_frame[..., -1]

        dep_consistency_i = 0.5 * abs(
            dep_target / torch.clamp(warped_depth, min=1e-6) - 1
        ) + 0.5 * abs(warped_depth / torch.clamp(dep_target, min=1e-6) - 1)
        # todo: this may be unstable... for fare away depth points!!!
        if huber_delta > 0:
            dep_consistency_i = huber_loss(
                dep_consistency_i, torch.zeros_like(dep_consistency_i)
            )
        dep_loss = (dep_consistency_i * cross_time_mask).sum() / (
            cross_time_mask.sum() + 1e-6
        )

        # camera smoothness reg
        if lambda_cam_smooth_rot > 0 or lambda_cam_smooth_trans > 0:
            cam_trans_loss, cam_rot_loss = cams.smoothness_loss()
        else:
            cam_trans_loss = torch.zeros_like(dep_loss)
            cam_rot_loss = torch.zeros_like(dep_loss)

        loss = (
            lambda_depth * dep_loss
            + lambda_flow * uv_loss
            + lambda_small_correction * dep_corr_loss
            + lambda_cam_smooth_rot * cam_rot_loss
            + lambda_cam_smooth_trans * cam_trans_loss
        )
        
        assert torch.isnan(loss).sum() == 0 and torch.isinf(loss).sum() == 0
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # viz
        with torch.no_grad():
            point_ref_mean = (point_ref * s_track_valid_mask_w[:, :, None]).sum(0)
            std = (point_ref - point_ref_mean[None]).norm(dim=-1, p=2)
            metric_std = (std * s_track_valid_mask_w).sum(0).mean()
            loss_list.append(loss.item())
            dep_corr_loss_list.append(dep_corr_loss.item())
            flow_loss_list.append(uv_loss.item())
            dep_loss_list.append(dep_loss.item())
            cam_rot_loss_list.append(cam_rot_loss.item())
            cam_trans_loss_list.append(cam_trans_loss.item())
            std_list.append(metric_std.item())
            fov = cams.fov
            fovx_list.append(float(fov[0]))
            fovy_list.append(float(fov[1]))
            if step % viz_verbose_n == 0 or step == total_steps - 1:
                logging.info(f"loss={loss.item():.6f}, dep={dep_loss.item():.6f}, loss flow={uv_loss.item():.6f}, fov={cams.fov}")
                logging.info(f"scale max={param_scale.max()} min={param_scale.min()}")

            viz_flag = (
                np.array([step >= r[0] and step <= r[1] for r in viz_denser_range])
                .any()
                .item()
            )
            viz_flag = viz_flag and step % viz_denser_interval == 0
            viz_flag = viz_flag or step % viz_fig_n == 0 or step == total_steps - 1
            if viz_flag:
                # viz the 3D aggregation as well as the pcl in 3D!
                viz_frame = viz_global_ba(
                    point_ref,
                    rgb_list,
                    s_track_valid_mask,
                    cams,
                    error=std,
                    text=f"Step={step}",
                )
                imageio.imsave(
                    osp.join(viz_dir, f"static_scaffold_init_{step:06d}.jpg"),
                    (viz_frame * 255).astype(np.uint8),
                )
            
        if step == 50000:
            # remove the outlier depth points after 1000 steps, which may have large influence on the camera optimization
            with torch.no_grad():
                mask = s_track_valid_mask.float()
                count = mask.sum(dim=0)
                mean = (point_ref * mask[:, :, None]).sum(0) / count[:, None].clamp(min=1)
                point_diff_var = ((point_ref - mean[None]).norm(dim=-1) ** 2 * mask).sum(0) / count.clamp(min=1)
                for i in range(10):
                    th = point_diff_var.quantile(0.1 * (i + 1))
                    logging.info(f"Quantile {0.1 * (i + 1):.1f}: {th}")
                point_threshold = point_diff_var.quantile(0.5)
                dep_list_mask = point_diff_var < point_threshold
                logging.info(f"Keep {dep_list_mask.sum().item()}/{dep_list_mask.shape[0]} depth points")

                s_track_valid_mask = s_track_valid_mask[:, dep_list_mask]
                dep_list = dep_list[:, dep_list_mask]
                homo_list = homo_list[:, dep_list_mask]
                s_track_valid_mask_w = s_track_valid_mask.float()
                s_track_valid_mask_w = s_track_valid_mask_w / s_track_valid_mask_w.sum(0)
                rgb_list = rgb_list[:, dep_list_mask]

    # viz
    make_video_from_pattern(
        osp.join(viz_dir, "static_scaffold_init_*.jpg"),
        osp.join(log_dir, "static_scaffold_init.mp4"),
    )

    if total_steps > 0:
        fig = plt.figure(figsize=(21, 3))
        for plt_i, plt_pack in enumerate(
            [
                ("loss", loss_list),
                ("loss_flow", flow_loss_list),
                ("loss_dep", dep_loss_list),
                ("loss_dep_corr", dep_corr_loss_list),
                ("cam_rot", cam_rot_loss_list),
                ("cam_trans", cam_trans_loss_list),
                ("std", std_list),
                ("fov-x", fovx_list),
                ("fov-y", fovy_list),
            ]
        ):
            plt.subplot(1, 9, plt_i + 1)
            plt.plot(plt_pack[1]), plt.title(
                plt_pack[0] + f" End={plt_pack[1][-1]:.6f}"
            )
            if plt_pack[0].startswith("loss"):
                plt.yscale("log")
        plt.tight_layout()
        plt.savefig(osp.join(log_dir, f"static_scaffold_init.jpg"))
        plt.close()

    world_point = get_world_points(homo_list, dep_list * param_scale[:, None] + param_dep_corr[:, None], cams)
    var = world_point.var(dim=0)
    logging.info(f"Mean variance of world points: {var.mean().item()}")

    # update the depth scale
    dep_scale = param_scale.abs()
    # dep_scale = dep_scale / dep_scale.mean()
    dep_corr = param_dep_corr.detach().clone()

    logging.info(f"Update the S2D depth scale with depth scale range [{dep_scale.min().item():.4f}, {dep_scale.max().item():.4f}]")
    logging.info(f"Update the S2D depth bias with depth bias range [{dep_corr.min().item():.4f}, {dep_corr.max().item():.4f}]")
    s2d.rescale_depth(dep_scale, dep_corr)
    torch.save(cams.state_dict(), osp.join(log_dir, "bundle_cams.pth"))
    torch.save(
        {
            # sol
            "dep_scale": dep_scale,  # ! important, to later rescale the depth
            "dep_correction": dep_corr,
            "s_track": s_track,
            "s_track_mask": s_track_valid_mask,
            "s_optimal": s_optimal,
        },
        osp.join(log_dir, "bundle.pth"),
    )
    # also save a reconstructed point cloud
    if save_more_flag:
        np.savetxt(
            osp.join(log_dir, "static_scaffold_pcl_unmerged.xyz"),
            torch.cat([point_ref, rgb_list], -1).reshape(-1, 6).detach().cpu().numpy(),
            fmt="%.6f",
        )

    viz_depth_list(s2d.dep.cpu().numpy(), osp.join(log_dir, "ba_depth_viz.mp4"))

    return cams, s_track, s_track_valid_mask, param_dep_corr.detach().clone()


@torch.no_grad()
def deform_depth_map(
    depth_list,
    mask_list,
    cams: MonocularCameras,
    track_uv_list,
    track_mask_list,
    dep_correction,
    K=16,
    rbf_factor=0.333,
    viz_fn=None,
):
    # depth_list: T,H,W; track_uv_list: T,N,2; track_mask_list: T,N; src_buffer: T,N,C
    logging.info("Deforming depth")
    T = len(track_mask_list)
    H, W = depth_list[0].shape
    assert depth_list.shape == mask_list.shape
    assert T == len(track_uv_list) == len(dep_correction)
    assert T == len(depth_list) == len(mask_list)
    homo_map = torch.from_numpy(get_homo_coordinate_map(H, W)).to(depth_list[0])

    dep_corr_map_list, dep_new_map_list = [], []
    for tid in tqdm(range(T)):
        mask2d = mask_list[tid]
        scf_mask = track_mask_list[tid]
        dep_map = depth_list[tid]
        scf_uv = track_uv_list[tid][scf_mask]
        scf_int_uv, scf_inside_mask = round_int_coordinates(scf_uv, H, W)
        if not scf_inside_mask.all():
            logging.warning(
                f"Warning, {(~scf_inside_mask).sum()} invalid uv in t={tid}! may due to round accuracy"
            )

        scf_dep = query_image_buffer_by_pix_int_coord(depth_list[tid], scf_int_uv)
        scf_homo = query_image_buffer_by_pix_int_coord(homo_map, scf_int_uv)
        # this pts is used to distribute the carrying interp_src in 3D cam frame
        scf_cam_pts = cams.backproject(scf_homo, scf_dep)
        dst_cam_pts = cams.backproject(homo_map[mask2d], dep_map[mask2d])
        scf_buffer = dep_correction[tid][scf_mask]

        interp_dep_corr = spatial_interpolation(
            src_xyz=scf_cam_pts,
            src_buffer=scf_buffer[:, None],
            query_xyz=dst_cam_pts,
            K=K,
            rbf_sigma_factor=rbf_factor,
        )

        # viz
        dep_corr_map = torch.zeros_like(dep_map)
        dep_corr_map[mask2d] = interp_dep_corr.squeeze(-1)
        scf_corr_interp = query_image_buffer_by_pix_int_coord(dep_corr_map, scf_int_uv)
        # check_interp_error = (
        #     abs(scf_corr_interp - scf_buffer.squeeze(-1)).median()
        #     / abs(scf_buffer).median()
        # )s
        dep_corr_map_list.append(dep_corr_map.detach())
        dep_new_map_list.append((dep_corr_map + dep_map).detach())

    if viz_fn is not None:
        viz_corr_list, viz_dep_list = [], []
        for tid in tqdm(range(T)):
            viz_corr = dep_corr_map_list[tid]
            viz_dep = dep_new_map_list[tid]
            viz_corr_radius = abs(viz_corr).max()
            viz_corr = (viz_corr / viz_corr_radius) + 0.5
            viz_dep = (viz_dep - viz_dep.min()) / (viz_dep.max() - viz_dep.min())
            viz_corr = cm.viridis(viz_corr.cpu().numpy())
            viz_dep = cm.viridis(viz_dep.cpu().numpy())
            viz_corr = (viz_corr * 255).astype(np.uint8)
            viz_dep = (viz_dep * 255).astype(np.uint8)
            viz_corr_list.append(viz_corr)
            viz_dep_list.append(viz_dep)
        imageio.mimsave(viz_fn.replace(".mp4", "_corr.mp4"), viz_corr_list)
        imageio.mimsave(viz_fn.replace(".mp4", "_dep_corr.mp4"), viz_dep_list)

    dep_new_map_list = torch.stack(dep_new_map_list, 0)
    return dep_new_map_list


def spatial_interpolation(src_xyz, src_buffer, query_xyz, K=16, rbf_sigma_factor=0.333):
    # src_xyz: M,3 src_buffer: M,C query_xyz: N,3
    # build RBG on each src and smoothly interpolate the buffer to query
    # first construct src_xyz nn graph
    _dist_sq_to_nn, _, _ = knn_points(src_xyz[None], src_xyz[None], K=2)
    dist_to_nn = torch.sqrt(torch.clamp(_dist_sq_to_nn[0, :, 1:], min=1e-8)).squeeze(-1)
    rbf_sigma = dist_to_nn * rbf_sigma_factor  # M
    # find the nearest K neighbors for each query point to the src
    dist_sq, ind, _ = knn_points(query_xyz[None], src_xyz[None], K=K)
    dist_sq, ind = dist_sq[0], ind[0]

    w = torch.exp(-dist_sq / (2.0 * (rbf_sigma[ind] ** 2)))  # N,K
    w = w / torch.clamp(w.sum(-1, keepdim=True), min=1e-8)

    value = src_buffer[ind]  # N,K,C
    ret = torch.einsum("nk, nkc->nc", w, value)
    return ret


def get_homo_coordinate_map(H, W):
    # the grid take the short side has (-1,+1)
    if H > W:
        u_range = [-1.0, 1.0]
        v_range = [-float(H) / W, float(H) / W]
    else:  # H<=W
        u_range = [-float(W) / H, float(W) / H]
        v_range = [-1.0, 1.0]
    # make uv coordinate
    u, v = np.meshgrid(np.linspace(*u_range, W), np.linspace(*v_range, H))
    uv = np.stack([u, v], axis=-1)  # H,W,2
    return uv


def round_int_coordinates(coord, H, W):
    ret = coord.round().long()
    valid_mask = (
        (ret[..., 0] >= 0) & (ret[..., 0] < W) & (ret[..., 1] >= 0) & (ret[..., 1] < H)
    )
    ret[..., 0] = torch.clamp(ret[..., 0], 0, W - 1)
    ret[..., 1] = torch.clamp(ret[..., 1], 0, H - 1)
    return ret, valid_mask


def query_image_buffer_by_pix_int_coord(buffer, pixel_int_coordinate):
    assert pixel_int_coordinate.ndim == 2 and pixel_int_coordinate.shape[-1] == 2
    assert (pixel_int_coordinate[..., 0] >= 0).all()
    assert (pixel_int_coordinate[..., 0] < buffer.shape[1]).all()
    assert (pixel_int_coordinate[..., 1] >= 0).all()
    assert (pixel_int_coordinate[..., 1] < buffer.shape[0]).all()
    # u is the col, v is the row
    col_id, row_id = pixel_int_coordinate[:, 0], pixel_int_coordinate[:, 1]
    H, W = buffer.shape[:2]
    index = col_id + row_id * W
    ret = buffer.reshape(H * W, *buffer.shape[2:])[index]
    if isinstance(ret, np.ndarray):
        ret = ret.copy()
    return ret


def prepare_track_homo_dep_rgb_buffers(s2d, track, track_mask, t_list):
    # track: T,N,2, track_mask: T,N
    device = track.device
    homo_list, ori_dep_list, rgb_list = [], [], []
    for ind, tid in enumerate(t_list):
        _uv = track[ind]
        _int_uv, _inside_mask = round_int_coordinates(_uv, s2d.H, s2d.W)
        _dep = query_image_buffer_by_pix_int_coord(
            s2d.dep[tid].clone().to(device), _int_uv
        )
        _homo = query_image_buffer_by_pix_int_coord(
            s2d.homo_map.clone().to(device), _int_uv
        )
        ori_dep_list.append(_dep.to(device))
        homo_list.append(_homo.to(device))
        # for viz purpose
        _rgb = query_image_buffer_by_pix_int_coord(
            s2d.rgb[tid].clone().to(device), _int_uv
        )
        rgb_list.append(_rgb.to(device))
    rgb_list = torch.stack(rgb_list, 0)
    ori_dep_list = torch.stack(ori_dep_list, 0)
    homo_list = torch.stack(homo_list)
    ori_dep_list[~track_mask] = -1
    homo_list[~track_mask] = 0.0
    return homo_list, ori_dep_list, rgb_list


def compute_track_epipolar_errors(
    track_uv, track_mask, cams, pair_list=None, normalize_by_f=True, device=None
):
    """
    Compute per-track Sampson epipolar error across a set of frame pairs.

    Args:
        track_uv: Tensor (T, N, 2) pixel coordinates (u,v) in image pixel units.
        track_mask: Bool Tensor (T, N) visibility mask.
        cams: MonocularCameras instance (provides intrinsics and poses).
        pair_list: list of (i,j) pairs to evaluate. If None, use all pairs (i,j) with i < j.
        normalize_by_f: if True, convert pixel coords to normalized camera coords consistent with cams.project/backproject.
        device: torch device to use (defaults to track_uv.device)

    Returns:
        per_track_mean_error: Tensor (N,) mean Sampson error across all evaluated pairs for each track (NaN where no valid observations).
        per_pair_errors: dict mapping (i,j) -> Tensor (N,) with per-track error for that pair (invalid entries set to +inf).
    """
    if device is None:
        device = track_uv.device
    T, N, _ = track_uv.shape
    if pair_list is None:
        jump = max(T // 400, 1)
        pair_list = [(i, j) for i in range(0, T) for j in range(i + 1, T, jump)]

    # intrinsic K as torch tensor (use image default size from cams)
    K = cams.K(int(cams.default_H.item()), int(cams.default_W.item())).to(device)
    K_inv = torch.linalg.inv(K)

    # convert pixel (u,v) -> normalized homogeneous coordinates x = K^{-1} [u,v,1]^T
    # track_uv is expected in pixel coordinate (col=u, row=v)
    ones = torch.ones((T, N, 1), device=device)
    uv_h = torch.cat([track_uv, ones], dim=-1).to(device)  # T,N,3

    # convert to normalized camera coords: x = K^{-1} @ uv_h
    uv_h_flat = uv_h.reshape(-1, 3).T  # 3, T*N
    x_flat = (K_inv.to(device) @ uv_h_flat).T.reshape(T, N, 3)

    per_pair_errors = {}
    all_errors = []
    for (i, j) in pair_list:
        # essential matrix E = [t]_x R, where R,t map from i->j in camera frame
        R_ij, t_ij = cams.Rt_ij(i, j)
        R_ij = R_ij.to(device)
        t_ij = t_ij.to(device)
        t_x = torch.tensor(
            [[0.0, -t_ij[2], t_ij[1]], [t_ij[2], 0.0, -t_ij[0]], [-t_ij[1], t_ij[0], 0.0]]
        ).to(device)
        E = t_x @ R_ij

        # points in normalized coords (3-vector). Use only visible points in both frames
        x_i = x_flat[i]  # N,3
        x_j = x_flat[j]
        vis = (track_mask[i] & track_mask[j]).to(device)

        # homogeneous coordinates
        xi = x_i
        xj = x_j

        # epipolar line in j: l_j = E x_i  (3-vector)
        l_j = (E @ xi.T).T  # N,3
        # epipolar line in i: l_i = E^T x_j
        l_i = (E.T @ xj.T).T

        # Sampson distance: (x_j^T E x_i)^2 / ( (E x_i)_0^2 + (E x_i)_1^2 + (E^T x_j)_0^2 + (E^T x_j)_1^2 )
        numer = (xj * (E @ xi.T).T).sum(dim=-1)  # N
        denom = l_j[:, 0].pow(2) + l_j[:, 1].pow(2) + l_i[:, 0].pow(2) + l_i[:, 1].pow(2)
        # avoid division by zero
        denom = denom + 1e-12
        sampson_sq = (numer.pow(2) / denom).detach()

        # set invalid (non-visible) to +inf so they won't be counted
        sampson_sq[~vis] = float("inf")
        per_pair_errors[(i, j)] = sampson_sq
        all_errors.append(sampson_sq)

    # stack and compute per-track mean (ignore inf entries)
    all_errors_stack = torch.stack(all_errors, dim=0)  # P, N
    # mask valid
    valid_mask = torch.isfinite(all_errors_stack)
    sum_err = torch.where(valid_mask, all_errors_stack, torch.zeros_like(all_errors_stack)).sum(dim=0)
    cnt = valid_mask.sum(dim=0).float()
    per_track_mean = torch.full((N,), float("nan"), device=device)
    nonzero = cnt > 0
    per_track_mean[nonzero] = (sum_err[nonzero] / cnt[nonzero]).sqrt()  # return RMS Sampson (not squared)

    return per_track_mean, per_pair_errors


def query_buffers_by_track(image_buffer, track, track_mask, default_value=0.0):
    # image_buffer: T,H,W,C; track: T,N,2, track_mask: T,N
    assert image_buffer.ndim == 4 and track.ndim == 3 and track_mask.ndim == 2
    assert len(image_buffer) == len(track) == len(track_mask)
    T, H, W, C = image_buffer.shape
    N = track.shape[1]
    ret_buffer = torch.ones(T, N, C).to(image_buffer) * default_value

    for i in range(T):
        _uv = track[i][..., :2]
        _int_uv, _inside_mask = round_int_coordinates(_uv, H, W)
        _value = query_image_buffer_by_pix_int_coord(image_buffer[i].clone(), _int_uv)
        valid_mask = track_mask[i] & _inside_mask
        # for outside, put default value
        _value[~valid_mask] = default_value
        ret_buffer[i] = _value
    return ret_buffer


def get_world_points(homo_list, dep_list, cams, cam_t_list=None, mask=None):
    T, M = dep_list.shape
    if cam_t_list is None:
        cam_t_list = torch.arange(T).to(homo_list.device)
    point_cam = cams.backproject(homo_list.reshape(-1, 2), dep_list.reshape(-1))
    point_cam = point_cam.reshape(T, M, 3)
    R_wc, t_wc = cams.Rt_wc_list()
    R_wc, t_wc = R_wc[cam_t_list], t_wc[cam_t_list]
    point_world = torch.einsum("tij,tmj->tmi", R_wc, point_cam) + t_wc[:, None]

    if mask is not None:
        point_world = point_world[:, mask]

    return point_world


def fovdeg2focal(fov_deg):
    focal = 1.0 / np.tan(np.deg2rad(fov_deg) / 2.0)
    return focal


def triangulate_3Dpoints_from_tracks(
    track_uv, track_mask, cams, min_obs=2, device=None, eps=1e-12, use_normalized_reproj=True
):
    """
    Triangulate 3D points from 2D tracks and known camera poses.

    Args:
        track_uv: Tensor (T, N, 2) pixel coordinates (u,v) in image pixel units (col,row).
        track_mask: Bool Tensor (T, N) visibility mask.
        cams: MonocularCameras instance (provides intrinsics and poses).
        min_obs: minimum number of observations required to triangulate a track.
        device: torch device or None to use track_uv.device.

    Returns:
        points_w: Tensor (N, 3) world 3D positions (NaN for tracks that cannot be triangulated).
    reproj_err: Tensor (N,) mean reprojection error (normalized image units by default; pixel units if use_normalized_reproj=False).
        obs_count: Tensor (N,) number of valid observations per track.
    """
    # Vectorized implementation - accumulate AtA and Atb per track using identity:
    # For a ray direction d and camera center C, the skew-matrix S(d) satisfies
    # S^T S = (d^T d) I - d d^T. The linear system contributions can be accumulated via
    # AtA += S^T S, Atb += S^T S @ C
    if device is None:
        device = track_uv.device
    T, N, _ = track_uv.shape

    # intrinsics
    H = int(cams.default_H.item())
    W = int(cams.default_W.item())
    K = cams.K(H, W).to(device)
    K_inv = torch.linalg.inv(K)

    # poses
    R_wc, t_wc = cams.Rt_wc_list()
    R_wc = R_wc.to(device)
    t_wc = t_wc.to(device)

    # prepare normalized camera rays in camera coords: x = K^{-1} [u,v,1]^T
    ones = torch.ones((T, N, 1), device=device)
    uv_h = torch.cat([track_uv.to(device), ones], dim=-1)  # T,N,3
    uv_h_flat = uv_h.reshape(-1, 3).T  # 3, T*N
    x_cam = (K_inv.to(device) @ uv_h_flat).T.reshape(T, N, 3)  # T,N,3

    # transform rays to world frame d_world = R_wc @ d_cam
    d_world = torch.einsum("tij,tnj->tni", R_wc, x_cam) # T,N,3

    # precompute components
    d_norm2 = (d_world ** 2).sum(-1)  # T,N
    # outer product d d^T : T,N,3,3
    outer = d_world[..., :, None] * d_world[..., None, :]
    I3 = torch.eye(3, device=device).view(1, 1, 3, 3)

    # AtA contribution per ray: d_norm2 * I - outer
    AtA_contrib = d_norm2[..., None, None] * I3 - outer  # T,N,3,3

    # apply visibility mask
    vis = track_mask.to(device).float()  # T,N
    vis4 = vis[..., None, None]
    AtA_contrib = AtA_contrib * vis4

    # sum over frames -> AtA per track (N,3,3)
    AtA = AtA_contrib.sum(dim=0)

    # Atb: sum_t AtA_contrib[t] @ C_t  with C_t = t_wc[t] (camera center in world)
    C_expand = t_wc[:, None, :].expand(-1, N, -1)  # T,N,3
    Atb_contrib = torch.einsum("tnij, tnj->tni", AtA_contrib, C_expand)  # T,N,3
    Atb = Atb_contrib.sum(dim=0)  # N,3

    # count observations per track
    obs_count = track_mask.to(device).sum(dim=0).long()

    # solve AtA X = Atb for each track (batched)
    eps_eye = eps * torch.eye(3, device=device).unsqueeze(0).expand(N, -1, -1)
    AtA_safe = AtA + eps_eye
    # handle tracks with too few observations by zeroing their AtA to avoid solve errors
    valid_tracks = obs_count >= min_obs
    # set invalid AtA to identity to avoid singularities (we'll mask outputs later)
    AtA_safe[~valid_tracks] = torch.eye(3, device=device)
    Atb_masked = Atb.clone()
    Atb_masked[~valid_tracks] = 0.0

    # solve batched 3x3 systems
    try:
        X = torch.linalg.solve(AtA_safe, Atb_masked.unsqueeze(-1)).squeeze(-1)  # N,3
    except RuntimeError:
        # fallback to pinv for stability
        X = torch.stack([torch.linalg.pinv(AtA_safe[i]) @ Atb_masked[i] for i in range(N)], dim=0)

    points_w = torch.full((N, 3), float("nan"), device=device)
    points_w[valid_tracks] = X[valid_tracks]

    # reprojection error: project points into each camera and compute error where visible
    # p_c(t,n,3) = R_cw[t] @ X[n] + t_cw[t]
    # compute for all t and n: use einsum
    # X: N,3 -> expand to T,N,3 via einsum
    R_cw, t_cw = cams.Rt_cw_list()
    p_c = torch.einsum("tij, nj->tni", R_cw, points_w) + t_cw[:, None, :]
    # predicted normalized coordinates (x_pred = X_cam[:2]/Z)
    x_pred = p_c[..., :2] / (p_c[..., 2:3] + 1e-12)  # T,N,2

    if use_normalized_reproj:
        # observed normalized coordinates were computed earlier as x_cam = K^{-1} [u,v,1]
        x_obs = x_cam[..., :2]
        diff_sq = ((x_pred - x_obs) ** 2).sum(dim=-1)  # T,N (normalized units)
    else:
        # project to pixels and compare in pixel units
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]
        u_pred = fx * x_pred[..., 0] + cx
        v_pred = fy * x_pred[..., 1] + cy
        uv_pred = torch.stack([u_pred, v_pred], dim=-1)  # T,N,2
        uv_obs = track_uv.to(device)
        diff_sq = ((uv_pred - uv_obs) ** 2).sum(dim=-1)  # T,N
    diff_sq = diff_sq * vis  # zero out non-visible
    # sum over frames and divide by obs_count (avoid division by zero)h 
    sum_sq = diff_sq.sum(dim=0)
    obs_count_f = obs_count.float().clamp(min=1.0)
    reproj_err = torch.full((N,), float("nan"), device=device)
    reproj_err[valid_tracks] = (sum_sq[valid_tracks] / obs_count_f[valid_tracks]).sqrt()

    return points_w, reproj_err, obs_count

def track2undistroed_homo(track, H, W):
    # the short side is -1,1, the long side may exceed
    H, W = float(H), float(W)
    L = min(H, W)
    u, v = track[..., 0], track[..., 1]
    u = 2.0 * u / L - W / L
    v = 2.0 * v / L - H / L
    uv = torch.stack([u, v], -1)
    return uv


def viz_ba_point(viz_video_rgb, s_track, s_track_valid_mask):
    # todo: color the points by importance robust weight
    viz_frames = []
    for t in tqdm(range(len(viz_video_rgb))):
        frame_rgb = viz_video_rgb[t].copy()
        uv = s_track[t].cpu().numpy()
        _viz_valid_mask = s_track_valid_mask[t].cpu().numpy()
        for i in range(len(uv)):
            if _viz_valid_mask[i]:
                u, v = int(uv[i, 0]), int(uv[i, 1])
                if 0 <= u < frame_rgb.shape[1] and 0 <= v < frame_rgb.shape[0]:
                    _color = np.array(cm.hsv(float(i) / len(uv)))[:3] * 255
                    # put a circel with color
                    cv2.circle(frame_rgb, (u, v), 3, _color, 1)
        # put total valid num as text valid_mask.sum()
        cv2.putText(
            frame_rgb,
            f"Visible BA points: {_viz_valid_mask.sum()}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        viz_frames.append(frame_rgb)
    return viz_frames

def viz_depth_list(depths, save_fn, viz_quantile=3):

    dep_list = np.stack(depths, axis=0)

    # use disparity to viz, not depth
    dep_valid_mask = dep_list > 1e-6
    # use robust min and max to visualize
    dep_max = np.percentile(dep_list[dep_valid_mask], 100 - viz_quantile)
    dep_min = np.percentile(dep_list[dep_valid_mask], viz_quantile)
    dep_list = np.clip(dep_list, dep_min, dep_max)
    dep_list = (dep_list - dep_min) / (dep_max - dep_min)
    dep_list[~dep_valid_mask] = 0
    # h_dep = dep_list.reshape(-1)
    # plt.hist(h_dep, bins=300), plt.title("Depth Histogram")
    # plt.savefig(save_fn.replace(".mp4", ".jpg"))
    # plt.close()
    viz_list = []
    for dep in dep_list:
        viz = cm.viridis(dep)[:, :, :3]
        viz_list.append((viz * 255).astype(np.uint8))
    imageio.mimsave(save_fn, viz_list)
    return