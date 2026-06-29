import os
import torch
import os.path as osp
import logging
from omegaconf import OmegaConf

from lib_prior.prior_loading import Saved2D
from lib_moca.moca import moca_solve
from lib_moca.camera import MonocularCameras

from mosca_evaluate import test_tum_cam, test_sintel_cam

from data_utils.iphone_helpers import load_iphone_gt_poses
from data_utils.nvidia_helpers import load_nvidia_gt_pose, get_nvidia_dummy_test

from recon_utils import (
    seed_everything,
    setup_recon_ws,
    auto_get_depth_dir_tap_mode,
    SEED,
)

from mosca_viz import viz_world_points_in_test_cam_frame
from viz_utils import viz_world_points_in_test_cam_frame_with_mask
import cv2

def load_gt_cam(ws, fit_cfg):
    mode = getattr(fit_cfg, "mode", "iphone")
    logging.info(f"Loading gt camera poses in mode {mode}")
    if mode == "iphone":
        return load_iphone_gt_poses(ws, t_subsample=getattr(fit_cfg, "t_subsample", 1))
    elif mode == "nvidia":
        (gt_training_cam_T_wi, gt_training_fov, gt_training_cxcy_ratio) = (
            load_nvidia_gt_pose(osp.join(ws, "poses_bounds.npy"))
        )
        (
            gt_testing_cam_T_wi_list,
            gt_testing_tids_list,
            gt_testing_fns_list,
            gt_testing_fov_list,
            gt_testing_cxcy_ratio_list,
        ) = get_nvidia_dummy_test(gt_training_cam_T_wi, gt_training_fov)
        return (
            gt_training_cam_T_wi,
            gt_testing_cam_T_wi_list,
            gt_testing_tids_list,
            gt_testing_fns_list,
            gt_training_fov,
            gt_testing_fov_list,
            gt_training_cxcy_ratio,
            gt_testing_cxcy_ratio_list,
        )
    else:
        raise RuntimeError(f"Unknown mode: {mode}")
    return


def static_reconstruct(ws, log_path, fit_cfg):
    seed_everything(SEED)
    DEPTH_DIR, TAP_MODE = auto_get_depth_dir_tap_mode(ws, fit_cfg)
    print(f"Using depth dir: {DEPTH_DIR}, tap mode: {TAP_MODE}")
    DEPTH_BOUNDARY_TH = getattr(fit_cfg, "depth_boundary_th", 1.0)
    INIT_GT_CAMERA_FLAG = getattr(fit_cfg, "init_gt_camera", False)
    DEP_MEDIAN = getattr(fit_cfg, "dep_median", 1.0)

    EPI_TH = getattr(fit_cfg, "ba_epi_th", getattr(fit_cfg, "epi_th", 1e-3))
    logging.info(f"Static BA with EPI_TH={EPI_TH}")
    print(f"Static BA with EPI_TH={EPI_TH}")
    device = torch.device("cuda:0")

    s2d: Saved2D = (
        Saved2D(ws)
        .load_mask()
        # .load_epi()
        .load_dep(DEPTH_DIR, DEPTH_BOUNDARY_TH)
        .normalize_depth(median_depth=DEP_MEDIAN)
        # .recompute_dep_mask(depth_boundary_th=DEPTH_BOUNDARY_TH)
        .load_track(
            f"*uniform*{TAP_MODE}",
            min_valid_cnt=getattr(fit_cfg, "tap_loading_min_valid_cnt", 4),
        )
        .load_vos()
    )

    if INIT_GT_CAMERA_FLAG:
        # if start form gt camera, load gt camera here
        logging.info(f"Initializing from GT camera")
        (
            gt_training_cam_T_wi,
            gt_testing_cam_T_wi_list,
            gt_testing_tids_list,
            gt_testing_fns_list,
            gt_training_fov,
            gt_testing_fov_list,
            gt_training_cxcy_ratio,
            gt_testing_cxcy_ratio_list,
        ) = load_gt_cam(ws, fit_cfg)
        gt_fovdeg = float(gt_training_fov)
        cxcy_ratio = gt_training_cxcy_ratio[0]  # gt camera center
        if getattr(fit_cfg, "init_gt_camera_focal_only", False):
            logging.info(f"Only init focal length")
            cams = MonocularCameras(
                n_time_steps=s2d.T,
                default_H=s2d.H,
                default_W=s2d.W,
                fxfycxcy=[gt_fovdeg, gt_fovdeg] + cxcy_ratio,
                delta_flag=True,
                init_camera_pose=torch.eye(4)
                .to(gt_training_cam_T_wi)[None]
                .expand(len(gt_training_cam_T_wi) - 1, -1, -1),
                iso_focal=getattr(fit_cfg, "iso_focal", False),
            )
        else:
            cams = MonocularCameras(
                n_time_steps=s2d.T,
                default_H=s2d.H,
                default_W=s2d.W,
                fxfycxcy=[gt_fovdeg, gt_fovdeg] + cxcy_ratio,
                delta_flag=False,
                init_camera_pose=gt_training_cam_T_wi,
                iso_focal=getattr(fit_cfg, "iso_focal", False),
            )
    else:
        cams = None

    # import imageio.v2 as imageio
    # import numpy as np
    # gt_depth_dir = osp.join(ws, "sensor_depth_gt")
    # gt_depth_list = []
    # for img_fn in sorted(os.listdir(gt_depth_dir)):
    #     if img_fn.endswith(".npz"):
    #         gt_depth_list.append(
    #             np.load(osp.join(gt_depth_dir, img_fn))["dep"].astype(np.float32)
    #         )

    # gt_depth_list = np.stack(gt_depth_list, axis=0)
    # gt_depth_list = torch.from_numpy(gt_depth_list).to(device)

    # dep_mask = s2d.dep_mask.clone().to(device)
    # dep_mask = dep_mask & (gt_depth_list > 0)

    # error = 0.0

    # scale_list = []
    # bias_list = []

    # for pred_dep, gt_dep, mask in zip(s2d.dep.clone().to(device), gt_depth_list, dep_mask):
    #     p = pred_dep[mask].reshape(-1)
    #     g = gt_dep[mask].reshape(-1)

    #     X = torch.stack([p, torch.ones_like(p)], dim=-1)

    #     sol = torch.linalg.lstsq(X, g).solution

    #     scale, bias = sol[0], sol[1]
    #     scale_list.append(scale.item())
    #     bias_list.append(bias.item())

    #     affined_pred = p * scale + bias

    #     error += (torch.abs(affined_pred - g) / g).mean()

    # error = error / len(s2d.dep)
    # print(f"Initial mean absolute depth error: {error.item() * 100:.2f}%")
    # print(f"Depth scale range: [{min(scale_list):.4f}, {max(scale_list):.4f}], bias range: [{min(bias_list):.4f}, {max(bias_list):.4f}]")

    # scale_list = torch.tensor(scale_list).to(device)
    # bias_list = torch.tensor(bias_list).to(device)
    # s2d.rescale_depth(scale_list, bias_list)

    logging.info("*" * 20 + "MoCa BA" + "*" * 20)
    cams, s2d, _ = moca_solve(
        ws=log_path,
        s2d=s2d,
        device=device,
        epi_th=EPI_TH,
        ba_total_steps=getattr(fit_cfg, "ba_total_steps", 2000),
        ba_switch_to_ind_step=getattr(fit_cfg, "ba_switch_to_ind_step", 500),
        ba_depth_correction_after_step=getattr(
            fit_cfg, "ba_depth_correction_after_step", 500
        ),
        ba_max_frames_per_step=32,
        static_id_mode="mask",  # "raft", "track", "mask"
        # * robust setting
        robust_depth_decay_th=getattr(fit_cfg, "robust_depth_decay_th", 2.0),
        robust_depth_decay_sigma=getattr(fit_cfg, "robust_depth_decay_sigma", 1.0),
        robust_std_decay_th=getattr(fit_cfg, "robust_std_decay_th", 0.2),
        robust_std_decay_sigma=getattr(fit_cfg, "robust_std_decay_sigma", 0.2),
        #
        gt_cam=cams,
        iso_focal=getattr(fit_cfg, "iso_focal", False),
        rescale_gt_cam_transl=getattr(fit_cfg, "rescale_gt_cam_transl", False),
        ba_lr_cam_f=getattr(fit_cfg, "ba_lr_cam_f", 0.0003),
        ba_lr_dep_c=getattr(fit_cfg, "ba_lr_dep_c", 0.001),
        ba_lr_dep_s=getattr(fit_cfg, "ba_lr_dep_s", 0.001),
        ba_lr_cam_q=getattr(fit_cfg, "ba_lr_cam_q", 0.0003),
        ba_lr_cam_t=getattr(fit_cfg, "ba_lr_cam_t", 0.0003),
        #
        ba_lambda_flow=getattr(fit_cfg, "ba_lambda_flow", 1.0),
        ba_lambda_depth=getattr(fit_cfg, "ba_lambda_depth", 0.1),
        ba_lambda_small_correction=getattr(fit_cfg, "ba_lambda_small_correction", 0.03),
        ba_lambda_cam_smooth_trans=getattr(fit_cfg, "ba_lambda_cam_smooth_trans", 0.0),
        ba_lambda_cam_smooth_rot=getattr(fit_cfg, "ba_lambda_cam_smooth_rot", 0.0),
        #
        depth_filter_th=getattr(fit_cfg, "ba_depth_remove_th", -1.0),
        init_cam_with_optimal_fov_results=getattr(
            fit_cfg, "init_cam_with_optimal_fov_results", True
        ),
        # fov
        fov_search_fallback=getattr(fit_cfg, "ba_fov_search_fallback", 53.0),
        fov_search_N=getattr(fit_cfg, "ba_fov_search_N", 100),
        fov_search_start=getattr(fit_cfg, "ba_fov_search_start", 30.0),
        fov_search_end=getattr(fit_cfg, "ba_fov_search_end", 90.0),
        viz_valid_ba_points=getattr(fit_cfg, "ba_viz_valid_points", False),
    )  # ! S2D is changed becuase the depth is re-scaled

    datamode = getattr(fit_cfg, "mode", "iphone")
    if datamode == "sintel":
        test_func = test_sintel_cam
    elif datamode == "tum":
        test_func = test_tum_cam
    else:
        test_func = None
    if test_func is not None:
        test_func(
            cam_pth_fn=osp.join(log_path, "bundle", "bundle_cams.pth"),
            ws=ws,
            save_path=osp.join(log_path, "cam_metrics_ba.txt"),
        )

    try:
        import imageio.v2 as imageio
        import numpy as np
        gt_depth_dir = osp.join(ws, "sensor_depth_gt")
        gt_depth_list = []
        for img_fn in sorted(os.listdir(gt_depth_dir)):
            if img_fn.endswith(".npz"):
                gt_depth_list.append(
                    np.load(osp.join(gt_depth_dir, img_fn))["dep"].astype(np.float32)
                )

        gt_depth_list = np.stack(gt_depth_list, axis=0)
        gt_depth_list = torch.from_numpy(gt_depth_list).to(device)

        bundle = torch.load(osp.join(log_path, "bundle", "bundle.pth"))
        s_optimal = bundle["s_optimal"]
        gt_depth_list = gt_depth_list * s_optimal
        print("Load s optimal from bundle:", s_optimal)
        scale_list = scale_list * s_optimal
        bias_list = bias_list * s_optimal
        print(f"After scaling with s_optimal, depth scale range: [{min(scale_list):.4f}, {max(scale_list):.4f}], bias range: [{min(bias_list):.4f}, {max(bias_list):.4f}]")

        dep_mask = s2d.dep_mask
        dep_mask = dep_mask & (gt_depth_list > 0)

        print("Evaluating depth error...")
        pred_depth_list = s2d.dep
        error = (torch.abs(pred_depth_list - gt_depth_list) / gt_depth_list)
        print(f"Mean absolute depth error: {error[dep_mask].mean().item() * 100:.2f}%")

        error_th = 0.1
        error_mask = error > error_th
        error_mask = error_mask.cpu().numpy().astype(np.uint8) * 255
        imageio.mimsave(
            osp.join(log_path, "bundle", f"depth_error_mask_error_th={error_th}.mp4"),
            [error_mask[i] for i in range(error_mask.shape[0])],
            fps=20,
        )

    except Exception as e:
        print(f"Error occurred while saving depth error mask: {e}")
        pass

    bundle = torch.load(osp.join(log_path, "bundle", "bundle.pth"))
    s_optimal = bundle["s_optimal"]

    for test_i in range(len(gt_testing_cam_T_wi_list)):
        # viz_list = viz_world_points_in_test_cam_frame(
        #     cams,
        #     gt_testing_cam_T_wi_list[test_i],
        #     s2d,
        #     s_optimal,
        # )
        gt_dir = osp.join(ws, "test_images")
        gt_video_fn = [f for f in os.listdir(gt_dir) if f.startswith(f"{test_i+1}_")]
        test_rgb = [imageio.imread(osp.join(gt_dir, f)) for f in sorted(gt_video_fn)]
        viz_list = viz_world_points_in_test_cam_frame_with_mask(
            cams,
            gt_testing_cam_T_wi_list[test_i],
            s2d,
            s_optimal,
            static_mask=~s2d.mask
        )
        
        
        for i in range(len(viz_list)):
            error = np.abs(viz_list[i].astype(np.float32) - test_rgb[i].astype(np.float32))
            error /= error.max() + 1e-8
            error = (error * 255.0).astype(np.uint8)
            error = cv2.applyColorMap(error, cv2.COLORMAP_JET)[:, :, ::-1]
            viz_list[i] = np.concatenate([test_rgb[i], viz_list[i], error], axis=1)
        
        imageio.mimsave(
            osp.join(log_path, "bundle", f"viz_world_points_in_test_cam_frame_{test_i}.mp4"),
            [viz_list[i] for i in range(len(viz_list))],
            fps=20,
        )

    return s2d


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("MoCa Reconstruction Camera Only")
    parser.add_argument("--ws", type=str, help="Source folder", required=True)
    parser.add_argument("--cfg", type=str, help="profile yaml file path", required=True)
    args, unknown = parser.parse_known_args()

    cfg = OmegaConf.load(args.cfg)
    cli_cfg = OmegaConf.from_dotlist([arg.lstrip("--") for arg in unknown])
    cfg = OmegaConf.merge(cfg, cli_cfg)

    logdir = setup_recon_ws(args.ws, fit_cfg=cfg)

    static_reconstruct(args.ws, logdir, cfg)