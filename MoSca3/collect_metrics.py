import os, os.path as osp
import pandas as pd
from tqdm import tqdm
import datetime
import shutil

scene_name_list = [
    "Jumping",
    # "Skating",
    # "Truck",
    # "Umbrella",
    # "Balloon1",
    # "Balloon2",
    # "Playground",
]

root = "/datasets/nvidia/"

backup_root = "/datasets/metrics_collected"
prefix = "tto_"
full_backup=False


for name in ["nvidia_fit_colfree_native"]:
    backup_dir = osp.join(
        backup_root, prefix + name + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    results = {}
    for scene_name in tqdm(scene_name_list):
        # find the latest log
        log_dirs = os.listdir(osp.join(root, scene_name, "logs"))
        log_dirs = [d for d in log_dirs if name in d]

        log_dirs.sort()
        log_dir = log_dirs[-1]
        fn = osp.join(
            root,
            scene_name,
            "logs",
            log_dir,
            f"{prefix}test_report",
            "nvidia_render_metrics.txt",
        )
        assert osp.exists(fn), f"{fn} does not exist"
        with open(fn, "r") as f:
            lines = f.readlines()
        psnr = float(lines[0].split(":")[-1][:-1])
        ssim = float(lines[1].split(":")[-1][:-1])
        lpips = float(lines[2].split(":")[-1][:-1])
        print(scene_name)
        print(fn)
        print(lines)
        print(psnr, ssim, lpips)
        results[f"{scene_name}-psnr"] = psnr
        # results[f"{scene_name}-ssim"] = ssim
        results[f"{scene_name}-lpips"] = lpips
        # copy the log_dir into the backup
        if full_backup:
            shutil.copytree(
                osp.join(root, scene_name, "logs", log_dir),
                osp.join(backup_dir, f"{scene_name}_{log_dir}"),
            )
        else:
            shutil.copytree(
                osp.join(root, scene_name, "logs", log_dir, f"{prefix}test"),
                osp.join(backup_dir, scene_name + "_test"),
            )
            shutil.copytree(
                osp.join(root, scene_name, "logs", log_dir, f"{prefix}test_report"),
                osp.join(backup_dir, scene_name + "_test_report"),
            )
    df = pd.DataFrame([results], index=[0])
    df.to_excel(osp.join(backup_dir, f"{prefix}{name}_nvidia_metrics.xlsx"))