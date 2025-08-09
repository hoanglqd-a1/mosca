import torch
import numpy as np
from sam2.sam2_video_predictor import SAM2VideoPredictor
import matplotlib.pyplot as plt
from PIL import Image
import os
import os.path as osp
import imageio

src_dir = "/datasets/iphone/apple/epi/error"

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
epi_mask = epi > 1e-3
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

video_dir = "/datasets/iphone/apple/jpg_images"
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-small")

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    inference_state = predictor.init_state(video_dir)

    for i in range(0, len(frame_names), 10):
        rows, cols = np.where(sample_mask[i])
        indices = np.column_stack((rows, cols))
        np.random.shuffle(indices)
        random_ind = indices[:3]
        for j, rand_idx in enumerate(random_ind):
            ann_obj_id = j  # give a unique id to each object we interact with (it can be any integers)
            ann_frame_idx = i # the frame index we interact with

            points = np.array([rand_idx], dtype=np.float32)
            # for labels, `1` means positive click and `0` means negative click
            labels = np.array([1], np.int32)
            predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id,
                points=points,
                labels=labels,
            )

    # ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)
    # ann_frame_idx = 0 # the frame index we interact with

    # points = np.array([[150, 250]], dtype=np.float32)
    # # for labels, `1` means positive click and `0` means negative click
    # labels = np.array([1], np.int32)
    # predictor.add_new_points_or_box(
    #     inference_state=inference_state,
    #     frame_idx=ann_frame_idx,
    #     obj_id=ann_obj_id,
    #     points=points,
    #     labels=labels,
    # )

    # ann_obj_id = 2  # give a unique id to each object we interact with (it can be any integers)
    # ann_frame_idx = 0  # the frame index we interact with

    # points = np.array([[250, 150]], dtype=np.float32)
    # # for labels, `1` means positive click and `0` means negative click
    # labels = np.array([1], np.int32)
    # predictor.add_new_points_or_box(
    #     inference_state=inference_state,
    #     frame_idx=ann_frame_idx,
    #     obj_id=ann_obj_id,
    #     points=points,
    #     labels=labels,
    # )

    video_segment = {}
    for idx, ids, mask_logits in predictor.propagate_in_video(inference_state):
        video_segment[idx] = {
            out_obj_id: (mask_logits[i] > 0.0)
            for i, out_obj_id in enumerate(ids)
        }
    mask_list = np.zeros((len(frame_names), 480, 360), dtype=bool)
    if os.path.exists("dynamic_mask"):
        import shutil
        shutil.rmtree("dynamic_mask")
    os.mkdir("dynamic_mask")
    for idx in range(len(frame_names)):
        for _, mask_logits in video_segment[idx].items():
            mask = (mask_logits[0] > 0.0).cpu().numpy()  
            mask_list[idx] = mask_list[idx] | mask
        plt.imsave(f"dynamic_mask/0_00{idx:03d}.jpg", mask_list[idx], cmap="gray")

    mask_list = np.stack(mask_list, axis=0)
    # np.save("/datasets/iphone/apple/extra/segment_mask.npy", mask_list)
