import torch
import numpy as np
from sam2.sam2_video_predictor import SAM2VideoPredictor
import matplotlib.pyplot as plt
from PIL import Image
import os

video_dir = "/datasets/iphone/spin/jpg_images"
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-small")

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    inference_state = predictor.init_state(video_dir)

    ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)
    ann_frame_idx = 0  # the frame index we interact with

    points = np.array([[180, 270], [200, 230]], dtype=np.float32)
    # for labels, `1` means positive click and `0` means negative click
    labels = np.array([1, 1], np.int32)
    predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )
#     mask = (out_mask_logits[0][0] > 0.0).cpu().numpy()  
#     plt.imsave("mask.jpg", mask, cmap="gray")
#     image = np.array(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
#     mask_3ch = np.stack([mask]*3, axis=-1)
#     masked = np.zeros_like(image)
#     masked[mask_3ch] = image[mask_3ch]
#     plt.imsave("image.jpg", masked)
    video_segment = {}
    for idx, ids, mask_logits in predictor.propagate_in_video(inference_state):
        video_segment[idx] = {
            out_obj_id: (mask_logits[i] > 0.0)
            for i, out_obj_id in enumerate(ids)
        }
    if os.path.exists("dynamic_mask"):
        import shutil
        shutil.rmtree("dynamic_mask")
    os.mkdir("dynamic_mask")
    mask_list = []
    for idx in range(len(frame_names)):
        for _, mask_logits in video_segment[idx].items():
            mask = (mask_logits[0] > 0.0).cpu().numpy()  
            mask_list.append(mask)
            plt.imsave(f"dynamic_mask/0_00{idx:03d}.jpg", mask, cmap="gray")

    mask_3ch = np.stack(mask_list, axis=0)
    np.save(os.path.join("dynamic_mask", "segment_mask.npy"), mask_3ch)