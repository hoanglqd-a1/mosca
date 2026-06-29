import cv2
import os
import sys
import argparse
import json
import shutil
import numpy as np

def copy_images(input_dir, output_dir, ratio, odd_skip=True):
    rgb_indir = os.path.join(input_dir, f"rgb/{ratio}x")
    train_image_outdir = os.path.join(output_dir, "images")
    test_image_outdir = os.path.join(output_dir, "test_images")

    if os.path.exists(train_image_outdir):
        shutil.rmtree(train_image_outdir) # Remove existing directory to avoid conflicts
    if os.path.exists(test_image_outdir):
        shutil.rmtree(test_image_outdir) # Remove existing directory to avoid conflicts

    os.makedirs(train_image_outdir, exist_ok=True)
    os.makedirs(test_image_outdir, exist_ok=True)

    for image_name in list(sorted(os.listdir(rgb_indir))):
        image = cv2.imread(os.path.join(rgb_indir, image_name))

        image_idx = int(image_name.split(".")[0].split("_")[0])
        if odd_skip and image_idx % 2 == 1: continue # Skip odd-indexed images to create a smaller dataset

        save_path = os.path.join(train_image_outdir, f"0_{image_name.split('.')[0].split('_')[0]}.png") \
                    if image_name.split(".")[0].endswith("left") else \
                    os.path.join(test_image_outdir, f"1_{image_name.split('.')[0].split('_')[0]}.png")
        cv2.imwrite(save_path, image)

def copy_segmentation_masks(input_dir, output_dir, ratio, odd_skip=True):
    mask_dir = os.path.join(input_dir, f"mask/1x")
    train_mask_outdir = os.path.join(output_dir, "masks")
    test_mask_outdir = os.path.join(output_dir, "test_masks")

    if os.path.exists(train_mask_outdir):
        shutil.rmtree(train_mask_outdir) # Remove existing directory to avoid conflicts
    if os.path.exists(test_mask_outdir):
        shutil.rmtree(test_mask_outdir) # Remove existing directory to avoid conflicts

    os.makedirs(train_mask_outdir, exist_ok=True)
    os.makedirs(test_mask_outdir, exist_ok=True)

    for mask_name in list(sorted(os.listdir(mask_dir))):
        mask = cv2.imread(os.path.join(mask_dir, mask_name), cv2.IMREAD_GRAYSCALE)

        image_idx = int(mask_name.split(".")[0].split("_")[0])
        if odd_skip and image_idx % 2 == 1: continue # Skip odd-indexed images to create a smaller dataset

        save_path = os.path.join(train_mask_outdir, f"0_{mask_name.split('.')[0].split('_')[0]}.png") \
                    if mask_name.split(".")[0].endswith("left") else \
                    os.path.join(test_mask_outdir, f"1_{mask_name.split('.')[0].split('_')[0]}.png")
        if ratio != 1:
            mask = cv2.resize(mask, (mask.shape[1] // ratio, mask.shape[0] // ratio), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(save_path, mask)

def copy_cameras(input_dir, output_dir, ratio, odd_skip=True):
    camera_indir = os.path.join(input_dir, f"camera")
    train_camera_outdir = os.path.join(output_dir, "cameras")
    test_camera_outdir = os.path.join(output_dir, "test_cameras")

    if os.path.exists(train_camera_outdir):
        shutil.rmtree(train_camera_outdir) # Remove existing directory to avoid conflicts
    if os.path.exists(test_camera_outdir):
        shutil.rmtree(test_camera_outdir) # Remove existing directory to avoid conflicts

    os.makedirs(train_camera_outdir, exist_ok=True)
    os.makedirs(test_camera_outdir, exist_ok=True)

    for camera_name in list(sorted(os.listdir(camera_indir))):
        camera_path = os.path.join(camera_indir, camera_name)

        image_idx = int(camera_name.split(".")[0].split("_")[0])
        if odd_skip and image_idx % 2 == 1: continue # Skip odd-indexed images to create a smaller dataset

        save_path = os.path.join(train_camera_outdir, f"0_{camera_name.split('.')[0].split('_')[0]}.json") \
                    if camera_name.split(".")[0].endswith("left") else \
                    os.path.join(test_camera_outdir, f"1_{camera_name.split('.')[0].split('_')[0]}.json")

        with open(camera_path, 'r', encoding='utf-8') as f:
            camera_data = json.load(f)

        # camera_data['image_size'][0] //= ratio
        # camera_data['image_size'][1] //= ratio

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(camera_data, f, indent=4)

def create_naive_covisible_pairs(output_dir):
    test_covisible_dir = os.path.join(output_dir, "test_covisible")
    if os.path.exists(test_covisible_dir):
        shutil.rmtree(test_covisible_dir) # Remove existing directory to avoid conflicts

    os.makedirs(test_covisible_dir, exist_ok=True)

    image_dir = os.path.join(output_dir, "test_images")
    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        image = cv2.imread(image_path)
        covisible_map = 255 * np.ones_like(image)[..., 0] # Create a white image as the covisible map
        save_path = os.path.join(test_covisible_dir, image_name)
        cv2.imwrite(save_path, covisible_map)

    print(f"Naive covisible maps created and saved to {test_covisible_dir}")

def main():
    parser = argparse.ArgumentParser(description="Convert NerfDS dataset to iPhone format")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the input NerfDS dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output iPhone dataset")
    parser.add_argument("--scale_ratio", choices=["1", "2", "4", "8"], default=1, help="Scale ratio for resizing images (default: 1)")
    parser.add_argument("--half", action="store_true", help="Whether to skip odd-indexed images to create a smaller dataset (default: False)")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    ratio = int(args.scale_ratio)
    odd_skip = args.half

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    copy_images(input_dir, output_dir, ratio, odd_skip)
    copy_segmentation_masks(input_dir, output_dir, ratio, odd_skip)
    copy_cameras(input_dir, output_dir, ratio, odd_skip)
    create_naive_covisible_pairs(output_dir)

    print("Conversion completed successfully!")
    
if __name__ == "__main__":
    main()