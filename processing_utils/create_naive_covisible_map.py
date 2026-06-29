import os
import sys
import argparse
import cv2
import numpy as np
import shutil

def main():
    parser = argparse.ArgumentParser(description="Create a naive covisible map for each image.")
    parser.add_argument("--work_dir", type=str, required=True, help="Path to the working directory containing images.")
    args = parser.parse_args()

    test_covisible_dir = os.path.join(args.work_dir, "test_covisible")
    if os.path.exists(test_covisible_dir):
        shutil.rmtree(test_covisible_dir) # Remove existing directory to avoid conflicts

    os.makedirs(test_covisible_dir, exist_ok=True)

    image_dir = os.path.join(args.work_dir, "test_images")
    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        image = cv2.imread(image_path)
        covisible_map = 255 * np.ones_like(image)[..., 0] # Create a white image as the covisible map
        save_path = os.path.join(test_covisible_dir, image_name)
        cv2.imwrite(save_path, covisible_map)

    print(f"Naive covisible maps created and saved to {test_covisible_dir}")
    
if __name__ == "__main__":
    main()