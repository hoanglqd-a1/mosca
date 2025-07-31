import os
from PIL import Image

def convert_png_to_jpg(input_folder: str, output_folder: str = None):
    """
    Converts all PNG images in a given folder to JPG format.

    Args:
        input_folder (str): The path to the directory containing the PNG images.
        output_folder (str, optional): The path to the directory where the converted JPG images
                                        will be saved. If None, images will be saved in a
                                        subfolder named 'converted_jpg' within the input_folder.
    """
    if not os.path.isdir(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist.")
        return

    # Determine the output folder
    if output_folder is None:
        output_folder = os.path.join(input_folder, "converted_jpg")
    os.makedirs(output_folder, exist_ok=True)
    print(f"Converted JPG images will be saved to: '{output_folder}'")

    print(f"Starting conversion of PNG images in '{input_folder}'...")
    converted_count = 0

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".png"):
            png_path = os.path.join(input_folder, filename)
            # Create the output filename by changing the extension
            jpg_filename = os.path.splitext(filename)[0] + ".jpg"
            jpg_path = os.path.join(output_folder, jpg_filename)

            try:
                # Open the PNG image
                with Image.open(png_path) as img:
                    # PNGs can have an alpha channel (transparency).
                    # JPGs do not support alpha, so convert to RGB mode.
                    # If you don't do this, transparent areas might become black.
                    if img.mode == 'RGBA':
                        img = img.convert('RGB')
                    elif img.mode == 'P': # Handle paletted images (common for PNG)
                        img = img.convert('RGB')

                    # Save the image as JPG
                    # quality (1-95, default is 75) can be adjusted for file size vs. quality
                    img.save(jpg_path, "jpeg", quality=90)
                print(f"Converted '{filename}' to '{jpg_filename}'")
                converted_count += 1
            except Exception as e:
                print(f"Error converting '{filename}': {e}")
        else:
            print(f"Skipping non-PNG file: '{filename}'")

    if converted_count > 0:
        print(f"\nConversion completed. Successfully converted {converted_count} PNG images to JPG.")
    else:
        print("\nNo PNG images found for conversion in the specified folder.")


# --- Example Usage ---
if __name__ == "__main__":
    # 1. Create a dummy input folder with some placeholder PNG images for testing
    dummy_input_folder = "/datasets/iphone/spin/images"
    os.makedirs(dummy_input_folder, exist_ok=True)

    # 2. Define the output folder (optional, will create 'converted_jpg' subfolder if None)
    dummy_output_folder = "/datasets/iphone/spin/jpg_images"
    # Or set to None: dummy_output_folder = None

    # 3. Call the function to perform the conversion
    convert_png_to_jpg(dummy_input_folder, dummy_output_folder)
