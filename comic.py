#!/usr/bin/env python3
"""
Comic Panel Segmentation using Classical Computer Vision

This script processes all images in an input folder, applies classical image processing 
techniques to detect comic panels, and saves each extracted panel as a separate image
directly into one specified output folder.

Pipeline:
    1. Load and convert the comic page to grayscale.
    2. Apply Canny edge detection to detect edges.
    3. Dilate the edges to bridge gaps.
    4. Fill holes to produce solid regions.
    5. Label connected regions and extract bounding boxes.
    6. Filter out small regions.
    7. Save each detected panel image into one output folder.

Usage:
    python comic.py --input_dir path/to/input_images --output_dir path/to/output_panels

Dependencies:
    - OpenCV (cv2)
    - NumPy
    - scikit-image
    - SciPy
    - Pillow
"""

import os
import argparse
import cv2
import numpy as np
from skimage import feature, measure
from scipy.ndimage import binary_fill_holes
from PIL import Image

def process_image(image_path, output_dir, area_thresh_ratio=0.01):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read image {image_path}")
        return

    height, width = image.shape[:2]

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = feature.canny(gray, sigma=2.0)
    edges_uint8 = (edges.astype(np.uint8)) * 255

    # Dilate the edges to thicken and connect them
    kernel = np.ones((3, 3), dtype=np.uint8)
    dilated = cv2.dilate(edges_uint8, kernel, iterations=2)

    # Fill holes to obtain solid regions corresponding to panels
    filled = binary_fill_holes(dilated > 0)
    filled_uint8 = (filled.astype(np.uint8)) * 255

    # Label connected components in the filled image
    labels = measure.label(filled, connectivity=2)
    regions = measure.regionprops(labels)

    # Define a minimum area threshold (as a ratio of the image area)
    min_area = area_thresh_ratio * (height * width)

    # Extract bounding boxes for regions that exceed the minimum area
    bboxes = []
    for region in regions:
        if region.area >= min_area:
            # region.bbox returns (min_row, min_col, max_row, max_col)
            min_row, min_col, max_row, max_col = region.bbox
            bboxes.append((min_col, min_row, max_col, max_row))  # (x1, y1, x2, y2)

    # Sort bounding boxes top-to-bottom, then left-to-right
    bboxes = sorted(bboxes, key=lambda box: (box[1], box[0]))

    # Use the base name of the input image for naming panels
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # Crop and save each detected panel into the single output directory
    for i, (x1, y1, x2, y2) in enumerate(bboxes):
        panel = image[y1:y2, x1:x2]
        panel_img = Image.fromarray(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
        panel_filename = os.path.join(output_dir, f"{base_name}_panel_{i}.png")
        panel_img.save(panel_filename)

    print(f"Processed {image_path}: {len(bboxes)} panels detected and saved.")

def main():
    parser = argparse.ArgumentParser(description="Comic Panel Segmentation using Classical CV")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to the folder containing comic page images.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to the folder where panel images will be saved.")
    parser.add_argument("--area_thresh_ratio", type=float, default=0.01,
                        help="Minimum area (as a ratio of total image area) to consider a region a panel (default: 0.01).")
    args = parser.parse_args()

    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Get list of image files from the input directory
    valid_exts = [".jpg", ".jpeg", ".png", ".bmp"]
    image_files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir)
                   if os.path.splitext(f)[1].lower() in valid_exts]

    if not image_files:
        print("No valid images found in the input directory.")
        return

    # Process each image
    for image_path in image_files:
        process_image(image_path, args.output_dir, args.area_thresh_ratio)

if __name__ == "__main__":
    main()
