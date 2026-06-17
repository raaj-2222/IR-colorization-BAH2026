import tifffile
import numpy as np
import os
import glob
import argparse
import logging
import cv2
from utils.logging_utils import setup_logging
from utils.visualization import percentile_stretch
from utils.file_utils import find_file

def load_rgb(directory):
    """
    Loads RGB data from a single file or from B2, B3, B4 bands.
    Expected to be 100m resolution.
    """
    rgb_file = find_file(directory, "*100m*RGB*")
    if rgb_file:
        return tifffile.imread(rgb_file)

    b2 = find_file(directory, "*100m*B2*")
    b3 = find_file(directory, "*100m*B3*")
    b4 = find_file(directory, "*100m*B4*")

    if b2 and b3 and b4:
        img2 = tifffile.imread(b2)
        img3 = tifffile.imread(b3)
        img4 = tifffile.imread(b4)
        return np.stack([img2, img3, img4], axis=0)

    return None

def save_as_png(data, path):
    """Saves a numpy array as a normalized PNG for visualization."""
    # Handle (C, H, W) or (H, W)
    if data.ndim == 3:
        # (C, H, W) -> (H, W, C)
        data = np.moveaxis(data, 0, -1)

    # Use percentile stretch for better visualization
    stretched = percentile_stretch(data)
    cv2.imwrite(path, stretched)

def create_patches(input_root, output_root):
    os.makedirs(output_root, exist_ok=True)
    logger = setup_logging(output_root)
    
    if not os.path.exists(input_root):
        logger.error(f"Input root directory {input_root} does not exist.")
        return

    # Find product directories in the input_root
    # If input_root is output_downscale_dir, we need to group files by product_id
    all_files = glob.glob(os.path.join(input_root, '*'))
    products = set()
    for f in all_files:
        filename = os.path.basename(f)
        # Assume filename format: {product_id}_...
        product_id = filename.split('_')[0]
        products.add(product_id)

    logger.info(f"Found {len(products)} products in {input_root}")

    for product_id in products:
        # Filter files for this product
        product_files = [f for f in all_files if os.path.basename(f).startswith(product_id)]
        
        # Identify required images for this product
        tir_200m_path = find_file(input_root, f'{product_id}*_tir_200m*')
        tir_100m_path = find_file(input_root, f'{product_id}*_tir_100m*')
        
        # For RGB, look for the rgb_100m file
        rgb_100m_path = find_file(input_root, f'{product_id}*_rgb_100m*')
        
        if not tir_200m_path or not tir_100m_path or not rgb_100m_path:
            logger.warning(f"Skipping {product_id}: Missing required images (200m TIR, 100m TIR, or 100m RGB).")
            continue

        try:
            tir_200m = tifffile.imread(tir_200m_path)
            tir_100m = tifffile.imread(tir_100m_path)
            rgb_100m = tifffile.imread(rgb_100m_path)
        except Exception as e:
            logger.error(f"Error reading images for {product_id}: {e}")
            continue

        h200, w200 = tir_200m.shape[-2:]
        logger.info(f"Generating full dataset patches for {product_id}...")

        count = 0
        for y in range(0, h200 - 256 + 1, 256):
            for x in range(0, w200 - 256 + 1, 256):
                patch_200m_tir = tir_200m[..., y:y+256, x:x+256]

                y100, x100 = 2*y, 2*x
                patch_100m_tir_512 = tir_100m[..., y100:y100+512, x100:x100+512]
                patch_100m_rgb_512 = rgb_100m[..., y100:y100+512, x100:x100+512]

                if patch_100m_tir_512.shape[-2:] != (512, 512) or patch_100m_rgb_512.shape[-2:] != (512, 512):
                    continue

                sample_dir = os.path.join(output_root, product_id, f'sample_{count:03d}')
                os.makedirs(sample_dir, exist_ok=True)

                # Save .npy and .png
                data_map = {
                    'tir_200m': patch_200m_tir,
                    'tir_100m_512': patch_100m_tir_512,
                    'rgb_100m_512': patch_100m_rgb_512
                }

                for name, data in data_map.items():
                    np.save(os.path.join(sample_dir, f'{name}.npy'), data)
                    save_as_png(data, os.path.join(sample_dir, f'{name}.png'))

                count += 1

        logger.info(f"Successfully created {count} samples for {product_id}.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create co-registered image patches.')
    parser.add_argument('--input_dir', type=str, default='input', help='Path to input root directory.')
    parser.add_argument('--output_dir', type=str, default='output/patches', help='Path to output directory.')
    args = parser.parse_args()
    create_patches(args.input_dir, args.output_dir)
