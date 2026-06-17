import numpy as np
import tifffile
import os
import argparse
import cv2
import logging

logger = logging.getLogger(__name__)

from utils.file_utils import validate_extension

def box_average_downscale(image, factor):
    h, w = image.shape
    new_h = int(round(h / factor))
    new_w = int(round(w / factor))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

def downscale_image(input_filepath, output_filepath, scale_factor):
    validate_extension(input_filepath)
    validate_extension(output_filepath)
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

    image_data = tifffile.imread(input_filepath)
    if image_data.ndim == 2:
        image_data = image_data[np.newaxis, ...]

    downscaled_bands = []
    for band in image_data:
        downscaled_band = box_average_downscale(band, scale_factor)
        downscaled_bands.append(downscaled_band)

    downscaled_data = np.stack(downscaled_bands, axis=0)
    tifffile.imwrite(output_filepath, downscaled_data.astype(image_data.dtype))
    logger.info(f'Downscaled {input_filepath} by factor {scale_factor} (box average) to {output_filepath}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Downscale a TIFF image using box average.')
    parser.add_argument('input_filepath', type=str, help='Path to the input TIFF file.')
    parser.add_argument('output_filepath', type=str, help='Path to save the downscaled TIFF file.')
    parser.add_argument('scale_factor', type=float, help='Factor by which to downscale (e.g., 3.33 for 30m to 100m).')

    args = parser.parse_args()

    try:
        downscale_image(args.input_filepath, args.output_filepath, args.scale_factor)
    except Exception as e:
        logger.error(f"Error downscaling image: {e}")
        exit(1)
