"""
End-to-end inference pipeline for the BAH 2026 IR Colorization challenge.

Takes a raw 200m TIR GeoTIFF as input and produces the two mandatory
output files specified in the README:

    output/model_outputs/tir_superresolved_100m/<product_id>.tif
    output/model_outputs/colorized_tir_100m/<product_id>.tif

Pipeline:
    raw 200m TIR
        -> Super-Resolution model (Stage A)
        -> sharpened 100m TIR
        -> Colorization model (Stage B)
        -> colorized 100m RGB
        -> reordered to Blue, Green, Red channel order (per README spec)
        -> saved as the second output file

Usage:
    python run_pipeline.py \
        --input_tir path/to/raw_200m_B10.TIF \
        --product_id LC09_L1TP_146048_20260322 \
        --sr_checkpoint ../models/super_resolution/checkpoints/sr/sr_best.pth \
        --color_checkpoint ../models/colorization/checkpoints/colorization/generator_latest.pth \
        --output_root output/model_outputs
"""

import argparse
import os
import sys
import time

import numpy as np
import rasterio
import torch

# Make the model definitions importable regardless of where this script
# is run from, as long as the relative folder structure stays intact.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, "..", "models", "super_resolution"))
sys.path.append(os.path.join(SCRIPT_DIR, "..", "models", "colorization"))

from sr_model import SRResNet  # noqa: E402
from colorization_model import UNetGenerator  # noqa: E402


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_tir_geotiff(path):
    """
    Reads a single-band TIR GeoTIFF and returns:
        - the pixel data as a float32 numpy array, shape (1, H, W)
        - the rasterio profile (for preserving georeferencing on output)
    """
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        profile = src.profile.copy()
    return data[np.newaxis, :, :], profile  # add channel dim -> (1, H, W)


def normalize_to_unit_range(array):
    """Min-max normalize to [0, 1] using the array's own range."""
    arr_min, arr_max = array.min(), array.max()
    denom = (arr_max - arr_min) if (arr_max - arr_min) != 0 else 1.0
    return (array - arr_min) / denom, arr_min, arr_max


def denormalize_from_unit_range(array, arr_min, arr_max):
    return array * (arr_max - arr_min) + arr_min


def to_minus_one_one(array_0_1):
    return array_0_1 * 2.0 - 1.0


def from_minus_one_one(array_minus1_1):
    return (array_minus1_1 + 1.0) / 2.0


def save_single_band_geotiff(array_hw, profile, output_path):
    """Saves a (H, W) float array as a single-band GeoTIFF."""
    out_profile = profile.copy()
    out_profile.update(count=1, dtype="float32", height=array_hw.shape[0], width=array_hw.shape[1])
    with rasterio.open(output_path, "w", **out_profile) as dst:
        dst.write(array_hw.astype(np.float32), 1)


def save_bgr_geotiff(rgb_array_chw, profile, output_path):
    """
    Saves a 3-channel array as a GeoTIFF with the README-mandated channel
    order: Layer 1 = Blue, Layer 2 = Green, Layer 3 = Red.

    rgb_array_chw is expected in (R, G, B) order, shape (3, H, W), since
    that is what the colorization model was trained to output. This
    function performs the R,G,B -> B,G,R reordering before saving.
    """
    red, green, blue = rgb_array_chw[0], rgb_array_chw[1], rgb_array_chw[2]
    bgr_array = np.stack([blue, green, red], axis=0)

    out_profile = profile.copy()
    out_profile.update(
        count=3,
        dtype="float32",
        height=bgr_array.shape[1],
        width=bgr_array.shape[2],
    )
    with rasterio.open(output_path, "w", **out_profile) as dst:
        dst.write(bgr_array.astype(np.float32))


def pad_to_multiple(array_chw, tile_size):
    """
    Pads a (C, H, W) array on the bottom/right so H and W are exact
    multiples of tile_size. Returns the padded array and the original
    (H, W) so the padding can be cropped off again after inference.
    """
    _, h, w = array_chw.shape
    pad_h = (tile_size - h % tile_size) % tile_size
    pad_w = (tile_size - w % tile_size) % tile_size
    padded = np.pad(array_chw, ((0, 0), (0, pad_h), (0, pad_w)), mode="edge")
    return padded, (h, w)


def run_tiled_inference(model, input_chw, device, tile_size, scale_factor, out_channels):
    """
    Runs a model over a large (C, H, W) array by splitting it into
    non-overlapping tile_size x tile_size tiles, running inference on
    each tile independently, and stitching the results back into one
    full-size output array.

    scale_factor: the output tile's spatial size relative to the input
                  tile (1 for colorization, 2 for the SR model's 2x
                  upscaling).
    out_channels: number of channels the model outputs (1 for SR,
                  3 for colorization).

    Note: tiles are processed independently, with no overlap or
    blending across tile boundaries. This can produce faint seams at
    tile edges in the stitched output - a known tradeoff of simple
    tiled inference, traded off here for simplicity and reliability
    on arbitrarily large inputs.
    """
    padded, original_hw = pad_to_multiple(input_chw, tile_size)
    _, padded_h, padded_w = padded.shape

    out_h, out_w = padded_h * scale_factor, padded_w * scale_factor
    output = np.zeros((out_channels, out_h, out_w), dtype=np.float32)

    num_tiles_h = padded_h // tile_size
    num_tiles_w = padded_w // tile_size
    total_tiles = num_tiles_h * num_tiles_w
    tile_count = 0

    with torch.no_grad():
        for row in range(num_tiles_h):
            for col in range(num_tiles_w):
                tile_count += 1
                y0, y1 = row * tile_size, (row + 1) * tile_size
                x0, x1 = col * tile_size, (col + 1) * tile_size

                tile = padded[:, y0:y1, x0:x1]
                tile_tensor = torch.from_numpy(tile).unsqueeze(0).to(device)

                result_tensor = model(tile_tensor)
                result = result_tensor.squeeze(0).cpu().numpy()

                out_y0, out_y1 = y0 * scale_factor, y1 * scale_factor
                out_x0, out_x1 = x0 * scale_factor, x1 * scale_factor
                output[:, out_y0:out_y1, out_x0:out_x1] = result

                if tile_count % 20 == 0 or tile_count == total_tiles:
                    print(f"  Processed tile {tile_count}/{total_tiles}")

    # Crop back to the original (unpadded) size, scaled appropriately.
    final_h, final_w = original_hw[0] * scale_factor, original_hw[1] * scale_factor
    return output[:, :final_h, :final_w]


def run_pipeline(args):
    device = get_device()
    print(f"Using device: {device}")

    os.makedirs(os.path.join(args.output_root, "tir_superresolved_100m"), exist_ok=True)
    os.makedirs(os.path.join(args.output_root, "colorized_tir_100m"), exist_ok=True)

    # --- Load input ---
    print(f"Loading input TIR: {args.input_tir}")
    tir_raw, profile = load_tir_geotiff(args.input_tir)  # (1, H, W)
    print(f"Input shape: {tir_raw.shape}")

    tir_norm, tir_min, tir_max = normalize_to_unit_range(tir_raw)

    # --- Stage A: Super-Resolution (tiled) ---
    print("Loading super-resolution model...")
    sr_model = SRResNet(num_res_blocks=args.sr_num_res_blocks).to(device)
    sr_model.load_state_dict(torch.load(args.sr_checkpoint, map_location=device))
    sr_model.eval()

    print(
        f"Running super-resolution inference in {args.sr_tile_size}x{args.sr_tile_size} "
        f"tiles (this may take a while for large scenes)..."
    )
    start = time.time()
    sr_output_norm = run_tiled_inference(
        sr_model, tir_norm, device,
        tile_size=args.sr_tile_size, scale_factor=2, out_channels=1,
    )
    sr_output_norm = np.clip(sr_output_norm, 0.0, 1.0)
    sr_elapsed = time.time() - start
    print(f"Super-resolution inference time: {sr_elapsed:.3f}s")

    sr_output_denorm = denormalize_from_unit_range(sr_output_norm, tir_min, tir_max)

    sr_output_path = os.path.join(
        args.output_root, "tir_superresolved_100m", f"{args.product_id}.tif"
    )
    save_single_band_geotiff(sr_output_denorm[0], profile, sr_output_path)
    print(f"Saved super-resolved TIR to: {sr_output_path}")

    # --- Stage B: Colorization (tiled) ---
    print("Loading colorization model...")
    color_model = UNetGenerator().to(device)
    color_model.load_state_dict(torch.load(args.color_checkpoint, map_location=device))
    color_model.eval()

    # The colorization model expects [-1, 1] input, matching its training
    # normalization. Convert the already [0, 1]-normalized SR output.
    color_input = to_minus_one_one(sr_output_norm)

    print(
        f"Running colorization inference in {args.color_tile_size}x{args.color_tile_size} "
        f"tiles..."
    )
    start = time.time()
    color_output = run_tiled_inference(
        color_model, color_input, device,
        tile_size=args.color_tile_size, scale_factor=1, out_channels=3,
    )
    color_elapsed = time.time() - start
    print(f"Colorization inference time: {color_elapsed:.3f}s")

    color_output_0_1 = from_minus_one_one(color_output)  # back to [0, 1] for saving

    color_output_path = os.path.join(
        args.output_root, "colorized_tir_100m", f"{args.product_id}.tif"
    )
    save_bgr_geotiff(color_output_0_1, profile, color_output_path)
    print(f"Saved colorized RGB (BGR channel order) to: {color_output_path}")

    total_elapsed = sr_elapsed + color_elapsed
    print(f"\nTotal inference time: {total_elapsed:.3f}s")
    print("Pipeline complete.")

    return {
        "sr_output_path": sr_output_path,
        "color_output_path": color_output_path,
        "sr_inference_time": sr_elapsed,
        "color_inference_time": color_elapsed,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the full TIR super-resolution + colorization pipeline."
    )
    parser.add_argument(
        "--input_tir",
        type=str,
        required=True,
        help="Path to the raw 200m TIR GeoTIFF (e.g. a B10.TIF band).",
    )
    parser.add_argument(
        "--product_id",
        type=str,
        required=True,
        help="Product ID used for naming output files. Must match the "
        "original input product ID per the README's submission spec.",
    )
    parser.add_argument(
        "--sr_checkpoint",
        type=str,
        default="../models/super_resolution/checkpoints/sr/sr_best.pth",
    )
    parser.add_argument(
        "--color_checkpoint",
        type=str,
        default="../models/colorization/checkpoints/colorization/generator_latest.pth",
    )
    parser.add_argument("--sr_num_res_blocks", type=int, default=8)
    parser.add_argument(
        "--sr_tile_size",
        type=int,
        default=256,
        help="Tile size for super-resolution inference, must match training patch size.",
    )
    parser.add_argument(
        "--color_tile_size",
        type=int,
        default=512,
        help="Tile size for colorization inference, must match training patch size.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="output/model_outputs",
        help="Root directory for the mandatory output structure.",
    )
    args = parser.parse_args()
    run_pipeline(args)