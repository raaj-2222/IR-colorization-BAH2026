"""
End-to-end inference pipeline for the BAH 2026 IR Colorization challenge.

Takes a raw 200m TIR GeoTIFF as input and produces the two mandatory
output files specified in the README:

    output/model_outputs/tir_superresolved_100m/<product_id>.tif
    output/model_outputs/colorized_tir_100m/<product_id>.tif

Pipeline:
    raw 200m TIR
        -> Super-Resolution model (Stage A), processed tile by tile
        -> sharpened 100m TIR, written directly to disk
        -> Colorization model (Stage B), reading the sharpened TIR
           back from disk tile by tile
        -> colorized 100m RGB
        -> reordered to Blue, Green, Red channel order (per README spec)
        -> written directly to disk as the second output file

Both stages stream tiles to/from disk rather than holding the full
satellite scene in memory at once. This matters because a full Landsat
scene (commonly 7000+ x 7000+ pixels) produces a multi-gigabyte array
once super-resolved and colorized - building that array fully in memory
can exhaust available RAM on constrained environments (e.g. free-tier
cloud GPU runtimes), even when GPU memory itself is not the bottleneck.

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
import rasterio.windows
import torch

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
    with rasterio.open(path) as src:
        profile = src.profile.copy()
        height, width = src.height, src.width
    return profile, height, width


def pad_size_to_multiple(size, tile_size):
    remainder = size % tile_size
    return size if remainder == 0 else size + (tile_size - remainder)


def compute_tir_min_max(path, tile_size=512):
    with rasterio.open(path) as src:
        height, width = src.height, src.width
        full_min, full_max = None, None
        for row_start in range(0, height, tile_size):
            row_count = min(tile_size, height - row_start)
            window = rasterio.windows.Window(0, row_start, width, row_count)
            block = src.read(1, window=window).astype(np.float32)
            block_min, block_max = block.min(), block.max()
            full_min = block_min if full_min is None else min(full_min, block_min)
            full_max = block_max if full_max is None else max(full_max, block_max)
    return full_min, full_max


def run_sr_tiled_to_disk(model, input_path, device, tile_size, output_path, value_min, value_max):
    with rasterio.open(input_path) as src:
        in_height, in_width = src.height, src.width
        in_profile = src.profile.copy()

        padded_h = pad_size_to_multiple(in_height, tile_size)
        padded_w = pad_size_to_multiple(in_width, tile_size)
        num_tiles_h = padded_h // tile_size
        num_tiles_w = padded_w // tile_size
        total_tiles = num_tiles_h * num_tiles_w

        out_h, out_w = in_height * 2, in_width * 2
        out_profile = in_profile.copy()
        out_profile.update(count=1, dtype="float32", height=out_h, width=out_w)

        denom = (value_max - value_min) or 1.0
        tile_count = 0

        with rasterio.open(output_path, "w", **out_profile) as dst:
            with torch.no_grad():
                for row in range(num_tiles_h):
                    for col in range(num_tiles_w):
                        tile_count += 1
                        y0 = row * tile_size
                        x0 = col * tile_size
                        read_h = min(tile_size, in_height - y0)
                        read_w = min(tile_size, in_width - x0)
                        if read_h <= 0 or read_w <= 0:
                            continue

                        window = rasterio.windows.Window(x0, y0, read_w, read_h)
                        raw_tile = src.read(1, window=window).astype(np.float32)

                        if raw_tile.shape != (tile_size, tile_size):
                            padded_tile = np.zeros((tile_size, tile_size), dtype=np.float32)
                            padded_tile[:read_h, :read_w] = raw_tile
                            raw_tile = padded_tile

                        tile_norm = (raw_tile - value_min) / denom
                        tile_tensor = (
                            torch.from_numpy(tile_norm[np.newaxis, np.newaxis, :, :])
                            .to(device)
                        )

                        result_tensor = model(tile_tensor)
                        result = result_tensor.squeeze(0).squeeze(0).cpu().numpy()
                        del tile_tensor, result_tensor
                        if device.type == "cuda":
                            torch.cuda.empty_cache()

                        result = np.clip(result, 0.0, 1.0)
                        result_denorm = result * denom + value_min

                        out_y0, out_x0 = y0 * 2, x0 * 2
                        write_h = min(read_h * 2, out_h - out_y0)
                        write_w = min(read_w * 2, out_w - out_x0)
                        out_window = rasterio.windows.Window(out_x0, out_y0, write_w, write_h)
                        dst.write(
                            result_denorm[:write_h, :write_w].astype(np.float32),
                            1,
                            window=out_window,
                        )

                        if tile_count % 50 == 0 or tile_count == total_tiles:
                            print(f"  SR: processed tile {tile_count}/{total_tiles}")


def run_colorization_tiled_to_disk(model, sr_output_path, device, tile_size, output_path):
    with rasterio.open(sr_output_path) as src:
        height, width = src.height, src.width
        in_profile = src.profile.copy()

        full_min, full_max = compute_tir_min_max(sr_output_path, tile_size=tile_size)
        denom = (full_max - full_min) or 1.0

        padded_h = pad_size_to_multiple(height, tile_size)
        padded_w = pad_size_to_multiple(width, tile_size)
        num_tiles_h = padded_h // tile_size
        num_tiles_w = padded_w // tile_size
        total_tiles = num_tiles_h * num_tiles_w

        out_profile = in_profile.copy()
        out_profile.update(count=3, dtype="float32", height=height, width=width)

        tile_count = 0
        with rasterio.open(output_path, "w", **out_profile) as dst:
            with torch.no_grad():
                for row in range(num_tiles_h):
                    for col in range(num_tiles_w):
                        tile_count += 1
                        y0 = row * tile_size
                        x0 = col * tile_size
                        read_h = min(tile_size, height - y0)
                        read_w = min(tile_size, width - x0)
                        if read_h <= 0 or read_w <= 0:
                            continue

                        window = rasterio.windows.Window(x0, y0, read_w, read_h)
                        raw_tile = src.read(1, window=window).astype(np.float32)

                        if raw_tile.shape != (tile_size, tile_size):
                            padded_tile = np.zeros((tile_size, tile_size), dtype=np.float32)
                            padded_tile[:read_h, :read_w] = raw_tile
                            raw_tile = padded_tile

                        tile_norm_0_1 = (raw_tile - full_min) / denom
                        tile_minus1_1 = tile_norm_0_1 * 2.0 - 1.0
                        tile_tensor = (
                            torch.from_numpy(tile_minus1_1[np.newaxis, np.newaxis, :, :])
                            .to(device)
                        )

                        result_tensor = model(tile_tensor)
                        result = result_tensor.squeeze(0).cpu().numpy()  # (3, tile, tile) R,G,B
                        del tile_tensor, result_tensor
                        if device.type == "cuda":
                            torch.cuda.empty_cache()

                        result_0_1 = (result + 1.0) / 2.0
                        result_bgr = result_0_1[[2, 1, 0], :, :]

                        out_window = rasterio.windows.Window(x0, y0, read_w, read_h)
                        dst.write(
                            result_bgr[:, :read_h, :read_w].astype(np.float32),
                            window=out_window,
                        )

                        if tile_count % 50 == 0 or tile_count == total_tiles:
                            print(f"  Colorization: processed tile {tile_count}/{total_tiles}")


def run_pipeline(args):
    device = get_device()
    print(f"Using device: {device}")

    os.makedirs(os.path.join(args.output_root, "tir_superresolved_100m"), exist_ok=True)
    os.makedirs(os.path.join(args.output_root, "colorized_tir_100m"), exist_ok=True)

    print(f"Loading input TIR: {args.input_tir}")
    _, in_height, in_width = load_tir_geotiff(args.input_tir)
    print(f"Input shape: (1, {in_height}, {in_width})")

    print("Computing input value range...")
    tir_min, tir_max = compute_tir_min_max(args.input_tir, tile_size=args.sr_tile_size)
    print(f"Input value range: [{tir_min}, {tir_max}]")

    print("Loading super-resolution model...")
    sr_model = SRResNet(num_res_blocks=args.sr_num_res_blocks).to(device)
    sr_model.load_state_dict(torch.load(args.sr_checkpoint, map_location=device))
    sr_model.eval()

    sr_output_path = os.path.join(
        args.output_root, "tir_superresolved_100m", f"{args.product_id}.tif"
    )
    print(
        f"Running super-resolution inference in {args.sr_tile_size}x{args.sr_tile_size} "
        f"tiles (this may take a while for large scenes)..."
    )
    start = time.time()
    run_sr_tiled_to_disk(
        sr_model, args.input_tir, device, args.sr_tile_size,
        sr_output_path, tir_min, tir_max,
    )
    sr_elapsed = time.time() - start
    print(f"Super-resolution inference time: {sr_elapsed:.3f}s")
    print(f"Saved super-resolved TIR to: {sr_output_path}")

    del sr_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print("Freed super-resolution model from memory.")

    print("Loading colorization model...")
    color_model = UNetGenerator().to(device)
    color_model.load_state_dict(torch.load(args.color_checkpoint, map_location=device))
    color_model.eval()

    color_output_path = os.path.join(
        args.output_root, "colorized_tir_100m", f"{args.product_id}.tif"
    )
    print(
        f"Running colorization inference in {args.color_tile_size}x{args.color_tile_size} "
        f"tiles..."
    )
    start = time.time()
    run_colorization_tiled_to_disk(
        color_model, sr_output_path, device, args.color_tile_size, color_output_path,
    )
    color_elapsed = time.time() - start
    print(f"Colorization inference time: {color_elapsed:.3f}s")
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
    parser.add_argument("--input_tir", type=str, required=True)
    parser.add_argument("--product_id", type=str, required=True)
    parser.add_argument(
        "--sr_checkpoint", type=str,
        default="../models/super_resolution/checkpoints/sr/sr_best.pth",
    )
    parser.add_argument(
        "--color_checkpoint", type=str,
        default="../models/colorization/checkpoints/colorization/generator_latest.pth",
    )
    parser.add_argument("--sr_num_res_blocks", type=int, default=8)
    parser.add_argument("--sr_tile_size", type=int, default=256)
    parser.add_argument("--color_tile_size", type=int, default=512)
    parser.add_argument("--output_root", type=str, default="output/model_outputs")
    args = parser.parse_args()
    run_pipeline(args)