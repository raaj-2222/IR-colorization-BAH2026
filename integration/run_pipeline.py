"""
End-to-end inference pipeline for the BAH 2026 IR Colorization challenge.

Takes a raw 200m TIR GeoTIFF as input and produces the two mandatory
output files specified in the README:

    output/model_outputs/tir_superresolved_100m/<product_id>.tif
    output/model_outputs/colorized_tir_100m/<product_id>.tif

Pipeline:
    raw 200m TIR
        -> Super-Resolution model (Stage A), processed tile by tile
           with overlapping context windows to avoid seam artifacts
        -> sharpened 100m TIR, written directly to disk
        -> Colorization model (Stage B), reading the sharpened TIR
           back from disk tile by tile, also with overlap
        -> colorized 100m RGB
        -> reordered to Blue, Green, Red channel order (per README spec)
        -> written directly to disk as the second output file

Both stages stream tiles to/from disk rather than holding the full
satellite scene in memory at once. This matters because a full Landsat
scene (commonly 7000+ x 7000+ pixels) produces a multi-gigabyte array
once super-resolved and colorized - building that array fully in memory
can exhaust available RAM on constrained environments (e.g. free-tier
cloud GPU runtimes), even when GPU memory itself is not the bottleneck.

Each tile is read with extra `overlap` pixels of context on every side
before being fed to the model, and only the central (non-overlapping)
region of the result is written to disk. Without this, every tile is
colorized/sharpened with zero awareness of its neighbors, producing
visible grid-line seams at tile boundaries in the final mosaic.

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


def run_sr_tiled_to_disk(model, input_path, device, tile_size, output_path,
                          value_min, value_max, scale_factor=2, overlap=16):
    """
    Super-resolution stage. Reads each tile with `overlap` pixels of extra
    context on every side, runs the model on the padded tile, then crops
    the output back down to just the core (non-overlapping) region before
    writing - this avoids seam artifacts at tile boundaries.
    """
    with rasterio.open(input_path) as src:
        in_height, in_width = src.height, src.width
        in_profile = src.profile.copy()

        out_h, out_w = in_height * scale_factor, in_width * scale_factor
        out_profile = in_profile.copy()
        out_profile.update(count=1, dtype="float32", height=out_h, width=out_w)

        denom = (value_max - value_min) or 1.0

        num_tiles_h = (in_height + tile_size - 1) // tile_size
        num_tiles_w = (in_width + tile_size - 1) // tile_size
        total_tiles = num_tiles_h * num_tiles_w
        tile_count = 0

        with rasterio.open(output_path, "w", **out_profile) as dst:
            with torch.no_grad():
                for y0 in range(0, in_height, tile_size):
                    for x0 in range(0, in_width, tile_size):
                        tile_count += 1
                        read_h = min(tile_size, in_height - y0)
                        read_w = min(tile_size, in_width - x0)
                        if read_h <= 0 or read_w <= 0:
                            continue

                        # expand read window by `overlap` on each side, clamped to image bounds
                        ry0 = max(0, y0 - overlap)
                        rx0 = max(0, x0 - overlap)
                        ry1 = min(in_height, y0 + read_h + overlap)
                        rx1 = min(in_width, x0 + read_w + overlap)

                        window = rasterio.windows.Window(rx0, ry0, rx1 - rx0, ry1 - ry0)
                        raw_tile = src.read(1, window=window).astype(np.float32)

                        # pad with value_min (not zero) so padding stays radiometrically
                        # neutral after normalization, and pad up to a fixed shape so
                        # the model always sees a consistent input size
                        pad_h = tile_size + 2 * overlap
                        pad_w = tile_size + 2 * overlap
                        padded_tile = np.full((pad_h, pad_w), value_min, dtype=np.float32)
                        padded_tile[: raw_tile.shape[0], : raw_tile.shape[1]] = raw_tile
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

                        # crop back to just the core (non-overlap) region, scaled up
                        crop_y0 = (y0 - ry0) * scale_factor
                        crop_x0 = (x0 - rx0) * scale_factor
                        crop_h = read_h * scale_factor
                        crop_w = read_w * scale_factor
                        cropped = result_denorm[
                            crop_y0: crop_y0 + crop_h, crop_x0: crop_x0 + crop_w
                        ]

                        out_y0, out_x0 = y0 * scale_factor, x0 * scale_factor
                        out_window = rasterio.windows.Window(out_x0, out_y0, crop_w, crop_h)
                        dst.write(cropped.astype(np.float32), 1, window=out_window)

                        if tile_count % 50 == 0 or tile_count == total_tiles:
                            print(f"  SR: processed tile {tile_count}/{total_tiles}")


def run_colorization_tiled_to_disk(model, sr_output_path, device, tile_size,
                                    output_path, overlap=32, nodata_value=None):
    """
    Colorization stage.

    IMPORTANT: UNetGenerator's skip connections require its input to be
    exactly divisible through every downsampling stage (it was trained on
    fixed `tile_size` x `tile_size` inputs, e.g. 512x512). Naively adding
    overlap on top of `tile_size` (e.g. 512 + 2*32 = 576) breaks that
    divisibility and causes encoder/decoder skip-connection shape
    mismatches at inference time.

    To get overlap *without* changing the model's input size, the model
    always receives a fixed `tile_size` x `tile_size` window, but only the
    central `core_size = tile_size - 2*overlap` region of each result is
    written to disk. The write grid steps by `core_size` instead of
    `tile_size`, so neighboring writes still tile seamlessly while each
    tile's edges (which the model saw with full surrounding context) are
    discarded instead of written.
    """
    if 2 * overlap >= tile_size:
        raise ValueError("overlap must be less than tile_size / 2")
    core_size = tile_size - 2 * overlap

    with rasterio.open(sr_output_path) as src:
        height, width = src.height, src.width
        in_profile = src.profile.copy()

        full_min, full_max = compute_tir_min_max(sr_output_path, tile_size=tile_size)
        denom = (full_max - full_min) or 1.0

        # --- Global color reference ---
        # UNetGenerator uses InstanceNorm, which normalizes each forward pass
        # using only that input's own statistics. Run independently tile by
        # tile, this causes every tile to land at a slightly different
        # brightness/contrast/hue - visible as a tile grid in the final
        # mosaic. To counteract this without retraining, run the model once
        # on a single downsampled view of the *whole* scene to get one
        # consistent "global style" reference, then rescale every real tile's
        # output to match that reference's per-channel mean/std before
        # writing. This doesn't change tile content, only brings tone/
        # contrast into agreement across tiles.
        print("  Colorization: computing global color reference...")
        with torch.no_grad():
            ref_full = src.read(
                1,
                out_shape=(1, tile_size, tile_size),
                resampling=rasterio.enums.Resampling.bilinear,
            ).astype(np.float32)
            ref_norm = (ref_full - full_min) / denom * 2.0 - 1.0
            ref_tensor = torch.from_numpy(ref_norm[np.newaxis, np.newaxis, :, :]).to(device)
            ref_result = model(ref_tensor).squeeze(0).cpu().numpy()
            del ref_tensor
            if device.type == "cuda":
                torch.cuda.empty_cache()
        ref_result_0_1 = (ref_result + 1.0) / 2.0
        ref_bgr = ref_result_0_1[[2, 1, 0], :, :]
        if nodata_value is not None:
            ref_is_nodata = np.abs(ref_full - nodata_value) <= (
                1e-3 * (abs(nodata_value) + 1.0)
            )
            ref_valid = ~ref_is_nodata
        else:
            ref_valid = np.ones_like(ref_full, dtype=bool)
        # per-channel mean/std over valid (non-nodata) reference pixels
        global_mean = np.array([
            ref_bgr[c][ref_valid].mean() if ref_valid.any() else 0.5 for c in range(3)
        ], dtype=np.float32)
        global_std = np.array([
            ref_bgr[c][ref_valid].std() if ref_valid.any() else 1.0 for c in range(3)
        ], dtype=np.float32)
        global_std = np.maximum(global_std, 1e-3)
        print(f"  Colorization: global reference mean={global_mean}, std={global_std}")

        out_profile = in_profile.copy()
        out_profile.update(count=3, dtype="float32", height=height, width=width)

        num_tiles_h = (height + core_size - 1) // core_size
        num_tiles_w = (width + core_size - 1) // core_size
        total_tiles = num_tiles_h * num_tiles_w
        tile_count = 0

        with rasterio.open(output_path, "w", **out_profile) as dst:
            with torch.no_grad():
                for out_y0 in range(0, height, core_size):
                    for out_x0 in range(0, width, core_size):
                        tile_count += 1
                        write_h = min(core_size, height - out_y0)
                        write_w = min(core_size, width - out_x0)
                        if write_h <= 0 or write_w <= 0:
                            continue

                        # read window is centered on the core region, expanded by
                        # `overlap` on each side, clamped to image bounds
                        ry0 = max(0, out_y0 - overlap)
                        rx0 = max(0, out_x0 - overlap)
                        ry1 = min(height, out_y0 + write_h + overlap)
                        rx1 = min(width, out_x0 + write_w + overlap)

                        window = rasterio.windows.Window(rx0, ry0, rx1 - rx0, ry1 - ry0)
                        raw_tile = src.read(1, window=window).astype(np.float32)

                        # Identify no-data pixels with a small tolerance rather than
                        # exact equality - sensor edge effects mean padding regions
                        # are rarely *perfectly* uniform.
                        nodata_tolerance = 1e-3 * (abs(nodata_value) + 1.0) if nodata_value is not None else 0.0
                        is_nodata_pixel = (
                            np.abs(raw_tile - nodata_value) <= nodata_tolerance
                            if nodata_value is not None else None
                        )

                        # Fully no-data tiles (e.g. corners entirely outside the
                        # rotated scene frame): skip the model call entirely and
                        # write zeros - running these through the model produces
                        # hallucinated solid-color output, since it never saw pure
                        # padding input during training.
                        if is_nodata_pixel is not None and np.all(is_nodata_pixel):
                            zeros = np.zeros((3, write_h, write_w), dtype=np.float32)
                            out_window = rasterio.windows.Window(out_x0, out_y0, write_w, write_h)
                            dst.write(zeros, window=out_window)
                            if tile_count % 50 == 0 or tile_count == total_tiles:
                                print(f"  Colorization: processed tile {tile_count}/{total_tiles} (skipped, no-data)")
                            continue

                        # always feed the model a fixed tile_size x tile_size input,
                        # padding with full_min (radiometrically neutral) as needed
                        padded_tile = np.full((tile_size, tile_size), full_min, dtype=np.float32)
                        # place the read data so that its position matches how far
                        # the read window was clamped in from the intended overlap
                        place_y0 = ry0 - (out_y0 - overlap)
                        place_x0 = rx0 - (out_x0 - overlap)
                        padded_tile[
                            place_y0: place_y0 + raw_tile.shape[0],
                            place_x0: place_x0 + raw_tile.shape[1],
                        ] = raw_tile

                        tile_norm_0_1 = (padded_tile - full_min) / denom
                        tile_minus1_1 = tile_norm_0_1 * 2.0 - 1.0
                        tile_tensor = (
                            torch.from_numpy(tile_minus1_1[np.newaxis, np.newaxis, :, :])
                            .to(device)
                        )

                        result_tensor = model(tile_tensor)
                        result = result_tensor.squeeze(0).cpu().numpy()  # (3, tile_size, tile_size) R,G,B
                        del tile_tensor, result_tensor
                        if device.type == "cuda":
                            torch.cuda.empty_cache()

                        result_0_1 = (result + 1.0) / 2.0
                        result_bgr = result_0_1[[2, 1, 0], :, :]  # README requires B, G, R order

                        # crop to the central core region (full real context on
                        # every side, no padding influence)
                        crop_y0 = overlap - (out_y0 - ry0)
                        crop_x0 = overlap - (out_x0 - rx0)
                        cropped = result_bgr[
                            :, crop_y0: crop_y0 + write_h, crop_x0: crop_x0 + write_w
                        ]

                        # No-data mask for this tile's core write region, used
                        # both to exclude no-data pixels from tone-matching
                        # statistics and to zero them out in the final output.
                        if is_nodata_pixel is not None:
                            mask_padded = np.zeros((tile_size, tile_size), dtype=bool)
                            mask_padded[
                                place_y0: place_y0 + is_nodata_pixel.shape[0],
                                place_x0: place_x0 + is_nodata_pixel.shape[1],
                            ] = is_nodata_pixel
                            mask_cropped = mask_padded[
                                crop_y0: crop_y0 + write_h, crop_x0: crop_x0 + write_w
                            ]
                            valid_cropped = ~mask_cropped
                        else:
                            mask_cropped = None
                            valid_cropped = np.ones((write_h, write_w), dtype=bool)

                        # Match this tile's tone/contrast to the global reference
                        # (computed once over the whole scene at the top of this
                        # function), using only this tile's valid (non-nodata)
                        # pixels for the per-tile mean/std. This counteracts
                        # InstanceNorm's per-tile-independent normalization,
                        # which otherwise causes a visible tile grid in the
                        # final mosaic.
                        if valid_cropped.any():
                            cropped = cropped.copy()
                            for c in range(3):
                                channel = cropped[c]
                                valid_vals = channel[valid_cropped]
                                tile_mean = valid_vals.mean()
                                tile_std = max(valid_vals.std(), 1e-3)
                                channel[valid_cropped] = (
                                    (valid_vals - tile_mean) / tile_std
                                ) * global_std[c] + global_mean[c]
                                cropped[c] = np.clip(channel, 0.0, 1.0)

                        # Zero out no-data pixels within this tile (handles
                        # tiles that are *partially* no-data, e.g. along the
                        # rotated scene's diagonal edge) - these were generated
                        # by the model but should be black, not hallucinated
                        # color, since they carry no real TIR signal.
                        if mask_cropped is not None:
                            cropped = cropped.copy()
                            cropped[:, mask_cropped] = 0.0

                        out_window = rasterio.windows.Window(out_x0, out_y0, write_w, write_h)
                        dst.write(cropped.astype(np.float32), window=out_window)

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
        f"tiles with {args.sr_overlap}px overlap (this may take a while for large scenes)..."
    )
    start = time.time()
    run_sr_tiled_to_disk(
        sr_model, args.input_tir, device, args.sr_tile_size,
        sr_output_path, tir_min, tir_max,
        scale_factor=2, overlap=args.sr_overlap,
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
        f"tiles with {args.color_overlap}px overlap..."
    )
    start = time.time()
    run_colorization_tiled_to_disk(
        color_model, sr_output_path, device, args.color_tile_size, color_output_path,
        overlap=args.color_overlap,
        nodata_value=args.nodata_value if args.nodata_value is not None else tir_min,
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
    parser.add_argument(
        "--color_tile_size", type=int, default=4096,
        help="Must be divisible by 256 (2^8, the UNetGenerator's downsampling depth). "
             "4096 confirmed to run without OOM; 8192 OOM'd. Larger tiles reduce visible "
             "tone/contrast variation between tiles, since InstanceNorm normalizes each "
             "tile independently using only that tile's own statistics.",
    )
    parser.add_argument(
        "--sr_overlap", type=int, default=16,
        help="Context padding (px) added around each SR tile before inference, "
             "cropped off before writing. Reduces seam artifacts at tile borders.",
    )
    parser.add_argument(
        "--color_overlap", type=int, default=256,
        help="Context padding (px) added around each colorization tile before "
             "inference, cropped off before writing. Must be < color_tile_size / 2.",
    )
    parser.add_argument(
        "--nodata_value", type=float, default=None,
        help="Pixel value in the SR output representing no-data/padding (e.g. outside "
             "the rotated scene frame). Tiles that are entirely this value are skipped "
             "during colorization instead of being run through the model, which avoids "
             "hallucinated solid-color output for background/no-data regions. Defaults "
             "to the SR stage's value_min if not set.",
    )
    parser.add_argument("--output_root", type=str, default="output/model_outputs")
    args = parser.parse_args()
    run_pipeline(args)