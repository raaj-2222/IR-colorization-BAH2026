"""
Visualize the super-resolution model's output on real samples.

Loads a trained SRResNet checkpoint, runs it on samples from your
patches folder, and saves side-by-side comparison images:
    [ Low-res input (200m) | Real high-res (100m) | Predicted high-res ]

Usage:
    python visualize_sr.py \
        --checkpoint checkpoints/sr/sr_best.pth \
        --patches_root output/patches \
        --num_samples 5 \
        --output_dir comparisons
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from sr_dataset import SRPatchDataset
from sr_model import SRResNet


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def tensor_to_uint8_image(tensor_chw):
    """
    Converts a (1, H, W) grayscale tensor in [0, 1] range (this dataset's
    normalization) to a displayable uint8 numpy array in (H, W, 3) order.
    """
    array = tensor_chw.detach().cpu().numpy()
    array = np.clip(array, 0.0, 1.0)
    array = np.repeat(array, 3, axis=0)  # grayscale -> fake RGB for display
    array = np.transpose(array, (1, 2, 0))  # (C, H, W) -> (H, W, C)
    return (array * 255).astype(np.uint8)


def main(args):
    device = get_device()
    print(f"Using device: {device}")

    dataset = SRPatchDataset(args.patches_root)
    print(f"Total samples found: {len(dataset)}")

    model = SRResNet(num_res_blocks=args.num_res_blocks).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    num_to_show = min(args.num_samples, len(dataset))
    with torch.no_grad():
        for i in range(num_to_show):
            low_res, high_res = dataset[i]
            low_res_batch = low_res.unsqueeze(0).to(device)

            predicted = model(low_res_batch).squeeze(0)
            predicted = torch.clamp(predicted, 0.0, 1.0)

            # Upscale the low-res input visually (nearest neighbor) just so
            # it sits at the same pixel size as the other two panels for
            # an easy side-by-side comparison. This is for display only -
            # it does not affect what the model actually saw as input.
            low_res_display = F.interpolate(
                low_res.unsqueeze(0), size=high_res.shape[-2:], mode="nearest"
            ).squeeze(0)

            low_img = tensor_to_uint8_image(low_res_display)
            real_img = tensor_to_uint8_image(high_res)
            pred_img = tensor_to_uint8_image(predicted)

            combined = np.concatenate([low_img, real_img, pred_img], axis=1)
            out_path = os.path.join(args.output_dir, f"comparison_{i:02d}.png")
            Image.fromarray(combined).save(out_path)
            print(
                f"Saved {out_path}  "
                f"(left: low-res input, upscaled for display | "
                f"middle: real high-res | right: predicted high-res)"
            )

    print(f"\nDone. {num_to_show} comparison images saved in '{args.output_dir}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize super-resolution model output.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/sr/sr_best.pth",
        help="Path to the trained SRResNet weights.",
    )
    parser.add_argument(
        "--patches_root",
        type=str,
        default="output/patches",
        help="Path to the output/patches directory.",
    )
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument(
        "--num_res_blocks",
        type=int,
        default=8,
        help="Must match the value used during training.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="comparisons",
        help="Where to save the comparison images.",
    )
    args = parser.parse_args()
    main(args)