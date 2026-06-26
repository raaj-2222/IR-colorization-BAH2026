"""
Visualize the colorization model's output on real samples.

Loads a trained generator checkpoint, runs it on samples from your
patches folder, and saves side-by-side comparison images:
    [ Input TIR | Real RGB | Generated RGB ]

Usage:
    python visualize_colorization.py \
        --checkpoint checkpoints/colorization/generator_latest.pth \
        --patches_root output/patches \
        --num_samples 5 \
        --output_dir comparisons
"""

import argparse
import os

import numpy as np
import torch
from PIL import Image

from colorization_dataset import ColorizationPatchDataset
from colorization_model import UNetGenerator


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def tensor_to_uint8_image(tensor_chw):
    """
    Converts a (C, H, W) tensor in [-1, 1] range to a displayable
    uint8 numpy array in (H, W, C) order, scaled to [0, 255].

    Grayscale (1-channel) tensors are replicated to 3 channels so they
    can sit side by side with RGB images in one combined figure.
    """
    array = tensor_chw.detach().cpu().numpy()
    array = (array + 1.0) / 2.0  # [-1, 1] -> [0, 1]
    array = np.clip(array, 0.0, 1.0)

    if array.shape[0] == 1:
        array = np.repeat(array, 3, axis=0)  # grayscale -> fake RGB for display

    array = np.transpose(array, (1, 2, 0))  # (C, H, W) -> (H, W, C)
    return (array * 255).astype(np.uint8)


def main(args):
    device = get_device()
    print(f"Using device: {device}")

    dataset = ColorizationPatchDataset(args.patches_root)
    print(f"Total samples found: {len(dataset)}")

    generator = UNetGenerator().to(device)
    generator.load_state_dict(torch.load(args.checkpoint, map_location=device))
    generator.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    num_to_show = min(args.num_samples, len(dataset))
    with torch.no_grad():
        for i in range(num_to_show):
            tir, real_rgb = dataset[i]
            tir_batch = tir.unsqueeze(0).to(device)  # add batch dimension

            fake_rgb = generator(tir_batch).squeeze(0)  # remove batch dimension

            tir_img = tensor_to_uint8_image(tir)
            real_img = tensor_to_uint8_image(real_rgb)
            fake_img = tensor_to_uint8_image(fake_rgb)

            combined = np.concatenate([tir_img, real_img, fake_img], axis=1)
            out_path = os.path.join(args.output_dir, f"comparison_{i:02d}.png")
            Image.fromarray(combined).save(out_path)
            print(f"Saved {out_path}  (left: input TIR | middle: real RGB | right: generated RGB)")

    print(f"\nDone. {num_to_show} comparison images saved in '{args.output_dir}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize colorization model output.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/colorization/generator_latest.pth",
        help="Path to the trained generator weights.",
    )
    parser.add_argument(
        "--patches_root",
        type=str,
        default="output/patches",
        help="Path to the output/patches directory.",
    )
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="comparisons",
        help="Where to save the comparison images.",
    )
    args = parser.parse_args()
    main(args)