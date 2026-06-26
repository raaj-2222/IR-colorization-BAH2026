"""
Dataset loader for the BAH 2026 IR Colorization task.

Expects the folder structure produced by driver.py:
    output/patches/<product_name>/<sample_name>/tir_100m_512.npy
    output/patches/<product_name>/<sample_name>/rgb_100m_512.npy

Each sample folder must contain both files. This loader scans every
product/sample subfolder under the given root and treats each as one
training pair.

Note: this is intentionally the same scanning logic as sr_dataset.py
but kept as a separate file/class so Role 2 and Role 3's code can
evolve independently without stepping on each other.
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


class ColorizationPatchDataset(Dataset):
    """
    Loads (grayscale TIR, RGB) pairs for the colorization stage.

    Input  (x): tir_100m_512.npy   shape (1, 512, 512)
    Target (y): rgb_100m_512.npy   shape (3, 512, 512)

    Pixel values are rescaled to [-1, 1], which is the standard range
    expected by Pix2Pix-style generators using a final Tanh activation.
    """

    def __init__(self, patches_root):
        self.pairs = []  # list of (tir_path, rgb_path)

        search_pattern = os.path.join(patches_root, "*", "*")
        for sample_dir in sorted(glob.glob(search_pattern)):
            if not os.path.isdir(sample_dir):
                continue
            tir_path = os.path.join(sample_dir, "tir_100m_512.npy")
            rgb_path = os.path.join(sample_dir, "rgb_100m_512.npy")
            if os.path.exists(tir_path) and os.path.exists(rgb_path):
                self.pairs.append((tir_path, rgb_path))

        if len(self.pairs) == 0:
            raise RuntimeError(
                f"No valid (tir_100m_512.npy, rgb_100m_512.npy) pairs found "
                f"under '{patches_root}'. Check the path and folder structure."
            )

    def __len__(self):
        return len(self.pairs)

    @staticmethod
    def _to_minus_one_one(arr):
        """Rescale an array's own min/max range to [-1, 1]."""
        arr_min, arr_max = arr.min(), arr.max()
        denom = (arr_max - arr_min) if (arr_max - arr_min) != 0 else 1.0
        scaled_0_1 = (arr - arr_min) / denom
        return scaled_0_1 * 2.0 - 1.0

    def __getitem__(self, idx):
        tir_path, rgb_path = self.pairs[idx]

        tir = np.load(tir_path).astype(np.float32)   # (1, 512, 512)
        rgb = np.load(rgb_path).astype(np.float32)    # (3, 512, 512)

        tir = self._to_minus_one_one(tir)
        rgb = self._to_minus_one_one(rgb)

        return torch.from_numpy(tir), torch.from_numpy(rgb)


if __name__ == "__main__":
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else "output/patches"
    ds = ColorizationPatchDataset(root)
    print(f"Found {len(ds)} samples.")
    x, y = ds[0]
    print(f"TIR input shape: {tuple(x.shape)}  range: [{x.min():.2f}, {x.max():.2f}]")
    print(f"RGB target shape: {tuple(y.shape)}  range: [{y.min():.2f}, {y.max():.2f}]")