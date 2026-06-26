"""
Dataset loader for the BAH 2026 IR Super-Resolution task.

Expects the folder structure produced by driver.py:
    output/patches/<product_name>/<sample_name>/tir_200m.npy
    output/patches/<product_name>/<sample_name>/tir_100m_512.npy

Each sample folder must contain both files. This loader scans every
product/sample subfolder under the given root and treats each as one
training pair.
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


class SRPatchDataset(Dataset):
    """
    Loads (low-res TIR, high-res TIR) pairs for the super-resolution stage.

    Input  (x): tir_200m.npy       shape (1, 256, 256)
    Target (y): tir_100m_512.npy   shape (1, 512, 512)
    """

    def __init__(self, patches_root, normalize=True):
        """
        Args:
            patches_root: path to the output/patches directory (contains
                           one subfolder per product, each containing one
                           or more sample_* subfolders).
            normalize:    if True, scale pixel values to [0, 1] using each
                           array's own min/max. Set False if your data is
                           already normalized upstream.
        """
        self.normalize = normalize
        self.pairs = []  # list of (low_res_path, high_res_path)

        # Find every sample folder that has both required files.
        search_pattern = os.path.join(patches_root, "*", "*")
        for sample_dir in sorted(glob.glob(search_pattern)):
            if not os.path.isdir(sample_dir):
                continue
            low_path = os.path.join(sample_dir, "tir_200m.npy")
            high_path = os.path.join(sample_dir, "tir_100m_512.npy")
            if os.path.exists(low_path) and os.path.exists(high_path):
                self.pairs.append((low_path, high_path))

        if len(self.pairs) == 0:
            raise RuntimeError(
                f"No valid (tir_200m.npy, tir_100m_512.npy) pairs found under "
                f"'{patches_root}'. Check the path and folder structure."
            )

    def __len__(self):
        return len(self.pairs)

    def _load_and_normalize(self, path):
        arr = np.load(path).astype(np.float32)
        if self.normalize:
            arr_min, arr_max = arr.min(), arr.max()
            denom = (arr_max - arr_min) if (arr_max - arr_min) != 0 else 1.0
            arr = (arr - arr_min) / denom
        return arr

    def __getitem__(self, idx):
        low_path, high_path = self.pairs[idx]
        low_res = self._load_and_normalize(low_path)   # (1, 256, 256)
        high_res = self._load_and_normalize(high_path)  # (1, 512, 512)
        return torch.from_numpy(low_res), torch.from_numpy(high_res)


if __name__ == "__main__":
    # Quick manual test: point this at your local output/patches folder.
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else "output/patches"
    ds = SRPatchDataset(root)
    print(f"Found {len(ds)} samples.")
    x, y = ds[0]
    print(f"Low-res input shape:  {tuple(x.shape)}")
    print(f"High-res target shape: {tuple(y.shape)}")