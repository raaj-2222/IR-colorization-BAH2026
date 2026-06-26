"""
Training script for the BAH 2026 super-resolution stage.

Usage:
    python train_sr.py --patches_root output/patches --epochs 50

With only a handful of samples (e.g. while your team is still
collecting more scenes), this script will still run end-to-end and
let you confirm the pipeline works. It will not produce a good model
on very few samples - that requires more training data, per the
dataset size discussion for this project.
"""

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from sr_dataset import SRPatchDataset
from sr_model import SRResNet


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train(args):
    device = get_device()
    print(f"Using device: {device}")

    full_dataset = SRPatchDataset(args.patches_root)
    print(f"Total samples found: {len(full_dataset)}")

    # Hold out a validation split if there's enough data to do so meaningfully.
    # With very few samples, skip the split entirely and validate on the same
    # data just to track the loss trend, not as a real generalization metric.
    if len(full_dataset) >= 5:
        val_size = max(1, int(0.2 * len(full_dataset)))
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    else:
        print(
            "WARNING: Very few samples found. Using all data for both "
            "training and validation. Treat the validation loss as a "
            "sanity check, not a real measure of generalization."
        )
        train_dataset = full_dataset
        val_dataset = full_dataset

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    model = SRResNet(num_res_blocks=args.num_res_blocks).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.L1Loss()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_total = 0.0
        for low_res, high_res in train_loader:
            low_res = low_res.to(device)
            high_res = high_res.to(device)

            optimizer.zero_grad()
            predicted = model(low_res)
            loss = criterion(predicted, high_res)
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item() * low_res.size(0)

        avg_train_loss = train_loss_total / len(train_dataset)

        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for low_res, high_res in val_loader:
                low_res = low_res.to(device)
                high_res = high_res.to(device)
                predicted = model(low_res)
                loss = criterion(predicted, high_res)
                val_loss_total += loss.item() * low_res.size(0)

        avg_val_loss = val_loss_total / len(val_dataset)

        print(
            f"Epoch {epoch}/{args.epochs} - "
            f"train_loss: {avg_train_loss:.4f} - val_loss: {avg_val_loss:.4f}"
        )

        # Save the most recent checkpoint every epoch, and a separate
        # "best" checkpoint whenever validation loss improves.
        latest_path = os.path.join(args.checkpoint_dir, "sr_latest.pth")
        torch.save(model.state_dict(), latest_path)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = os.path.join(args.checkpoint_dir, "sr_best.pth")
            torch.save(model.state_dict(), best_path)

    print(f"Training complete. Checkpoints saved in '{args.checkpoint_dir}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the SR (Stage A) model.")
    parser.add_argument(
        "--patches_root",
        type=str,
        default="output/patches",
        help="Path to the output/patches directory produced by driver.py.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument(
        "--num_res_blocks",
        type=int,
        default=8,
        help="Number of residual blocks. Lower this if training is too slow on CPU.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints/sr",
        help="Where to save model weights.",
    )
    args = parser.parse_args()
    train(args)