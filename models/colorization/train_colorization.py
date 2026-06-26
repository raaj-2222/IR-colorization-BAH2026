"""
Training script for the BAH 2026 colorization stage (Pix2Pix).

Usage:
    python train_colorization.py --patches_root output/patches --epochs 50

With only a handful of samples (while your team is still collecting
more scenes), this script will still run end-to-end and let you
confirm the pipeline works. GANs in particular need substantially
more data than a plain regression model to train stably - expect
visibly noisy/unstable losses on a very small dataset. That is
expected behavior, not a bug, given the dataset size discussion for
this project.
"""

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from colorization_dataset import ColorizationPatchDataset
from colorization_model import UNetGenerator, PatchGANDiscriminator


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train(args):
    device = get_device()
    print(f"Using device: {device}")

    dataset = ColorizationPatchDataset(args.patches_root)
    print(f"Total samples found: {len(dataset)}")

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )

    generator = UNetGenerator().to(device)
    discriminator = PatchGANDiscriminator().to(device)

    optimizer_g = torch.optim.Adam(generator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))

    adversarial_loss = nn.MSELoss()  # LSGAN-style loss, more stable on small datasets than BCE
    l1_loss = nn.L1Loss()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        generator.train()
        discriminator.train()

        g_loss_total = 0.0
        d_loss_total = 0.0

        for tir, real_rgb in loader:
            tir = tir.to(device)
            real_rgb = real_rgb.to(device)
            batch_size_actual = tir.size(0)

            # Discriminator's verdict on a "real" pair should be close to 1,
            # and on a "fake" (generated) pair should be close to 0. The
            # PatchGAN outputs a grid, so the labels are full grids of 1s/0s
            # matching the discriminator's output spatial size.
            with torch.no_grad():
                patch_output_shape = discriminator(tir, real_rgb).shape
            real_labels = torch.ones(patch_output_shape, device=device)
            fake_labels = torch.zeros(patch_output_shape, device=device)

            # --- Train Discriminator ---
            optimizer_d.zero_grad()

            fake_rgb = generator(tir)
            pred_real = discriminator(tir, real_rgb)
            pred_fake = discriminator(tir, fake_rgb.detach())

            d_loss_real = adversarial_loss(pred_real, real_labels)
            d_loss_fake = adversarial_loss(pred_fake, fake_labels)
            d_loss = 0.5 * (d_loss_real + d_loss_fake)

            d_loss.backward()
            optimizer_d.step()

            # --- Train Generator ---
            optimizer_g.zero_grad()

            fake_rgb = generator(tir)
            pred_fake_for_g = discriminator(tir, fake_rgb)

            g_adv_loss = adversarial_loss(pred_fake_for_g, real_labels)
            g_l1_loss = l1_loss(fake_rgb, real_rgb)
            g_loss = g_adv_loss + args.l1_weight * g_l1_loss

            g_loss.backward()
            optimizer_g.step()

            g_loss_total += g_loss.item() * batch_size_actual
            d_loss_total += d_loss.item() * batch_size_actual

        avg_g_loss = g_loss_total / len(dataset)
        avg_d_loss = d_loss_total / len(dataset)

        print(
            f"Epoch {epoch}/{args.epochs} - "
            f"generator_loss: {avg_g_loss:.4f} - discriminator_loss: {avg_d_loss:.4f}"
        )

        # Save the generator's weights every epoch. The discriminator is
        # only needed during training, so we keep it separate and optional.
        gen_path = os.path.join(args.checkpoint_dir, "generator_latest.pth")
        torch.save(generator.state_dict(), gen_path)

        if epoch % args.save_every == 0:
            disc_path = os.path.join(args.checkpoint_dir, f"discriminator_epoch{epoch}.pth")
            torch.save(discriminator.state_dict(), disc_path)

    print(f"Training complete. Checkpoints saved in '{args.checkpoint_dir}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the colorization (Stage B) model.")
    parser.add_argument(
        "--patches_root",
        type=str,
        default="output/patches",
        help="Path to the output/patches directory produced by driver.py.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument(
        "--l1_weight",
        type=float,
        default=100.0,
        help="Weight on the L1 reconstruction loss relative to the adversarial loss. "
        "Pix2Pix's original paper uses 100, which keeps the output close to the "
        "ground truth rather than just 'plausible looking'.",
    )
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints/colorization",
        help="Where to save model weights.",
    )
    args = parser.parse_args()
    train(args)