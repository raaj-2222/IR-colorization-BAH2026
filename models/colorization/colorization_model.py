"""
Pix2Pix-style model for the BAH 2026 IR Colorization task.

Generator:    U-Net, 1-channel TIR in -> 3-channel RGB out, same spatial size.
Discriminator: PatchGAN, judges whether (input, output) pairs look real,
               patch by patch rather than as a single whole-image verdict.

Input/output spatial size: 512x512 (must be divisible by 2 enough times
for the U-Net's downsampling path - 512 = 2^9, so an 8-level U-Net is safe).
"""

import torch
import torch.nn as nn


class UNetDown(nn.Module):
    """One downsampling step: conv -> (optional norm) -> LeakyReLU."""

    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNetUp(nn.Module):
    """One upsampling step: deconv -> norm -> ReLU, then concat with skip."""

    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x, skip_connection):
        x = self.block(x)
        return torch.cat((x, skip_connection), dim=1)


class UNetGenerator(nn.Module):
    """
    U-Net generator for 512x512 inputs.

    8 downsampling steps (512 -> 256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2... wait,
    actually 8 steps takes 512 down to 2). We use 8 down/up steps which is
    standard for 512x512 Pix2Pix.
    """

    def __init__(self, in_channels=1, out_channels=3):
        super().__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)


class PatchGANDiscriminator(nn.Module):
    """
    Judges (input_image, target_or_generated_image) pairs concatenated
    along the channel dimension. Outputs a grid of real/fake scores
    rather than a single number, which encourages locally realistic
    texture rather than just a globally plausible average.
    """

    def __init__(self, in_channels=1, target_channels=3):
        super().__init__()
        combined_channels = in_channels + target_channels

        def block(in_c, out_c, normalize=True):
            layers = [nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(combined_channels, 64, normalize=False),
            *block(64, 128),
            *block(128, 256),
            *block(256, 512),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, input_image, target_or_generated):
        combined = torch.cat((input_image, target_or_generated), dim=1)
        return self.model(combined)


if __name__ == "__main__":
    # Quick shape sanity check, no real data needed.
    generator = UNetGenerator()
    discriminator = PatchGANDiscriminator()

    dummy_tir = torch.randn(2, 1, 512, 512)
    fake_rgb = generator(dummy_tir)
    print(f"Generator input shape:  {tuple(dummy_tir.shape)}")
    print(f"Generator output shape: {tuple(fake_rgb.shape)}")
    assert fake_rgb.shape == (2, 3, 512, 512), "Generator output shape mismatch!"

    disc_output = discriminator(dummy_tir, fake_rgb)
    print(f"Discriminator output shape: {tuple(disc_output.shape)}")

    print("Shape checks passed.")