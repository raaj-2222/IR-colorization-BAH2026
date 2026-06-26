"""
A compact SRResNet for 4x single-channel super-resolution.

Input:  (batch, 1, 256, 256)
Output: (batch, 1, 512, 512)

Architecture: an initial conv, a stack of residual blocks, a global
skip connection, then two pixel-shuffle upsampling stages (2x each,
total 4x) followed by a final conv back to 1 channel.

This is intentionally compact (fewer residual blocks, fewer channels
than the original SRResNet paper) since the goal here is a fast,
trainable baseline for a hackathon timeline, not a state-of-the-art
result. It can be scaled up later by increasing num_res_blocks or
num_channels if training time allows.
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return out + residual


class UpsampleBlock(nn.Module):
    """Upscales spatial dimensions by 2x using pixel shuffle."""

    def __init__(self, num_channels):
        super().__init__()
        self.conv = nn.Conv2d(num_channels, num_channels * 4, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.pixel_shuffle(self.conv(x)))


class SRResNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_channels=64, num_res_blocks=8):
        super().__init__()

        # Initial feature extraction
        self.input_conv = nn.Conv2d(in_channels, num_channels, kernel_size=9, padding=4)
        self.input_relu = nn.ReLU(inplace=True)

        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_channels) for _ in range(num_res_blocks)]
        )

        # Conv after residual blocks, before the global skip connection
        self.mid_conv = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.mid_bn = nn.BatchNorm2d(num_channels)

        # Upsampling: one 2x stage (256 -> 512 matches our target scale)
        self.upsample1 = UpsampleBlock(num_channels)

        # Final conv back to the target number of channels
        self.output_conv = nn.Conv2d(num_channels, out_channels, kernel_size=9, padding=4)

    def forward(self, x):
        initial_features = self.input_relu(self.input_conv(x))

        out = self.res_blocks(initial_features)
        out = self.mid_bn(self.mid_conv(out))
        out = out + initial_features  # global residual skip connection

        out = self.upsample1(out)

        out = self.output_conv(out)
        return out


if __name__ == "__main__":
    # Quick shape sanity check, no real data needed.
    model = SRResNet()
    dummy_input = torch.randn(2, 1, 256, 256)  # batch of 2
    output = model(dummy_input)
    print(f"Input shape:  {tuple(dummy_input.shape)}")
    print(f"Output shape: {tuple(output.shape)}")
    assert output.shape == (2, 1, 512, 512), "Output shape mismatch!"
    print("Shape check passed.")