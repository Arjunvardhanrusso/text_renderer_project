import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):

    def __init__(
        self,
        input_channels=3,
        output_channels=3,
        base_channels=32
    ):
        super().__init__()

        # -------------------------
        # Encoder
        # -------------------------

        self.enc1 = DoubleConv(
            input_channels,
            base_channels
        )

        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(
            base_channels,
            base_channels * 2
        )

        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(
            base_channels * 2,
            base_channels * 4
        )

        self.pool3 = nn.MaxPool2d(2)

        # -------------------------
        # Bottleneck
        # -------------------------

        self.bottleneck = DoubleConv(
            base_channels * 4,
            base_channels * 8
        )

        # -------------------------
        # Decoder
        # -------------------------

        self.up3 = nn.ConvTranspose2d(
            base_channels * 8,
            base_channels * 4,
            kernel_size=2,
            stride=2
        )

        self.dec3 = DoubleConv(
            base_channels * 8,
            base_channels * 4
        )

        self.up2 = nn.ConvTranspose2d(
            base_channels * 4,
            base_channels * 2,
            kernel_size=2,
            stride=2
        )

        self.dec2 = DoubleConv(
            base_channels * 4,
            base_channels * 2
        )

        self.up1 = nn.ConvTranspose2d(
            base_channels * 2,
            base_channels,
            kernel_size=2,
            stride=2
        )

        self.dec1 = DoubleConv(
            base_channels * 2,
            base_channels
        )

        # -------------------------
        # Final RGB output
        # -------------------------

        self.output = nn.Conv2d(
            base_channels,
            output_channels,
            kernel_size=1
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # Encoder
        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool1(e1)
        )

        e3 = self.enc3(
            self.pool2(e2)
        )

        # Bottleneck
        b = self.bottleneck(
            self.pool3(e3)
        )

        # Decoder
        d3 = self.up3(b)

        d3 = torch.cat(
            [d3, e3],
            dim=1
        )

        d3 = self.dec3(d3)

        d2 = self.up2(d3)

        d2 = torch.cat(
            [d2, e2],
            dim=1
        )

        d2 = self.dec2(d2)

        d1 = self.up1(d2)

        d1 = torch.cat(
            [d1, e1],
            dim=1
        )

        d1 = self.dec1(d1)

        output = self.output(d1)

        # Keep output between 0 and 1
        return self.sigmoid(output)


if __name__ == "__main__":

    print("=" * 50)
    print("U-NET MODEL CHECK")
    print("=" * 50)

    # -------------------------
    # Baseline
    # -------------------------

    baseline = UNet(
        input_channels=3
    )

    baseline_input = torch.randn(
        2, 3, 128, 128
    )

    baseline_output = baseline(
        baseline_input
    )

    print(
        "Baseline output:",
        baseline_output.shape
    )

    # -------------------------
    # Layout
    # -------------------------

    layout_model = UNet(
        input_channels=4
    )

    layout_input = torch.randn(
        2, 4, 128, 128
    )

    layout_output = layout_model(
        layout_input
    )

    print(
        "Layout output:",
        layout_output.shape
    )

    # -------------------------
    # Layout + Glyph
    # -------------------------

    glyph_model = UNet(
        input_channels=5
    )

    glyph_input = torch.randn(
        2, 5, 128, 128
    )

    glyph_output = glyph_model(
        glyph_input
    )

    print(
        "Layout + Glyph output:",
        glyph_output.shape
    )

    # Parameter count
    parameters = sum(
        p.numel()
        for p in glyph_model.parameters()
        if p.requires_grad
    )

    print(
        f"\nTrainable parameters: "
        f"{parameters:,}"
    )