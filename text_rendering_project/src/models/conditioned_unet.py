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


class ConditionedUNet(nn.Module):

    def __init__(
        self,
        image_channels=3,
        condition_channels=2,
        output_channels=3,
        base_channels=32
    ):
        super().__init__()

        # ==========================================
        # IMAGE ENCODER
        # ==========================================

        self.img_enc1 = DoubleConv(
            image_channels,
            base_channels
        )

        self.img_pool1 = nn.MaxPool2d(2)

        self.img_enc2 = DoubleConv(
            base_channels,
            base_channels * 2
        )

        self.img_pool2 = nn.MaxPool2d(2)

        self.img_enc3 = DoubleConv(
            base_channels * 2,
            base_channels * 4
        )

        self.img_pool3 = nn.MaxPool2d(2)

        self.img_bottleneck = DoubleConv(
            base_channels * 4,
            base_channels * 8
        )

        # ==========================================
        # CONDITION ENCODER
        # Layout + Glyph
        # ==========================================

        self.cond_enc1 = DoubleConv(
            condition_channels,
            base_channels
        )

        self.cond_pool1 = nn.MaxPool2d(2)

        self.cond_enc2 = DoubleConv(
            base_channels,
            base_channels * 2
        )

        self.cond_pool2 = nn.MaxPool2d(2)

        self.cond_enc3 = DoubleConv(
            base_channels * 2,
            base_channels * 4
        )

        self.cond_pool3 = nn.MaxPool2d(2)

        self.cond_bottleneck = DoubleConv(
            base_channels * 4,
            base_channels * 8
        )

        # ==========================================
        # FEATURE FUSION
        # ==========================================

        self.fuse1 = nn.Conv2d(
            base_channels * 2,
            base_channels,
            kernel_size=1
        )

        self.fuse2 = nn.Conv2d(
            base_channels * 4,
            base_channels * 2,
            kernel_size=1
        )

        self.fuse3 = nn.Conv2d(
            base_channels * 8,
            base_channels * 4,
            kernel_size=1
        )

        self.fuse_bottleneck = nn.Conv2d(
            base_channels * 16,
            base_channels * 8,
            kernel_size=1
        )

        # ==========================================
        # DECODER
        # ==========================================

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

        self.output = nn.Conv2d(
            base_channels,
            output_channels,
            kernel_size=1
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, image, condition):

        # ==========================================
        # IMAGE FEATURES
        # ==========================================

        i1 = self.img_enc1(image)

        i2 = self.img_enc2(
            self.img_pool1(i1)
        )

        i3 = self.img_enc3(
            self.img_pool2(i2)
        )

        ib = self.img_bottleneck(
            self.img_pool3(i3)
        )

        # ==========================================
        # CONDITION FEATURES
        # ==========================================

        c1 = self.cond_enc1(condition)

        c2 = self.cond_enc2(
            self.cond_pool1(c1)
        )

        c3 = self.cond_enc3(
            self.cond_pool2(c2)
        )

        cb = self.cond_bottleneck(
            self.cond_pool3(c3)
        )

        # ==========================================
        # FUSE IMAGE + CONDITION
        # ==========================================

        f1 = self.fuse1(
            torch.cat([i1, c1], dim=1)
        )

        f2 = self.fuse2(
            torch.cat([i2, c2], dim=1)
        )

        f3 = self.fuse3(
            torch.cat([i3, c3], dim=1)
        )

        fb = self.fuse_bottleneck(
            torch.cat([ib, cb], dim=1)
        )

        # ==========================================
        # DECODER
        # ==========================================

        d3 = self.up3(fb)

        d3 = torch.cat(
            [d3, f3],
            dim=1
        )

        d3 = self.dec3(d3)

        d2 = self.up2(d3)

        d2 = torch.cat(
            [d2, f2],
            dim=1
        )

        d2 = self.dec2(d2)

        d1 = self.up1(d2)

        d1 = torch.cat(
            [d1, f1],
            dim=1
        )

        d1 = self.dec1(d1)

        output = self.output(d1)

        return self.sigmoid(output)


# ==============================================
# MODEL TEST
# ==============================================

if __name__ == "__main__":

    print("=" * 60)
    print("CONDITIONED U-NET CHECK")
    print("=" * 60)

    model = ConditionedUNet(
        image_channels=3,
        condition_channels=2
    )

    image = torch.randn(
        2, 3, 128, 128
    )

    condition = torch.randn(
        2, 2, 128, 128
    )

    output = model(
        image,
        condition
    )

    print("Image:", image.shape)
    print("Condition:", condition.shape)
    print("Output:", output.shape)

    parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Trainable parameters: {parameters:,}"
    )