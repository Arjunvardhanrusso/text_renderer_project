import torch
import torch.nn as nn


class GlyphRenderer(nn.Module):
    """
    Predicts the RGB appearance/color of text.

    The glyph itself is used as a hard structural mask,
    so the network cannot remove the character shapes.
    """

    def __init__(self):
        super().__init__()

        # Input:
        # masked RGB = 3
        # reconstructed RGB = 3
        # layout = 1
        # glyph = 1
        #
        # Total = 8 channels

        self.network = nn.Sequential(

            nn.Conv2d(
                8,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                64,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                32,
                3,
                kernel_size=1
            ),

            nn.Sigmoid()
        )

    def forward(
        self,
        masked,
        reconstructed,
        layout,
        glyph
    ):

        x = torch.cat(
            [
                masked,
                reconstructed,
                layout,
                glyph
            ],
            dim=1
        )

        # Network learns the RGB appearance
        # that should be used for text pixels.
        text_rgb = self.network(x)

        # HARD GLYPH CONSTRAINT
        #
        # Wherever glyph = 1,
        # use predicted text appearance.
        #
        # Everywhere else,
        # keep reconstructed image.

        glyph_rgb = glyph.expand(
            -1,
            3,
            -1,
            -1
        )

        output = (
            reconstructed * (1 - glyph_rgb)
            + text_rgb * glyph_rgb
        )

        return output


if __name__ == "__main__":

    model = GlyphRenderer()

    masked = torch.randn(
        2, 3, 128, 128
    )

    reconstructed = torch.randn(
        2, 3, 128, 128
    )

    layout = torch.randn(
        2, 1, 128, 128
    )

    glyph = torch.rand(
        2, 1, 128, 128
    )

    output = model(
        masked,
        reconstructed,
        layout,
        glyph
    )

    print("=" * 50)
    print("GLYPH RENDERER CHECK")
    print("=" * 50)

    print(
        "Output:",
        output.shape
    )

    parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Trainable parameters: {parameters:,}"
    )