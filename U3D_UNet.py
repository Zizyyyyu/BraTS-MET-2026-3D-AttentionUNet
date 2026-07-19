import torch
import torch.nn as nn


class ConvBlock3d(nn.Module):
    """ConvBlock3d: double 3D convolution with instance norm and ReLU activation."""

    def __init__(self, in_channels: int, out_channels: int):
        """Initialize ConvBlock3d with in_channels and out_channels."""
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor):
        """Apply double Conv3d->InstanceNorm->ReLU blocks to input x."""
        return self.block(x)


class AttentionBlock(nn.Module):
    """AttentionBlock: attention gating for U-Net skip connections."""

    def __init__(self, encoder_channels: int, decoder_channels: int, hidden_channels: int):
        """Initialize AttentionBlock with encoder, decoder, and hidden channel sizes."""
        super().__init__()
        self.encoder_transform = nn.Sequential(
            nn.Conv3d(in_channels=encoder_channels, out_channels=hidden_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.InstanceNorm3d(hidden_channels)
        )
        self.decoder_transform = nn.Sequential(
            nn.Conv3d(in_channels=decoder_channels, out_channels=hidden_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.InstanceNorm3d(hidden_channels)
        )
        self.relu = nn.ReLU(inplace=True)
        self.attention_map = nn.Sequential(
            nn.Conv3d(in_channels=hidden_channels, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, encoder_features: torch.Tensor, decoder_features: torch.Tensor):
        """Compute attention weights and apply to encoder features."""
        encoder = self.encoder_transform(encoder_features)
        decoder = self.decoder_transform(decoder_features)
        hidden = self.relu(encoder + decoder)
        attention = self.attention_map(hidden)
        filtered_encoder = encoder_features * attention
        return filtered_encoder


class UpBlock3D(nn.Module):
    """UpBlock3D: upsampling with attention gating and convolution."""

    def __init__(self, encoder_channels: int, decoder_channels: int, out_channels: int):
        """Initialize UpBlock3D with encoder, decoder, and output channels."""
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels=decoder_channels, out_channels=out_channels, kernel_size=2, stride=2)
        self.attention = AttentionBlock(encoder_channels=encoder_channels, decoder_channels=out_channels, hidden_channels=encoder_channels // 2)
        self.conv = ConvBlock3d(in_channels=encoder_channels + out_channels, out_channels=out_channels)

    def forward(self, encoder_features: torch.Tensor, decoder_features: torch.Tensor):
        """Upsample decoder, apply attention, concatenate, and convolve."""
        decoder = self.up(decoder_features)
        filtered_encoder = self.attention(encoder_features, decoder)
        merged = torch.cat([filtered_encoder, decoder], dim=1)
        output = self.conv(merged)
        return output


class AttentionUNet(nn.Module):
    """AttentionUNet: 3D U-Net with attention gates for multi-class segmentation."""

    def __init__(self, in_channels: int = 4, num_classes: int = 5, base_channels: int = 16):
        """Initialize AttentionUNet with input channels, output classes, and base channel count."""
        super().__init__()
        self.encoder1 = ConvBlock3d(in_channels=in_channels, out_channels=base_channels)
        self.encoder2 = ConvBlock3d(in_channels=base_channels, out_channels=base_channels * 2)
        self.encoder3 = ConvBlock3d(in_channels=base_channels * 2, out_channels=base_channels * 4)
        self.encoder4 = ConvBlock3d(in_channels=base_channels * 4, out_channels=base_channels * 8)
        self.down_sample = nn.MaxPool3d(kernel_size=2, stride=2)
        self.bottleneck = ConvBlock3d(in_channels=base_channels * 8, out_channels=base_channels * 16)
        self.decoder4 = UpBlock3D(encoder_channels=base_channels * 8, decoder_channels=base_channels * 16, out_channels=base_channels * 8)
        self.decoder3 = UpBlock3D(encoder_channels=base_channels * 4, decoder_channels=base_channels * 8, out_channels=base_channels * 4)
        self.decoder2 = UpBlock3D(encoder_channels=base_channels * 2, decoder_channels=base_channels * 4, out_channels=base_channels * 2)
        self.decoder1 = UpBlock3D(encoder_channels=base_channels, decoder_channels=base_channels * 2, out_channels=base_channels)
        self.output_conv = nn.Conv3d(in_channels=base_channels, out_channels=num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor):
        """Forward pass through encoder, bottleneck, and decoder with attention gates."""
        encoder1 = self.encoder1(x)
        encoder2 = self.encoder2(self.down_sample(encoder1))
        encoder3 = self.encoder3(self.down_sample(encoder2))
        encoder4 = self.encoder4(self.down_sample(encoder3))
        bottleneck = self.bottleneck(self.down_sample(encoder4))
        decoder4 = self.decoder4(encoder4, bottleneck)
        decoder3 = self.decoder3(encoder3, decoder4)
        decoder2 = self.decoder2(encoder2, decoder3)
        decoder1 = self.decoder1(encoder1, decoder2)
        logits = self.output_conv(decoder1)
        return logits
