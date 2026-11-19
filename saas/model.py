from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class SpatialFourierEmbedding(nn.Module):
    def __init__(self, num_bands=4):
        super().__init__()
        self.frequencies = 2.0 ** torch.arange(num_bands)

    def forward(self, coords):
        scaled = coords.unsqueeze(-1) * self.frequencies.to(coords.device) * np.pi
        sin_f = torch.sin(scaled).view(coords.shape[0], -1)
        cos_f = torch.cos(scaled).view(coords.shape[0], -1)
        return torch.cat([sin_f, cos_f], dim=1)


class SpatialSpectralAttentionPooling(nn.Module):
    def __init__(self, in_channels=256):
        super().__init__()
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, max(8, in_channels // 16), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(8, in_channels // 16), in_channels, 1),
            nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid(),
        )

    def forward(self, x, mask_prior):
        ch_attn = self.channel_gate(x)
        x_ch = x * ch_attn
        avg_out = torch.mean(x_ch, dim=1, keepdim=True)
        max_out, _ = torch.max(x_ch, dim=1, keepdim=True)
        sp_attn = self.spatial_gate(torch.cat([avg_out, max_out], dim=1))
        combined_attn = sp_attn * (0.5 + 0.5 * mask_prior)
        weighted_features = x_ch * combined_attn
        pooled = torch.sum(weighted_features, dim=[2, 3]) / (
            torch.sum(combined_attn, dim=[2, 3]) + 1e-6
        )
        return torch.nan_to_num(pooled, nan=0.0)


class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.GELU(),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
        )
        self.shortcut = nn.Sequential()
        if in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1),
                nn.BatchNorm2d(out_c),
            )

    def forward(self, x):
        return torch.nn.functional.gelu(self.conv(x) + self.shortcut(x))


class OptimizedMultimodalSOCNet(nn.Module):
    def __init__(self, in_channels=12, env_dim=5, target_mean=29.62):
        super().__init__()
        self.input_norm = nn.BatchNorm2d(in_channels)
        self.enc1 = ResidualBlock(in_channels, 64)
        self.enc2 = ResidualBlock(64, 128)
        self.pool = nn.MaxPool2d(2, 2)
        self.bottleneck = ResidualBlock(128, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = ResidualBlock(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = ResidualBlock(128, 64)
        self.seg_head = nn.Conv2d(64, 1, 1)
        self.s2ap_bottleneck = SpatialSpectralAttentionPooling(256)
        self.s2ap_enc2 = SpatialSpectralAttentionPooling(128)
        self.fourier_coord_embed = SpatialFourierEmbedding(4)
        self.reg_head = nn.Sequential(
            nn.Linear(256 + 128 + 16 + env_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        nn.init.constant_(self.reg_head[-1].bias, np.log1p(target_mean))

    def forward(self, x, coords, env_priors):
        x = torch.nan_to_num(x, nan=0.0)
        x = self.input_norm(x)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b = self.bottleneck(self.pool(e2))
        d2 = self.upconv2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.upconv1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        pred_seg_logits = self.seg_head(d1)
        mask_prob = torch.sigmoid(pred_seg_logits)
        b_mask = nn.functional.interpolate(
            mask_prob, size=b.shape[2:], mode="bilinear", align_corners=False
        )
        e2_mask = nn.functional.interpolate(
            mask_prob, size=e2.shape[2:], mode="bilinear", align_corners=False
        )
        feat_b = self.s2ap_bottleneck(b, b_mask)
        feat_e2 = self.s2ap_enc2(e2, e2_mask)
        spatial_embed = self.fourier_coord_embed(coords)
        fusion_vector = torch.cat([feat_b, feat_e2, spatial_embed, env_priors], dim=1)
        pred_soc = self.reg_head(fusion_vector)
        return pred_seg_logits, torch.nan_to_num(pred_soc, nan=0.0)
