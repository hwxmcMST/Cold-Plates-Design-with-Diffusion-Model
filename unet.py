import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ConditionalUNet(nn.Module):
    def __init__(self, image_channels=1, condition_dim=4, base_channels=64):
        super().__init__()

        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, base_channels * 4),
            nn.GELU(),
            nn.Linear(base_channels * 4, base_channels * 4)
        )

        self.condition_proj = nn.Sequential(
            nn.Linear(condition_dim, base_channels * 4),
            nn.GELU(),
            nn.Linear(base_channels * 4, base_channels * 4)
        )

        self.condition_adapter = nn.Sequential(
            nn.Linear(base_channels * 4, base_channels * 8),
            nn.GELU()
        )

        # 编码器
        self.enc1 = self._block(image_channels, base_channels)  # 64
        self.enc2 = self._block(base_channels, base_channels * 2)  # 128
        self.enc3 = self._block(base_channels * 2, base_channels * 4)  # 256
        self.enc4 = self._block(base_channels * 4, base_channels * 8)  # 512

        # 瓶颈
        self.bottleneck = self._block(base_channels * 8, base_channels * 8)  # 512

        # self.img_cond_encoder = nn.Sequential(
        #     self._block(image_channels, base_channels),
        #     nn.MaxPool2d(2),
        #     self._block(base_channels, base_channels * 2),
        #     nn.MaxPool2d(2),
        #     self._block(base_channels * 2, base_channels * 4),
        #     nn.MaxPool2d(2),
        #     self._block(base_channels * 4, base_channels * 8),
        #     nn.MaxPool2d(2),
        # )

        # 解码器
        self.up1 = nn.ConvTranspose2d(base_channels * 8, base_channels * 8, kernel_size=2, stride=2)
        self.dec1 = self._block(base_channels * 16, base_channels * 8)  # 512 -> 512

        self.up2 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec2 = self._block(base_channels * 8, base_channels * 4)  # 512 -> 256

        self.up3 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec3 = self._block(base_channels * 4, base_channels * 2)  # 256 -> 128

        self.up4 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec4 = self._block(base_channels * 2, base_channels)  # 128 -> 64

        self.output = nn.Conv2d(base_channels, image_channels, kernel_size=1)

        # 条件适配器
        self.condition_adapter = nn.Sequential(
            nn.Linear(base_channels * 4, base_channels * 8),
            nn.GELU()
        )

        # 下采样层
        self.pool = nn.MaxPool2d(2)

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
        )

    def forward(self, x, time, conditions):
        # 时间嵌入
        t_embed = self.time_embed(time)

        # 条件嵌入
        c_embed = self.condition_proj(conditions)

        # 融合时间和条件
        condition_embed = t_embed + c_embed

        # 编码器路径
        e1 = self.enc1(x)  # [B, 64, H, W]
        e2 = self.enc2(self.pool(e1))  # [B, 128, H/2, W/2]
        e3 = self.enc3(self.pool(e2))  # [B, 256, H/4, W/4]
        e4 = self.enc4(self.pool(e3))  # [B, 512, H/8, W/8]

        # 瓶颈
        bottleneck = self.bottleneck(self.pool(e4))  # [B, 512, H/16, W/16]

        # 将条件嵌入添加到瓶颈
        b, c, h, w = bottleneck.shape
        adapted_condition = self.condition_adapter(condition_embed)
        adapted_condition = adapted_condition.view(b, -1, 1, 1).expand(-1, -1, h, w)
        bottleneck = bottleneck + adapted_condition

        # 解码器路径
        d1 = self.up1(bottleneck)  # [B, 512, H/8, W/8]
        # 检查尺寸并调整
        if d1.shape[2:] != e4.shape[2:]:
            d1 = F.interpolate(d1, size=e4.shape[2:], mode='bilinear', align_corners=False)
        d1 = torch.cat([d1, e4], dim=1)  # [B, 1024, H/8, W/8]
        d1 = self.dec1(d1)  # [B, 512, H/8, W/8]

        d2 = self.up2(d1)  # [B, 256, H/4, W/4]
        if d2.shape[2:] != e3.shape[2:]:
            d2 = F.interpolate(d2, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, e3], dim=1)  # [B, 512, H/4, W/4]
        d2 = self.dec2(d2)  # [B, 256, H/4, W/4]

        d3 = self.up3(d2)  # [B, 128, H/2, W/2]
        if d3.shape[2:] != e2.shape[2:]:
            d3 = F.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d3 = torch.cat([d3, e2], dim=1)  # [B, 256, H/2, W/2]
        d3 = self.dec3(d3)  # [B, 128, H/2, W/2]

        d4 = self.up4(d3)  # [B, 64, H, W]
        if d4.shape[2:] != e1.shape[2:]:
            d4 = F.interpolate(d4, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d4 = torch.cat([d4, e1], dim=1)  # [B, 128, H, W]
        d4 = self.dec4(d4)  # [B, 64, H, W]

        return self.output(d4)