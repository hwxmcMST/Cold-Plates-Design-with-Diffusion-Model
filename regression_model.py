import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiModalRegressionModel(nn.Module):
    def __init__(self, image_channels=1, condition_dim=4, num_targets=8, base_channels=64):
        super().__init__()

        # 图像编码器 (CNN)
        self.image_encoder = nn.Sequential(
            # 输入: [batch, 1, 128, 128]
            nn.Conv2d(image_channels, base_channels, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64x64

            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32

            nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16

            nn.Conv2d(base_channels * 4, base_channels * 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8

            nn.AdaptiveAvgPool2d((4, 4)),  # 统一到4x4
            nn.Flatten()
        )

        # 图像特征维度计算: base_channels*8 * 4 * 4 = 512 * 16 = 8192
        image_feature_dim = base_channels * 8 * 4 * 4

        # 条件编码器
        self.condition_encoder = nn.Sequential(
            nn.Linear(condition_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(image_feature_dim + 128, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # 回归头
        self.regression_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_targets)
        )

    def forward(self, image, conditions):
        # 编码图像
        image_features = self.image_encoder(image)

        # 编码条件
        condition_features = self.condition_encoder(conditions)

        # 融合特征
        combined_features = torch.cat([image_features, condition_features], dim=1)
        fused_features = self.fusion(combined_features)

        # 回归预测
        predictions = self.regression_head(fused_features)

        return predictions