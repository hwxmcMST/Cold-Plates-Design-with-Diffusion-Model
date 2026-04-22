import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import models
from scipy import linalg
from piq import ssim, FID
import torch.nn.functional as F
from matplotlib import pyplot as plt


def calculate_fid(real_images, generated_images, device, dims=2048):
    """计算FID分数"""
    # 加载InceptionV3模型用于特征提取
    inception_model = models.inception_v3(pretrained=True, transform_input=False)
    inception_model.fc = nn.Identity()  # 移除最后的分类层
    inception_model = inception_model.to(device)
    inception_model.eval()

    def get_inception_features(images):
        """从图像中提取Inception特征"""
        with torch.no_grad():
            # 如果图像是单通道，复制为3通道
            if images.shape[1] == 1:
                images = images.repeat(1, 3, 1, 1)

            # 调整图像大小到299x299（InceptionV3的输入大小）
            if images.shape[2] != 299 or images.shape[3] != 299:
                images = nn.functional.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)

            # 前向传播获取特征
            features = inception_model(images)
            return features.cpu().numpy()

    # 提取特征
    real_features = get_inception_features(real_images)
    generated_features = get_inception_features(generated_images)

    # 计算均值和协方差
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = generated_features.mean(axis=0), np.cov(generated_features, rowvar=False)

    # 计算Fréchet距离
    fid_value = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid_value


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """计算Fréchet距离"""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2

    # 乘积可能会是奇异的
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # 数值误差可能产生虚部
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


def calculate_metrics(real_images, generated_images, generated_mask, device):
    """计算图像质量指标"""
    metrics = {}

    # 确保图像在正确的设备上
    real_images = real_images.to(device)
    generated_images = generated_images.to(device)
    generated_mask = generated_mask.to(device)

    # 计算SSIM
    try:
        ssim_value = ssim(real_images, generated_mask, data_range=1.0)
        metrics['ssim'] = ssim_value.item()
    except Exception as e:
        print(f"SSIM计算错误: {e}")
        metrics['ssim'] = 0.0

    # 计算FID
    try:
        fid_value = calculate_fid(real_images, generated_images, device)
        metrics['fid'] = fid_value.item()
    except Exception as e:
        print(f"FID计算错误: {e}")
        metrics['fid'] = float('inf')

    # 计算PSNR
    try:
        metrics['psnr'] = calculate_psnr(real_images, generated_images)
    except Exception as e:
        print(f"PSNR计算错误: {e}")
        metrics['psnr'] = 0.0

    return metrics


def save_comparison_images(real_images, generated_images, epoch):
    """保存真实图像和生成图像的对比"""
    os.makedirs("comparison_results", exist_ok=True)

    fig, axes = plt.subplots(2, len(real_images), figsize=(15, 6))
    if len(real_images) == 1:
        axes = axes.reshape(2, 1)

    for i in range(len(real_images)):
        # 真实图像
        axes[0, i].imshow(real_images[i].squeeze(), cmap='gray')
        axes[0, i].set_title(f"Real_{i}")
        axes[0, i].axis('off')

        # 生成图像
        axes[1, i].imshow(generated_images[i].squeeze(), cmap='gray')
        axes[1, i].set_title(f"Gen_{i}")
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig(f"comparison_results/epoch_{epoch}_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()


def calculate_psnr(real_images, generated_images):
    """计算PSNR"""
    mse = F.mse_loss(real_images, generated_images)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()


import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import json


class FeatureNormalizer:
    def __init__(self, normalization_method='standard'):
        """
        特征归一化器
        Args:
            normalization_method: 'standard' (标准化) 或 'minmax' (最小最大归一化)
        """
        self.normalization_method = normalization_method
        self.feature_stats = {}
        self.scalers = {}
        self.is_fitted = False

        # 定义特征名称
        self.feature_names = [
            'Thermal Resistance (K/W)',
            'Nusselt Number',
            'Pressure Drop (Pa)',
            'Average Surface Temperature (K)',
            'Maximum Surface Temperature (K)',
            'Effective Heat Transfer Area m^2',
            'Average Outlet Temperature (K)',
            'Maximum Outlet Temperature (K)'
        ]

    def fit(self, all_targets):
        """拟合归一化参数"""
        print("正在计算特征统计信息...")

        for i, feature_name in enumerate(self.feature_names):
            feature_data = all_targets[:, i]

            # 计算统计信息
            stats = {
                'min': float(np.min(feature_data)),
                'max': float(np.max(feature_data)),
                'mean': float(np.mean(feature_data)),
                'std': float(np.std(feature_data)),
                'range': float(np.max(feature_data) - np.min(feature_data))
            }
            self.feature_stats[feature_name] = stats

            # 初始化归一化器
            if self.normalization_method == 'standard':
                scaler = StandardScaler()
            else:  # minmax
                scaler = MinMaxScaler()

            # 拟合归一化器
            scaler.fit(feature_data.reshape(-1, 1))
            self.scalers[feature_name] = scaler

            print(f"{feature_name}:")
            print(f"  范围: [{stats['min']:.6e}, {stats['max']:.6e}]")
            print(f"  均值: {stats['mean']:.6e}, 标准差: {stats['std']:.6e}")
            print(f"  数据范围: {stats['range']:.6e}")

        self.is_fitted = True
        print("特征归一化参数计算完成！")

    def transform(self, targets):
        """转换目标值"""
        if not self.is_fitted:
            raise ValueError("归一化器尚未拟合，请先调用fit方法")

        normalized_targets = np.zeros_like(targets)

        for i, feature_name in enumerate(self.feature_names):
            feature_data = targets[:, i].reshape(-1, 1)
            normalized_data = self.scalers[feature_name].transform(feature_data)
            normalized_targets[:, i] = normalized_data.flatten()

        return normalized_targets

    def inverse_transform(self, normalized_targets):
        """反转换归一化值"""
        if not self.is_fitted:
            raise ValueError("归一化器尚未拟合，请先调用fit方法")

        original_targets = np.zeros_like(normalized_targets)

        for i, feature_name in enumerate(self.feature_names):
            normalized_data = normalized_targets[:, i].reshape(-1, 1)
            original_data = self.scalers[feature_name].inverse_transform(normalized_data)
            original_targets[:, i] = original_data.flatten()

        return original_targets

    def save(self, filepath):
        """保存归一化参数"""
        if not self.is_fitted:
            raise ValueError("归一化器尚未拟合")

        save_data = {
            'normalization_method': self.normalization_method,
            'feature_stats': self.feature_stats,
            'scalers': {}
        }

        # 保存scaler参数
        for feature_name, scaler in self.scalers.items():
            if hasattr(scaler, 'mean_'):
                save_data['scalers'][feature_name] = {
                    'type': 'standard',
                    'mean': scaler.mean_.tolist(),
                    'scale': scaler.scale_.tolist()
                }
            elif hasattr(scaler, 'data_min_'):
                save_data['scalers'][feature_name] = {
                    'type': 'minmax',
                    'data_min': scaler.data_min_.tolist(),
                    'data_max': scaler.data_max_.tolist(),
                    'data_range': scaler.data_range_.tolist()
                }

        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)

        print(f"归一化参数已保存到: {filepath}")

    def load(self, filepath):
        """加载归一化参数"""
        with open(filepath, 'r') as f:
            save_data = json.load(f)

        self.normalization_method = save_data['normalization_method']
        self.feature_stats = save_data['feature_stats']
        self.scalers = {}

        for feature_name, scaler_data in save_data['scalers'].items():
            if scaler_data['type'] == 'standard':
                scaler = StandardScaler()
                scaler.mean_ = np.array(scaler_data['mean'])
                scaler.scale_ = np.array(scaler_data['scale'])
                scaler.var_ = scaler.scale_ ** 2
                scaler.n_features_in_ = 1
            else:  # minmax
                scaler = MinMaxScaler()
                scaler.data_min_ = np.array(scaler_data['data_min'])
                scaler.data_max_ = np.array(scaler_data['data_max'])
                scaler.data_range_ = np.array(scaler_data['data_range'])
                scaler.n_features_in_ = 1

            self.scalers[feature_name] = scaler

        self.is_fitted = True
        print(f"归一化参数已从 {filepath} 加载")