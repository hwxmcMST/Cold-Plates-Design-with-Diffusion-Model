import math
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


class DiffusionModel:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cuda", denoising_step=1000
                 , ddim_steps=100):
        self.timesteps = timesteps
        self.device = device
        self.ddpm_steps = timesteps
        self.ddim_steps = ddim_steps

        # 定义beta schedule
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1. - self.alpha_bars)

    def add_noise(self, x0, t):
        """向前扩散过程：添加噪声"""
        sqrt_alpha_bar = self.sqrt_alpha_bars[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1, 1)

        noise = torch.randn_like(x0)
        noisy_x = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise

        return noisy_x, noise

    def sample_timesteps(self, n):
        """随机采样时间步"""
        return torch.randint(low=1, high=self.timesteps, size=(n,))

    def set_ddim_timesteps(self, num_inference_steps=50):
        """设置DDIM采样时间步"""
        self.ddim_timesteps = np.linspace(0, self.timesteps - 1, num_inference_steps, dtype=int)[::-1].copy()
        return self.ddim_timesteps

    def ddpm_sample(self, model, conditions, image_size, batch_size=1, channels=1):
        """基于DDPM公式的x0预测采样"""
        model.eval()
        with torch.no_grad():
            x = torch.randn((batch_size, channels, image_size, image_size)).to(self.device)

            for i in tqdm(reversed(range(1, self.ddpm_steps)), desc="DDPM Sampling"):
                # 时间步张量
                t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)

                # 模型预测x0
                predicted_x0 = model(x, t, conditions)

                # # 计算参数
                # alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1, 1)
                # alpha_bar_t_prev = self.alpha_bars[t - 1].view(-1, 1, 1, 1) if i > 1 else torch.ones_like(alpha_bar_t)
                #
                # # 计算后验均值的系数
                # posterior_mean_coef1 = (torch.sqrt(alpha_bar_t_prev) * self.betas[t] / (1 - alpha_bar_t)).view(-1, 1, 1,
                #                                                                                                1)
                # posterior_mean_coef2 = (
                #             torch.sqrt(self.alphas[t - 1]) * (1 - alpha_bar_t_prev) / (1 - alpha_bar_t)).view(-1, 1, 1,
                #                                                                                               1)
                #
                # # 计算均值
                # mean = posterior_mean_coef1 * predicted_x0 + posterior_mean_coef2 * x

                # 当前时间步参数
                alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1, 1)
                beta_t = self.betas[t].view(-1, 1, 1, 1)

                if i > 1:
                    # 前一个时间步参数
                    t_prev = torch.full((batch_size,), i - 1, device=self.device, dtype=torch.long)
                    alpha_bar_t_prev = self.alpha_bars[t_prev].view(-1, 1, 1, 1)
                    alpha_t_prev = self.alphas[t_prev].view(-1, 1, 1, 1)

                    # 均值计算
                    mean = (torch.sqrt(alpha_bar_t_prev) * beta_t / (1 - alpha_bar_t)) * predicted_x0 + \
                           (torch.sqrt(alpha_t_prev) * (1 - alpha_bar_t_prev) / (1 - alpha_bar_t)) * x

                    # 方差和采样
                    variance = beta_t * (1 - alpha_bar_t_prev) / (1 - alpha_bar_t)
                    std = torch.sqrt(variance)
                    noise = torch.randn_like(x)
                    x = mean + std * noise
                else:
                    # 最后一步
                    x = predicted_x0

        return x
        # return torch.clamp(x, -1.0, 1.0)

    def ddim_sample(self, model, conditions, image_size, batch_size=1, channels=1,
                                 eta=0.0):
        """当模型预测x0时的DDIM采样"""
        model.eval()
        with torch.no_grad():
            # 设置DDIM时间步
            if not hasattr(self, 'ddim_timesteps') or self.ddim_timesteps is None:
                self.ddim_timesteps = np.linspace(0, self.timesteps - 1, self.ddim_steps, dtype=int)[::-1].copy()

            timesteps = self.ddim_timesteps

            # 从纯噪声开始
            x = torch.randn((batch_size, channels, image_size, image_size)).to(self.device)

            for i, timestep in enumerate(tqdm(timesteps, desc="DDIM Sampling (x0)")):
                t = torch.full((batch_size,), timestep, device=self.device, dtype=torch.long)

                # 模型直接预测x0
                predicted_x0 = model(x, t, conditions)

                # 当前时间步的参数
                alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1, 1)

                if i < len(timesteps) - 1:
                    # 前一个时间步
                    t_prev = timesteps[i + 1]
                    alpha_bar_t_prev = self.alpha_bars[t_prev].view(-1, 1, 1, 1)

                    # 计算sigma
                    sigma = eta * torch.sqrt((1 - alpha_bar_t_prev) / (1 - alpha_bar_t)) * torch.sqrt(
                        1 - alpha_bar_t / alpha_bar_t_prev)

                    # 预测的方向（指向x0）
                    pred_direction = torch.sqrt(1 - alpha_bar_t_prev - sigma ** 2) * (
                                x - torch.sqrt(alpha_bar_t) * predicted_x0) / torch.sqrt(1 - alpha_bar_t)

                    # 随机噪声
                    noise = torch.randn_like(x) if eta > 0 else torch.zeros_like(x)

                    # 更新x
                    x = torch.sqrt(alpha_bar_t_prev) * predicted_x0 + pred_direction + sigma * noise
                else:
                    # 最后一步，直接使用预测的x0
                    x = predicted_x0

        return torch.clamp(x, -1.0, 1.0)
