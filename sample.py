import argparse
import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import random_split, DataLoader
from torchvision.utils import save_image

from dataset.dataset import ConditionDataset
from model.diffusion import DiffusionModel
from model.unet import ConditionalUNet


def generate_new_data(model_path, conditions_list, image_size=128, num_samples=10):

    """生成新的图像数据"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = ConditionalUNet(image_channels=1, condition_dim=4, base_channels=64)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 初始化扩散过程
    diffusion = DiffusionModel(device=device)

    # 创建数据集实例用于归一化（不需要实际数据，只需要范围信息）
    dataset = ConditionDataset("dummy_path", image_size=image_size)

    generated_images = []

    for i, condition_values in enumerate(conditions_list):
        Re, H, Pr, W = condition_values

        # ✅ 步骤 1: 归一化输入条件
        normalized_conditions = np.array([
            dataset.normalize(Re, 'Re'),
            dataset.normalize(H, 'H'),
            dataset.normalize(Pr, 'Pr'),
            dataset.normalize(W, 'W')
        ], dtype=np.float32)

        conditions = torch.FloatTensor(normalized_conditions).unsqueeze(0).to(device)

        # ✅ 步骤 2: 使用条件扩散模型生成图像
        with torch.no_grad():
            generated_image = diffusion.ddpm_sample(
                model, conditions, image_size=image_size, batch_size=1, channels=1
            )

        # ✅ 步骤 3: 后处理与阈值化
        threshold = 0.5
        generated_image = (generated_image.squeeze(0).cpu() * 0.5 + 0.5).clamp(0, 1)
        generated_mask = (generated_image > threshold).float()
        generated_images.append(generated_mask)

        # ✅ 步骤 4: 用原始物理量命名文件
        Re_str = str(int(Re))
        H_str = str(int(H))
        Pr_str = str(Pr).replace('.', '')  # 10.17 → 1017
        W_str = str(W).replace('.', '').lstrip('0') or '0'
        filename = f"Generated_Re{Re_str}H{H_str}Pr{Pr_str}W{W_str}.png"

        save_image(generated_mask, f"generated/{filename}", normalize=True)
        print(f"✅ Saved {filename}")

    return generated_images


# 验证训练数据
def analyze_training_data():
    dataset = ConditionDataset("./dataset/train", image_size=260)

    # 统计条件分布
    conditions_array = torch.stack(dataset.conditions).numpy()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    param_names = ['Re', 'H', 'Pr', 'W']

    for i, ax in enumerate(axes.flat):
        if i < 4:
            # 反归一化显示原始值
            original_values = [
                dataset.denormalize(val, param_names[i]) for val in conditions_array[:, i]
            ]
            ax.hist(original_values, bins=20, alpha=0.7)
            ax.set_title(f'{param_names[i]} Distribution')
            ax.set_xlabel(param_names[i])
            ax.set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig('condition_distributions.png')
    plt.show()

    return conditions_array


def validate_conditions(conditions_list):
    """验证输入条件是否在有效范围内"""
    valid_ranges = {
        'Re': (200, 400),
        'H': (100, 200),
        'Pr': (6.78, 13.56),
        'W': (0.1, 0.9)
    }

    for i, cond in enumerate(conditions_list):
        Re, H, Pr, W = cond

        if not (valid_ranges['Re'][0] <= Re <= valid_ranges['Re'][1]):
            print(f"条件 {i}: Re={Re} 超出范围 [{valid_ranges['Re'][0]}, {valid_ranges['Re'][1]}]")
            return False

        if not (valid_ranges['H'][0] <= H <= valid_ranges['H'][1]):
            print(f"条件 {i}: H={H} 超出范围 [{valid_ranges['H'][0]}, {valid_ranges['H'][1]}]")
            return False

        if not (valid_ranges['Pr'][0] <= Pr <= valid_ranges['Pr'][1]):
            print(f"条件 {i}: Pr={Pr} 超出范围 [{valid_ranges['Pr'][0]}, {valid_ranges['Pr'][1]}]")
            return False

        if not (valid_ranges['W'][0] <= W <= valid_ranges['W'][1]):
            print(f"条件 {i}: W={W} 超出范围 [{valid_ranges['W'][0]}, {valid_ranges['W'][1]}]")
            return False

    print("所有条件都在有效范围内")
    return True


def example_generation(args):
    """生成示例数据 - 使用实际的条件范围"""
    image_size = 260
    train_ratio = 0.9  # 训练集比例
    val_ratio = 0.1  # 验证集比例

    # 创建数据集和数据加载器
    full_dataset = ConditionDataset("./dataset/train", image_size=image_size)

    # 划分数据集
    dataset_size = len(full_dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = dataset_size - train_size

    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # 固定随机种子确保可重复性
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    # 使用所有数据进行生成
    # val_loader = DataLoader(full_dataset, batch_size=1, shuffle=False, num_workers=4)

    # # 定义要生成的条件（使用原始范围，不是归一化值）
    # target_conditions = [
    #     [350, 150, 10.17, 0.5],  # 中间值
    #     [200, 100, 6.78, 0.1],  # 最小值
    #     [400, 200, 13.56, 0.9],  # 最大值
    #     [300, 120, 8.47, 0.3],  # 自定义值
    #     [380, 180, 12.25, 0.7],  # 自定义值
    # ]

    # 创建输出目录
    generated_images = []
    os.makedirs("generated", exist_ok=True)
    model_path = os.path.join(args.log_dir, args.model_dir)
    # 收集真实图像
    for i, (images, conditions) in enumerate(val_loader):
        # target_conditions = torch.cat(conditions, dim=0)
        target_conditions = conditions
        # 生成图像
        generated_image = generate_new_data(
            model_path,
            target_conditions,
            image_size=260,
            num_samples=len(target_conditions)
        )
        generated_images.append(generated_image)

    print(f"生成了 {len(generated_images)} 张新图像")
    return generated_images, target_conditions


def generate_grid_conditions(re_steps=6, h_steps=6, pr_steps=6, w_steps=6):
    """生成网格化的条件用于系统性的数据扩充"""
    re_values = np.linspace(200, 400, re_steps)
    h_values = np.linspace(100, 200, h_steps)
    pr_values = np.round(np.linspace(6.78, 13.56, pr_steps), 2)  # ✅ 保留2位小数
    w_values = np.round(np.linspace(0.1, 0.9, w_steps), 2) 

    grid_conditions = []

    for re in re_values:
        for h in h_values:
            for pr in pr_values:
                for w in w_values:
                    grid_conditions.append([re, h, pr, w])

    print(f"生成了 {len(grid_conditions)} 个网格条件")
    return grid_conditions



if __name__ == "__main__":

    #parser = argparse.ArgumentParser()
    #parser.add_argument('--data_path',    type=str,     default='./dataset')
    #parser.add_argument('--batch_size',   type=int,     default=16)
    #parser.add_argument('--img_size',     type=tuple,   default=(260, 260))
    #parser.add_argument('--num_epoch',    type=int,     default=100)
    #parser.add_argument('--lr',           type=float,   default=1e-3)
    #parser.add_argument('--manual_seed',  type=float,   default=0)
    #parser.add_argument('--noise_steps',  type=int,     default=1000)
    #parser.add_argument('--ddim_steps',   type=int,     default=100)
    #parser.add_argument('--log_dir',      type=str,     default='./logs')
    #parser.add_argument('--model_dir',    type=str,     default='conditional_diffusion.pth')

    #args = parser.parse_args()

    # # 验证示例条件: Re H Pr W
    # example_conditions = [
    #     [350, 150, 10.17, 0.5],
    #     [250, 180, 7.5, 0.2]
    # ]
    # validate_conditions(example_conditions)
    #
    # # 生成网格条件
    # grid_conditions = generate_grid_conditions(2, 2, 2, 2)  # 2x2x2x2网格
    # print("网格条件示例:", grid_conditions[:5])

    # 生成新数据
    #generated_images, used_conditions = example_generation(args)
        # generated_images, used_conditions = example_generation(args)
    # new_conditions = generate_grid_conditions(re_steps=2, h_steps=2, pr_steps=2, w_steps=2)

    # ✅ Step 2: validate if within valid physical range
    # validate_conditions(new_conditions)

    # ✅ Step 3: generate new images conditioned on these values
    example_conditions =  generate_grid_conditions(re_steps=6, h_steps=6, pr_steps=6, w_steps=6)
    model_path = r"C:\Users\Administrator\Desktop\project-10.8\logs\conditional_diffusion.pth"
    generated_images = generate_new_data(
        model_path=model_path,
        conditions_list=example_conditions,
        image_size=260,
        num_samples=len(example_conditions)
    )