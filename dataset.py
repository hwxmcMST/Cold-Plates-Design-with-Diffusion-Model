import os
import re

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob


class ConditionDataset(Dataset):
    def __init__(self, image_dir, transform=None, image_size=260):
        self.image_dir = image_dir
        self.image_paths = glob.glob(os.path.join(image_dir, "*.png")) + \
                           glob.glob(os.path.join(image_dir, "*.jpg")) + \
                           glob.glob(os.path.join(image_dir, "*.jpeg"))

        self.transform = transform or transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        # 定义条件范围
        self.condition_ranges = {
            'Re': (200, 400),  # Reynolds number range
            'H': (100, 200),  # Height/Size parameter range
            'Pr': (6.78, 13.56),  # Prandtl number range
            'W': (0.1, 0.9)  # Width/Other parameter range
        }

        # 提取所有条件
        self.conditions = []
        for path in self.image_paths:
            filename = os.path.basename(path).split('.')[0]
            conditions = self.extract_conditions(filename)
            self.conditions.append(conditions)

    def extract_conditions(self, filename):
        """从文件名中提取条件参数（支持 Generated_ 前缀 和 Pr=1122 → 11.22 格式）"""
        patterns = [
            r'(?:Generated_)?Re(\d+)H(\d+)Pr([\d.]+)W([\d.]+)',                # e.g. Generated_Re200H100Pr9W1
            r'(?:Generated_)?Re(\d+\.?\d*)H(\d+\.?\d*)Pr(\d+\.?\d*)W(\d+\.?\d*)',  # e.g. Generated_Re200.0H150.0Pr6.78W0.35
            r'(?:Generated_)?Re_(\d+\.?\d*)_H_(\d+\.?\d*)_Pr_(\d+\.?\d*)_W_(\d+\.?\d*)'  # e.g. Re_200_H_150_Pr_6.78_W_0.35
        ]

        for pattern in patterns:
            match = re.match(pattern, filename)
            if match:
                Re = float(match.group(1))
                H = float(match.group(2))
                Pr = float(match.group(3))
                W = float(match.group(4))

                # 处理Pr的无小数格式（如1122表示11.22）
                if Pr > 100:
                    if Pr > 999:  # e.g. 1122 → 11.22
                        Pr = Pr / 100.0
                    else:         # e.g. 678 → 6.78
                        Pr = Pr / 100.0

                # 归一化
                conditions = np.array([
                    self.normalize(Re, 'Re'),
                    self.normalize(H, 'H'),
                    self.normalize(Pr, 'Pr'),
                    self.normalize(W, 'W')
                ], dtype=np.float32)

                # Debug 打印
                # print(f"[DEBUG] Parsed {filename} -> Re={Re}, H={H}, Pr={Pr}, W={W}")

                return torch.FloatTensor(conditions)

        raise ValueError(f"❌ 无法解析文件名: {filename}")


    def normalize(self, value, param_type):
        """将参数值归一化到[0,1]范围"""
        min_val, max_val = self.condition_ranges[param_type]
        return (value - min_val) / (max_val - min_val)

    def denormalize(self, normalized_value, param_type):
        """将归一化值反归一化回原始范围"""
        min_val, max_val = self.condition_ranges[param_type]
        return normalized_value * (max_val - min_val) + min_val

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        conditions = self.conditions[idx]

        # 加载图像
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, conditions


class MultiModalDataset(Dataset):
    def __init__(self, data_dir, gen_data_dir, transform=None, image_size=128, table_ext=".csv",
                 normalizer=None, fit_normalizer=False):
        self.data_dir = data_dir
        self.gen_data_dir = gen_data_dir
        self.transform = transform
        self.image_size = image_size
        self.table_ext = table_ext
        self.normalizer = normalizer

        # 定义条件范围
        self.condition_ranges = {
            'Re': (200, 400),  # Reynolds number range
            'H': (100, 200),  # Height/Size parameter range
            'Pr': (6.78, 13.56),  # Prandtl number range
            'W': (0.1, 0.9)  # Width/Other parameter range
        }

        # 获取生成数据集中所有图片文件
        self.image_files = []
        for ext in ['.png', '.jpg', '.jpeg']:
            self.image_files.extend([f for f in os.listdir(gen_data_dir) if f.endswith(ext)])

        self.samples = []
        all_targets_list = []

        for img_file in self.image_files:
            # 在原始数据集中获取对应的表格文件路径
            raw_name = img_file.split('_', 1)[-1]
            table_file = os.path.splitext(raw_name)[0] + self.table_ext
            table_path = os.path.join(data_dir, table_file)

            if not os.path.exists(table_path):
                print(f"警告: 找不到表格文件 {table_path}")
                continue

            # 提取条件
            conditions = self.extract_conditions(raw_name)

            # 读取表格数据
            targets_dict = self.read_table_data(table_path)

            if targets_dict is not None:
                # 转换为tensor
                targets_tensor = torch.FloatTensor([
                    targets_dict['Thermal Resistance (K/W)'],
                    targets_dict['Nusselt Number'],
                    targets_dict['Pressure Drop (Pa)'],
                    targets_dict['Average Surface Temperature (K)'],
                    targets_dict['Maximum Surface Temperature (K)'],
                    targets_dict['Effective Heat Transfer Area m^2'],
                    targets_dict['Average Outlet Temperature (K)'],
                    targets_dict['Maximum Outlet Temperature (K)']
                ])

                # 收集原始目标值用于归一化
                all_targets_list.append(targets_tensor.numpy())

                self.samples.append({
                    'image_file': img_file,
                    'conditions': conditions,
                    'targets': targets_tensor,
                    'targets_raw': targets_tensor.clone()  # 保存原始值
                })

        print(f"成功加载 {len(self.samples)} 个样本")
        # 如果需要拟合归一化器
        if fit_normalizer and self.normalizer and len(all_targets_list) > 0:
            all_targets_array = np.array(all_targets_list)
            self.normalizer.fit(all_targets_array)

            # 对样本进行归一化
            for sample in self.samples:
                targets_raw = sample['targets_raw'].numpy().reshape(1, -1)
                targets_normalized = self.normalizer.transform(targets_raw)
                sample['targets'] = torch.FloatTensor(targets_normalized.flatten())

    def extract_conditions(self, filename):
        """从文件名中提取条件参数"""
        # 匹配模式: Re350H250Pr678W1 (注意Pr可能是小数)
        pattern = r'(?:Generated_)?Re(\d+)H(\d+)Pr([\d.]+)W([\d.]+)'
        match = re.match(pattern, filename)

        if match:
            Re = int(match.group(1))  # Reynolds number
            H = int(match.group(2))  # Height/Size parameter
            Pr = float(match.group(3))  # Prandtl number (可能是6.78或678格式)
            W = float(match.group(4))  # Width/Other parameter

            # 处理Pr可能的不同格式（6.78或678）
            if Pr > 100:  # 如果是678格式，转换为6.78
                Pr = Pr / 100.0

            # 根据实际范围归一化条件
            conditions = np.array([
                self.normalize(Re, 'Re'),
                self.normalize(H, 'H'),
                self.normalize(Pr, 'Pr'),
                self.normalize(W, 'W')
            ], dtype=np.float32)

            return torch.FloatTensor(conditions)
        else:
            # 尝试其他可能的格式
            patterns = [
                r'(?:Generated_)?Re(\d+\.?\d*)H(\d+\.?\d*)Pr(\d+\.?\d*)W(\d+\.?\d*)',
                r'(?:Generated_)?Re_(\d+\.?\d*)_H_(\d+\.?\d*)_Pr_(\d+\.?\d*)_W_(\d+\.?\d*)'
            ]
            for pattern in patterns:
                match = re.match(pattern, filename)
                if match:
                    Re = float(match.group(1))
                    H = float(match.group(2))
                    Pr = float(match.group(3))
                    W = float(match.group(4))

                    # 处理Pr格式
                    if Pr > 100:
                        Pr = Pr / 100.0

                    conditions = np.array([
                        self.normalize(Re, 'Re'),
                        self.normalize(H, 'H'),
                        self.normalize(Pr, 'Pr'),
                        self.normalize(W, 'W')
                    ], dtype=np.float32)

                    return torch.FloatTensor(conditions)

            raise ValueError(f"无法解析文件名: {filename}")

    def normalize(self, value, param_type):
        """将参数值归一化到[0,1]范围"""
        min_val, max_val = self.condition_ranges[param_type]
        return (value - min_val) / (max_val - min_val)

    def denormalize(self, normalized_value, param_type):
        """将归一化值反归一化回原始范围"""
        min_val, max_val = self.condition_ranges[param_type]
        return normalized_value * (max_val - min_val) + min_val

    def read_table_data(self, table_path):
        """读取表格文件数据"""
        try:
            # 根据文件格式读取数据
            if table_path.endswith('.txt'):
                # 处理文本格式的表格数据
                return self.read_txt_table(table_path)
            elif table_path.endswith('.csv'):
                # 处理CSV格式
                return self.read_csv_table(table_path)
            else:
                print(f"不支持的表格文件格式: {table_path}")
                return None
        except Exception as e:
            print(f"读取表格文件 {table_path} 时出错: {e}")
            return None

    def read_txt_table(self, table_path):
        """读取文本格式的表格数据"""
        targets = {}

        with open(table_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line or ':' not in line:
                continue

            # 解析键值对
            parts = line.split(':', 1)
            if len(parts) != 2:
                continue

            key = parts[0].strip()
            value = parts[1].strip()

            # 提取数值（处理科学计数法）
            try:
                # 移除可能的单位和其他非数字字符
                numeric_value = re.search(r'[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?', value)
                if numeric_value:
                    num = float(numeric_value.group())
                    targets[key] = num
            except ValueError:
                continue

        # 确保所有必需的字段都存在
        required_fields = [
            'Thermal Resistance (K/W)',
            'Nusselt Number',
            'Pressure Drop (Pa)',
            'Average Surface Temperature (K)',
            'Maximum Surface Temperature (K)',
            'Effective Heat Transfer Area m^2',
            'Average Outlet Temperature (K)',
            'Maximum Outlet Temperature (K)'
        ]

        for field in required_fields:
            if field not in targets:
                print(f"警告: 表格文件 {table_path} 缺少字段 {field}")
                return None

        return targets

    def read_csv_table(self, table_path):
        """读取CSV格式的表格数据 - 修正版"""
        try:
            # 读取所有行
            with open(table_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # 找到数据部分的开始（跳过注释行）
            data_start = 0
            for i, line in enumerate(lines):
                if line.strip() and not line.startswith('%'):
                    # 检查是否是表头行
                    if 'Thermal Resistance' in line or 'Temperature' in line:
                        data_start = i
                        break

            targets = {}

            # 从第8行开始解析（索引7，因为从0开始）
            for i in range(max(7, data_start), min(len(lines), 16)):  # 第8-15行
                line = lines[i].strip()
                if not line or line.startswith('%'):
                    continue

                # 分割行
                parts = [p.strip() for p in line.split(',') if p.strip()]

                if len(parts) >= 2:
                    key = parts[0]
                    value_str = parts[1]

                    # 处理特征名称可能的变体
                    feature_map = {
                        'Thermal Resistance (K/W)': 'Thermal Resistance (K/W)',
                        'Thermal Resistance': 'Thermal Resistance (K/W)',
                        'Nusselt Number': 'Nusselt Number',
                        'Pressure Drop (Pa)': 'Pressure Drop (Pa)',
                        'Pressure Drop': 'Pressure Drop (Pa)',
                        'Average Surface Temperature (K)': 'Average Surface Temperature (K)',
                        'Average Surface Temperature': 'Average Surface Temperature (K)',
                        'Maximum Surface Temperature (K)': 'Maximum Surface Temperature (K)',
                        'Maximum Surface Temperature': 'Maximum Surface Temperature (K)',
                        'Effective Heat Transfer Area m^2': 'Effective Heat Transfer Area m^2',
                        'Effective Heat Transfer Area': 'Effective Heat Transfer Area m^2',
                        'Average Outlet Temperature (K)': 'Average Outlet Temperature (K)',
                        'Average Outlet Temperature': 'Average Outlet Temperature (K)',
                        'Maximum Outlet Temperature (K)': 'Maximum Outlet Temperature (K)',
                        'Maximum Outlet Temperature': 'Maximum Outlet Temperature (K)'
                    }

                    normalized_key = feature_map.get(key, key)

                    if normalized_key in [
                        'Thermal Resistance (K/W)',
                        'Nusselt Number',
                        'Pressure Drop (Pa)',
                        'Average Surface Temperature (K)',
                        'Maximum Surface Temperature (K)',
                        'Effective Heat Transfer Area m^2',
                        'Average Outlet Temperature (K)',
                        'Maximum Outlet Temperature (K)'
                    ]:
                        try:
                            # 处理科学计数法
                            if 'E' in value_str or 'e' in value_str:
                                value = float(value_str)
                            else:
                                value = float(value_str)

                            targets[normalized_key] = value
                        except ValueError as e:
                            print(f"解析数值错误: '{value_str}' 对于特征 '{key}': {e}")

            # 验证是否获取了所有必需特征
            required_features = [
                'Thermal Resistance (K/W)',
                'Nusselt Number',
                'Pressure Drop (Pa)',
                'Average Surface Temperature (K)',
                'Maximum Surface Temperature (K)',
                'Effective Heat Transfer Area m^2',
                'Average Outlet Temperature (K)',
                'Maximum Outlet Temperature (K)'
            ]

            missing = [f for f in required_features if f not in targets]
            if missing:
                print(f"警告: 表格文件 {table_path} 缺少特征: {missing}")
                return None

            return targets

        except Exception as e:
            print(f"读取CSV表格文件 {table_path} 时出错: {e}")
            return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 加载图像
        image_path = os.path.join(self.gen_data_dir, sample['image_file'])
        image = Image.open(image_path).convert('L')  # 灰度图

        if self.transform:
            image = self.transform(image)
        else:
            # 默认转换
            transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
            image = transform(image)

        return {
            'image': image,
            'conditions': sample['conditions'],
            'targets': sample['targets'],
            'targets_raw': sample['targets_raw'],  # 原始值用于反归一化
            'filename': sample['image_file']
        }


# 测试数据集
def test_dataset():
    dataset = ConditionDataset("C:/Users/Administrator/Desktop/project-10.8/dataset/train", image_size=260)
    print(f"数据集大小: {len(dataset)}")

    image, conditions = dataset[0]
    print(f"图像形状: {image.shape}")
    print(f"条件形状: {conditions.shape}")
    print(f"条件值: {conditions}")

    return dataset

test_dataset()