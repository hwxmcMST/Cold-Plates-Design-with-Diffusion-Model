
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr
from torchvision import transforms

from dataset.dataset import MultiModalDataset
from model.regression_model import MultiModalRegressionModel
from tools import FeatureNormalizer


def train_regression_model(model, train_loader, val_loader, normalizer,
                           optimizer, scheduler, device, epochs=100):
    """训练回归模型"""
    model.to(device)
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    # 特征名称
    feature_names = [
        'thermal_resistance', 'nusselt_number', 'pressure_drop',
        'avg_surface_temp', 'max_surface_temp', 'heat_transfer_area',
        'avg_outlet_temp', 'max_outlet_temp'
    ]

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for batch in train_loader:
            images = batch['image'].to(device)
            conditions = batch['conditions'].to(device)
            targets = batch['targets'].to(device)

            optimizer.zero_grad()
            predictions = model(images, conditions)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # 验证阶段
        model.eval()
        val_loss = 0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                conditions = batch['conditions'].to(device)
                targets = batch['targets'].to(device)

                predictions = model(images, conditions)
                loss = criterion(predictions, targets)
                val_loss += loss.item()

                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # 计算验证集指标
        all_predictions = np.vstack(all_predictions)
        all_targets = np.vstack(all_targets)

        val_metrics = calculate_regression_metrics(all_targets, all_predictions, normalizer, feature_names)

        # 更新学习率
        if scheduler:
            scheduler.step()

        print(f'Epoch {epoch +1}/{epochs}:')
        print(f'  Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        print(f'  Val RMSE: {val_metrics["overall_rmse"]:.6f}, R²: {val_metrics["overall_r2"]:.4f}')

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics
            }, 'best_regression_model.pth')
            print(f'  保存最佳模型，验证损失: {val_loss:.6f}')

    return train_losses, val_losses

def calculate_regression_metrics(targets, predictions, normalizer, feature_names):
    """计算回归指标"""
    metrics = {}

    # 反归一化
    if normalizer and normalizer.is_fitted:
        targets_denorm = normalizer.inverse_transform(targets)
        predictions_denorm = normalizer.inverse_transform(predictions)
    else:
        targets_denorm = targets
        predictions_denorm = predictions

    # 逐特征计算
    feature_rmse = []
    feature_r2 = []
    feature_mae = []

    for i, name in enumerate(feature_names):
        # RMSE (原始尺度)
        rmse = np.sqrt(mean_squared_error(targets_denorm[:, i], predictions_denorm[:, i]))
        feature_rmse.append(rmse)

        # R² (原始尺度)
        r2 = r2_score(targets_denorm[:, i], predictions_denorm[:, i])
        feature_r2.append(r2)

        # MAE (原始尺度)
        mae = mean_absolute_error(targets_denorm[:, i], predictions_denorm[:, i])
        feature_mae.append(mae)

        metrics[f'{name}_rmse'] = rmse
        metrics[f'{name}_r2'] = r2
        metrics[f'{name}_mae'] = mae

        # # 打印特征范围对比
        # stats = normalizer.feature_stats[name] if normalizer else {}
        # print(f"{name}:")
        # print(f"  真实范围: [{targets_denorm[:, i].min():.6e}, {targets_denorm[:, i].max():.6e}]")
        # print(f"  预测范围: [{predictions_denorm[:, i].min():.6e}, {predictions_denorm[:, i].max():.6e}]")
        # if stats:
        #     print(f"  数据范围: [{stats['min']:.6e}, {stats['max']:.6e}]")
        # print(f"  RMSE: {rmse:.6e}, R²: {r2:.4f}, MAE: {mae:.6e}")

        # 整体指标
    metrics['overall_rmse'] = np.mean(feature_rmse)
    metrics['overall_r2'] = np.mean(feature_r2)
    metrics['overall_mae'] = np.mean(feature_mae)

    return metrics


def main_regression_training():
    """主训练函数"""
    # 参数设置
    image_size = 260
    batch_size = 16
    epochs = 200
    learning_rate = 1e-4
    normalization_method = 'standard'  # 或 'minmax'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 数据转换
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # 初始化归一化器
    normalizer = FeatureNormalizer(normalization_method=normalization_method)

    # 创建数据集
    dataset = MultiModalDataset(
        data_dir="./dataset/train",  # 包含图片和表格文件的目录
        gen_data_dir='./dataset/train',
        transform=transform,
        image_size=image_size,
        table_ext=".csv",  # 根据实际文件扩展名调整
        normalizer=normalizer,
        fit_normalizer=True
    )

    # 保存归一化参数
    normalizer.save("feature_normalizer_params.json")

    # # 划分数据集
    # train_size = int(0.95* len(dataset))
    # val_size = len(dataset) - train_size
    # train_dataset, val_dataset = torch.utils.data.random_split(
    #    dataset, [train_size, val_size],
    #    generator=torch.Generator().manual_seed(42))

    # 全数据集训练测试
    train_dataset = dataset
    val_dataset = dataset

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")

    # 初始化模型
    model = MultiModalRegressionModel(
        image_channels=1,
        condition_dim=4,
        num_targets=8,
        base_channels=64
    )

    # 优化器和调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=learning_rate * 0.01)

    # 训练模型
    train_losses, val_losses = train_regression_model(
        model, train_loader, val_loader, normalizer, optimizer, scheduler, device, epochs
    )

    # 绘制训练曲线
    plot_training_curves(train_losses, val_losses, "Regression Model Training Curve")

    return model, train_losses, val_losses


def plot_training_curves(train_losses, val_losses, title):
    """绘制训练曲线"""
    plt.figure(figsize=(10, 6))
    plt.rcParams.update({
        'font.size': 14,          # 全局字体
        'axes.titlesize': 18,     # 标题字体
        'axes.labelsize': 16,     # 坐标轴标签字体
        'xtick.labelsize': 14,    # x轴刻度字体
        'ytick.labelsize': 14,    # y轴刻度字体
        'legend.fontsize': 14     # 图例字体
    })
    plt.plot(train_losses, label='training loss', linewidth=2)
    plt.plot(val_losses, label='validation loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.show()
    plt.grid(True, alpha=0.3)
    plt.savefig('regression_training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()


def test_regression_model(model_path, test_loader, device):
    """测试回归模型"""
    # 加载模型
    model = MultiModalRegressionModel(image_channels=1, condition_dim=4, num_targets=8)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    feature_names = [
        'thermal_resistance', 'nusselt_number', 'pressure_drop',
        'avg_surface_temp', 'max_surface_temp', 'heat_transfer_area',
        'avg_outlet_temp', 'max_outlet_temp'
    ]

    all_predictions = []
    all_targets = []
    all_filenames = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            conditions = batch['conditions'].to(device)
            targets = batch['targets'].to(device)

            predictions = model(images, conditions)

            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_filenames.extend(batch['filename'])

    all_predictions = np.vstack(all_predictions)
    all_targets = np.vstack(all_targets)

    # 计算测试指标
    test_metrics = calculate_regression_metrics(all_targets, all_predictions, feature_names)

    # 打印详细结果
    print("\n" + "=" * 60)
    print("回归模型测试结果")
    print("=" * 60)
    print(f"整体 RMSE: {test_metrics['overall_rmse']:.6f}")
    print(f"整体 R²: {test_metrics['overall_r2']:.4f}")
    print(f"整体 Pearson相关系数: {test_metrics['overall_corr']:.4f}")
    print("\n各特征详细指标:")
    for name in feature_names:
        print(f"  {name}:")
        print(f"    RMSE: {test_metrics[f'{name}_rmse']:.6f}")
        print(f"    R²: {test_metrics[f'{name}_r2']:.4f}")
        print(f"    相关系数: {test_metrics[f'{name}_corr']:.4f}")
    print("=" * 60)

    # 保存结果
    results_df = pd.DataFrame({
        'filename': all_filenames,
        **{f'{name}_true': all_targets[:, i] for i, name in enumerate(feature_names)},
        **{f'{name}_pred': all_predictions[:, i] for i, name in enumerate(feature_names)}
    })
    results_df.to_csv('regression_test_results.csv', index=False)

    return test_metrics, all_predictions, all_targets


if __name__ == '__main__':
    main_regression_training()