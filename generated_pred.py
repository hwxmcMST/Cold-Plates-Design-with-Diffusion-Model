import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from dataset.dataset import ConditionDataset
from model.regression_model import MultiModalRegressionModel
from tools import FeatureNormalizer


def predict_generated_dataset(model_path, image_dir, normalizer_path, device):
    """
    Use trained regression model to predict physical quantities
    (8 outputs) for generated heat sink geometries.
    """
    # ------------------------------
    # 1. Load normalizer
    # ------------------------------
    normalizer = FeatureNormalizer()
    normalizer.load(normalizer_path)

    # ------------------------------
    # 2. Load trained regression model
    # ------------------------------
    model = MultiModalRegressionModel(image_channels=1, condition_dim=4, num_targets=8)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # ------------------------------
    # 3. Build dataset from generated images
    # ------------------------------
    transform = transforms.Compose([
        transforms.Resize((260, 260)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    dataset = ConditionDataset(image_dir=image_dir, transform=transform, image_size=260)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)

    feature_names = [
        'Thermal Resistance (K/W)',
        'Nusselt Number',
        'Pressure Drop (Pa)',
        'Average Surface Temperature (K)',
        'Maximum Surface Temperature (K)',
        'Effective Heat Transfer Area m^2',
        'Average Outlet Temperature (K)',
        'Maximum Outlet Temperature (K)'
    ]

    all_predictions = []
    all_filenames = []

    # ------------------------------
    # 4. Model inference
    # ------------------------------
    with torch.no_grad():
        for i, (images, conditions) in enumerate(tqdm(dataloader, desc="Predicting")):
            images = images.to(device)
            conditions = conditions.to(device)

            preds = model(images, conditions)  # Normalized outputs
            preds = preds.cpu().numpy()

            all_predictions.append(preds)
            start_idx = i * dataloader.batch_size
            all_filenames.extend([os.path.basename(p) for p in dataset.image_paths[start_idx:start_idx + len(preds)]])

    all_predictions = np.vstack(all_predictions)

    # ------------------------------
    # 5. Denormalize predictions
    # ------------------------------
    preds_denorm = normalizer.inverse_transform(all_predictions)

    # ------------------------------
    # 6. Save to CSV
    # ------------------------------
    results_df = pd.DataFrame(preds_denorm, columns=feature_names)
    results_df.insert(0, "filename", all_filenames)

    output_path = "generated_predictions_denormalized.csv"
    results_df.to_csv(output_path, index=False, float_format="%.6e")

    print(f"\n✅ Prediction complete. Results saved to: {output_path}")
    print(f"Example:\n{results_df.head()}")
    return results_df


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    predict_generated_dataset(
        model_path="best_regression_model.pth",
        image_dir=r"C:\Users\Administrator\Desktop\project-10.8\generated",
        normalizer_path="feature_normalizer_params.json",
        device=device
    )
