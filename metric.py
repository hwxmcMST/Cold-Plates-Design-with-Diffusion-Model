import os
import torch
import math
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import torchvision.models as models
from scipy import linalg

# ======================
# 配置区域（请根据你实际生成路径修改）
# ======================
REAL_FOLDER = r"C:\Users\Administrator\Desktop\project-10.8\dataset\train"
GEN_FOLDER = r"C:\Users\Administrator\Desktop\project-10.8\generated\ddpm"
MODEL_PATH = r"C:\Users\Administrator\Desktop\project-10.8\model\inception_v3_google-0cc3c7bd.pth"
IMAGE_SIZE = (260, 260)  # 依据你的数据尺寸


# ======================
# 自动处理图像格式（灰度/RGB）统一接口
# ======================
transform_base = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x if x.shape[0] == 3 else x.repeat(3, 1, 1))  # 灰度 -> RGB
])


# ======================
# 1. SSIM函数
# ======================
def ssim_torch(img1, img2):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    mu1 = img1.mean()
    mu2 = img2.mean()
    sigma1 = img1.var()
    sigma2 = img2.var()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
    return ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2)
    )


# ======================
# 2. 计算 SSIM 和 PSNR
# ======================
def calculate_ssim_psnr(real_folder, gen_folder):
    real_imgs = sorted([os.path.join(real_folder, f) for f in os.listdir(real_folder) if f.lower().endswith(('png','jpg','jpeg'))])
    gen_imgs = sorted([os.path.join(gen_folder, f) for f in os.listdir(gen_folder) if f.lower().endswith(('png','jpg','jpeg'))])

    length = min(len(real_imgs), len(gen_imgs))
    total_ssim, total_psnr = 0, 0

    print(f"✅ Comparing {length} pairs of images...")

    for r_path, g_path in zip(real_imgs[:length], gen_imgs[:length]):
        img1 = transform_base(Image.open(r_path)).unsqueeze(0)
        img2 = transform_base(Image.open(g_path)).unsqueeze(0)

        total_ssim += ssim_torch(img1, img2).item()
        mse = F.mse_loss(img1, img2)
        total_psnr += 20 * math.log10(1.0 / math.sqrt(mse.item() + 1e-8))

    return total_ssim / length, total_psnr / length


# ======================
# 3. 自定义FID实现（使用本地 Inception-V3）
# ======================
def load_inception_model():
    model = models.inception_v3(pretrained=False, transform_input=False)
    state_dict = torch.load(MODEL_PATH, map_location='cuda')
    model.load_state_dict(state_dict)
    model.fc = torch.nn.Identity()  # 去除全连接层
    model.eval().cuda()
    return model

def get_activations(folder, model):
    activations = []
    files = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('png','jpg','jpeg'))])
    
    for file in files:
        img = transform_base(Image.open(file)).unsqueeze(0).cuda()
        with torch.no_grad():
            act = model(img).cpu().numpy().reshape(-1)
        activations.append(act)
    return np.array(activations)

def calculate_fid(real_folder, gen_folder):
    print("✅ Extracting features with Inception-V3 (local weights)...")
    model = load_inception_model()

    real_acts = get_activations(real_folder, model)
    gen_acts = get_activations(gen_folder, model)

    mu1, sigma1 = real_acts.mean(axis=0), np.cov(real_acts, rowvar=False)
    mu2, sigma2 = gen_acts.mean(axis=0), np.cov(gen_acts, rowvar=False)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)


# ======================
# 4. 主评估函数
# ======================
def compute_all_metrics():
    print("🚀 Calculating SSIM and PSNR ...")
    ssim_value, psnr_value = calculate_ssim_psnr(REAL_FOLDER, GEN_FOLDER)

    print("🚀 Calculating FID ...")
    fid_value = calculate_fid(REAL_FOLDER, GEN_FOLDER)

    print("\n========= 📊 Evaluation Result =========")
    print(f"✅ SSIM: {ssim_value:.6f}")
    print(f"✅ PSNR: {psnr_value:.6f} dB")
    print(f"✅ FID:  {fid_value:.6f}")
    print("========================================\n")


if __name__ == "__main__":
    compute_all_metrics()
