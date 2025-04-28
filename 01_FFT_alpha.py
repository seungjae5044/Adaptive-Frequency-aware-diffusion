import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
import pandas as pd
import os

# 1. 데이터 로딩
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # FFT 용으로 흑백 변환
    transforms.ToTensor()
])

dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# 2. FFT + 3. Spectrum 계산
def compute_alpha(img):
    img = img.squeeze(0).numpy()  # [1,H,W] -> [H,W]
    F = np.fft.fftshift(np.fft.fft2(img))
    P = np.abs(F)**2
    breakpoint()
    
    # 4. Radial frequency axis
    h, w = P.shape
    y, x = np.indices((h, w))
    cx, cy = w//2, h//2
    r = np.sqrt((x - cx)**2 + (y - cy)**2).flatten()
    P = P.flatten()
    
    # 5. log-log plot
    valid = r > 0
    log_r = np.log(r[valid]).reshape(-1, 1)
    log_P = np.log(P[valid])
    
    # 6. Linear regression
    reg = LinearRegression().fit(log_r, log_P)
    alpha = -reg.coef_[0]  # 주의! 부호 반전
    
    return alpha

# 결과 저장
alphas = []

for idx, (img, _) in tqdm(enumerate(loader), total=len(loader)):
    alpha = compute_alpha(img)
    alphas.append({'id': idx, 'alpha': alpha})

df = pd.DataFrame(alphas)

# 7. 저장
os.makedirs('./fft_alpha', exist_ok=True)
df.to_csv('./fft_alpha/alpha_values.csv', index=False)

print("✅ Alpha 추정 완료! 데이터 수:", len(df))
