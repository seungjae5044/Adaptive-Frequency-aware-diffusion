import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader
from sklearn.linear_model import LinearRegression
from tqdm.auto import tqdm
import pandas as pd
import os
import matplotlib.pyplot as plt
from torch.utils.data import Subset
from models import UNet_small, UNet

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'mps' if torch.mps.is_available() else device

def save_checkpoint(model, optimizer, epoch, loss, path="checkpoint.pth"):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved at {path}")

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

transform = transforms.Compose([
    # transforms.Grayscale(num_output_channels=1),  # FFT 용으로 흑백 변환
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
small_dataset = Subset(dataset, range(30000))  # 0~4999번 이미지만 사용
dataloader = DataLoader(small_dataset, batch_size=128, shuffle=True, num_workers=6)

T_step = 1000

# 모델
model = UNet(in_channel=3, out_channel=3, inner_channel=64).to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

# Noise scheduler
betas = cosine_beta_schedule(T_step)
betas = torch.tensor(betas, device=device)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

def q_sample(x0, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x0)
    # 각 t에 대해 sqrt(alpha_bar)와 sqrt(1 - alpha_bar) 계산
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod[t]).view(-1, 1, 1, 1)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod[t]).view(-1, 1, 1, 1)
    return sqrt_alphas_cumprod * x0 + sqrt_one_minus_alphas_cumprod * noise

def p_losses(x_start, t):
    noise = torch.randn_like(x_start)
    x_noisy = q_sample(x_start, t, noise)
    noise_pred = model(x_noisy, t)
    return F.mse_loss(noise_pred, noise)

# Training
save_path = f"./checkpoints/epoch_0.pth"
torch.save({
    'epoch': 0,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': 0,
}, save_path)
print(f"Checkpoint saved at {save_path}")
epochs = 2000
for epoch in tqdm(range(epochs)):
    avg_loss = 0
    for batch in dataloader:
        x, _ = batch
        x = x.to(device)
        t = torch.randint(0, T_step, (x.size(0),), device=device).long()
        
        loss = p_losses(x, t)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()
    
    avg_loss /= len(dataloader)
    print()

    if (epoch + 1) % 50 == 0:  # 매 10 에폭마다 저장
        save_path = f"./checkpoints/epoch_{epoch+1}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, save_path)
        print(f"Checkpoint saved at {save_path}")
    print(f"Epoch {epoch}: Loss {avg_loss:.4f}")

save_path = f"./checkpoints/epoch_{epoch+1}.pth"
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': avg_loss,
}, save_path)
print(f"Checkpoint saved at {save_path}")

# sampling

@torch.no_grad()
def sample_grid(model, img_size=32, num_steps=20, n_row=4, n_col=4, sampling_list_ = [0]):
    model.eval()

    x_pred = []

    if len(sampling_list_)<1:
        sampling_list_.append(0)

    n_samples = n_row * n_col
    x = torch.randn(n_samples, 3, img_size, img_size).to(device)

    # Prepare scheduler
    betas = cosine_beta_schedule(num_steps)
    betas = torch.tensor(betas, device=device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)

    for t in tqdm(range(num_steps-1, -1 ,-1)):
        t_batch = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)

        noise_pred = model(x, t_batch)

        sqrt_alpha = sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha = sqrt_one_minus_alphas_cumprod[t]
        x0_pred = (x - sqrt_one_minus_alpha * noise_pred) / sqrt_alpha

        # Stability clamp
        x0_pred = torch.clamp(x0_pred, -1., 1.)

        if t > sampling_list_[-1]:
            beta = betas[t]
            noise = torch.randn_like(x)
            x = torch.sqrt(1 - beta) * x0_pred + torch.sqrt(beta) * noise
        else:
            sampling_list_.pop()
            x_pred.append(x0_pred)
    x_pred.reverse()
    return x_pred

sampling_list = [0, 20, 50, 100, 200, 500, 1000]
# 샘플링 실행
sampled_imgs = sample_grid(model, img_size=32, num_steps=T_step, n_row=4, n_col=4, sampling_list_ = sampling_list.copy())

# 시각화
for i, imgs in enumerate(sampled_imgs):
    imgs = imgs.squeeze(1).cpu().numpy()

    fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    for j, ax in enumerate(axes.flatten()):
        ax.imshow(imgs[j], cmap='gray')
        ax.axis('off')
    plt.suptitle(f'4x4 Sampled Images ({sampling_list[i]} Steps)')
    plt.tight_layout()
    plt.savefig(f"./figure/4x4sampling({sampling_list[i]} Steps)")
    plt.show()

