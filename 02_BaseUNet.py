import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleUNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=64):
        super().__init__()
        
        self.enc1 = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.enc2 = nn.Conv2d(base_channels, base_channels*2, 3, padding=1)
        self.enc3 = nn.Conv2d(base_channels*2, base_channels*4, 3, padding=1)
        
        self.dec2 = nn.Conv2d(base_channels*4, base_channels*2, 3, padding=1)
        self.dec1 = nn.Conv2d(base_channels*2, base_channels, 3, padding=1)
        self.out = nn.Conv2d(base_channels, 1, 1)
        
        self.act = nn.SiLU()

    def forward(self, x, t_emb):
        # t_emb는 timestep embedding (broadcasting)
        x1 = self.act(self.enc1(x))
        x2 = self.act(self.enc2(F.avg_pool2d(x1, 2)))
        x3 = self.act(self.enc3(F.avg_pool2d(x2, 2)))
        
        x = F.interpolate(x3, scale_factor=2)
        x = self.act(self.dec2(x + x2))
        
        x = F.interpolate(x, scale_factor=2)
        x = self.act(self.dec1(x + x1))
        
        out = self.out(x)
        return out

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

import torch.optim as optim

device = 'mps' if torch.mps.is_available() else 'cpu'

# 모델
model = SimpleUNet(in_channels=1).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Noise scheduler
betas = cosine_beta_schedule(1000)
betas = torch.tensor(betas, device=device)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

def q_sample(x_start, t, noise):
    sqrt_alpha_cumprod = torch.sqrt(alphas_cumprod[t]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    sqrt_one_minus_alpha = torch.sqrt(1 - alphas_cumprod[t]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    return sqrt_alpha_cumprod * x_start + sqrt_one_minus_alpha * noise

def p_losses(x_start, t):
    noise = torch.randn_like(x_start)
    x_noisy = q_sample(x_start, t, noise)
    noise_pred = model(x_noisy, t)
    return F.mse_loss(noise_pred, noise)

# Training
epochs = 100
for epoch in range(epochs):
    for batch in dataloader:
        x, _ = batch
        x = x.to(device)
        t = torch.randint(0, 1000, (x.size(0),), device=device).long()
        
        loss = p_losses(x, t)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch}: Loss {loss.item():.4f}")
