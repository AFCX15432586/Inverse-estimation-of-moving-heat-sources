import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time

print(torch.cuda.is_available())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def torch_laser_source(t, x, y):
    cx = 0.5 + 0.3 * torch.cos(2 * np.pi * t)
    cy = 0.5 + 0.3 * torch.sin(2 * np.pi * t)
    return 50.0 * torch.exp(-((x - cx)**2 + (y - cy)**2) / (2 * 0.05**2))

class FourierLayer(nn.Module):
    def __init__(self, in_c, out_c, scale=10.0):
        super().__init__()
        self.B = nn.Parameter(torch.randn(in_c, out_c//2)*scale, requires_grad=False)
    def forward(self, x):
        x_proj = 2 * np.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class NDT_PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_u = FourierLayer(3, 64, scale=1.0)
        self.net_u = nn.Sequential(nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
        
        self.embed_k = FourierLayer(2, 64, scale=5.0) # High scale for sharp edges
        self.net_k = nn.Sequential(nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1), nn.Softplus())

    def forward(self, t, x, y):
        u = self.net_u(self.embed_u(torch.cat([t, x, y], 1)))
        k = self.net_k(self.embed_k(torch.cat([x, y], 1))) + 0.05
        return u, k

def train_ndt(data_dir):
    data = np.load(os.path.join(data_dir, 'ndt_data.npz'))
    t_u = torch.tensor(data['t_train'], dtype=torch.float32).view(-1,1).to(device)
    x_u = torch.tensor(data['x_train'], dtype=torch.float32).view(-1,1).to(device)
    y_u = torch.tensor(data['y_train'], dtype=torch.float32).view(-1,1).to(device)
    u_meas = torch.tensor(data['u_train'], dtype=torch.float32).view(-1,1).to(device)

    model = NDT_PINN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    print("开始 Phase 2 NDT 训练...")
    time_start = time.time()
    for epoch in range(15001):
        optimizer.zero_grad()
        u_pred, _ = model(t_u, x_u, y_u)
        loss_data = torch.mean((u_pred - u_meas)**2)
        
        # PDE Loss
        tp = torch.rand(10000, 1).to(device).requires_grad_(True)
        xp = torch.rand(10000, 1).to(device).requires_grad_(True)
        yp = torch.rand(10000, 1).to(device).requires_grad_(True)
        u_hat, k_hat = model(tp, xp, yp)
        
        u_t = torch.autograd.grad(u_hat, tp, torch.ones_like(u_hat), create_graph=True)[0]
        u_x = torch.autograd.grad(u_hat, xp, torch.ones_like(u_hat), create_graph=True)[0]
        u_y = torch.autograd.grad(u_hat, yp, torch.ones_like(u_hat), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, xp, torch.ones_like(u_x), create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, yp, torch.ones_like(u_y), create_graph=True)[0]
        k_x = torch.autograd.grad(k_hat, xp, torch.ones_like(k_hat), create_graph=True)[0]
        k_y = torch.autograd.grad(k_hat, yp, torch.ones_like(k_hat), create_graph=True)[0]
        
        res = u_t - (k_hat*(u_xx+u_yy) + k_x*u_x + k_y*u_y) - torch_laser_source(tp, xp, yp)
        loss = loss_data * 100 + torch.mean(res**2)
        
        loss.backward()
        optimizer.step()
        if epoch % 2000 == 0: print(f"Epoch {epoch}, Loss: {loss.item():.5f}")

    # 保存 K map
    x_g, y_g = np.meshgrid(data['x'], data['y'], indexing='ij')
    x_flat = torch.tensor(x_g.flatten(), dtype=torch.float32).view(-1,1).to(device)
    y_flat = torch.tensor(y_g.flatten(), dtype=torch.float32).view(-1,1).to(device)
    with torch.no_grad(): _, k_pred = model(torch.zeros_like(x_flat), x_flat, y_flat)
    np.savez(os.path.join(data_dir, 'ndt_result.npz'), k_pred=k_pred.cpu().numpy().reshape(x_g.shape))
    print(f"NDT 训练完成，耗时 {time.time() - time_start:.2f} 秒")

if __name__ == "__main__": train_ndt("NDT_Data")