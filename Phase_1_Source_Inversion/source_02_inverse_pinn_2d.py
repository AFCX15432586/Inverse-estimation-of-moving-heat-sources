import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PINN_2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net_u = nn.Sequential(nn.Linear(3, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
        self.net_f = nn.Sequential(nn.Linear(3, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))

    def forward(self, t, x, y):
        inpt = torch.cat([t, x, y], dim=1)
        return self.net_u(inpt), self.net_f(inpt)

def get_k_torch(x, y):
    return 0.01 + 0.005 * torch.sin(np.pi * x) * torch.cos(np.pi * y)

def train_2d(target_dir):
    data = np.load(os.path.join(target_dir, 'source_data_2d.npz'))
    t_u = torch.tensor(data['t_train'], dtype=torch.float32).view(-1,1).to(device)
    x_u = torch.tensor(data['x_train'], dtype=torch.float32).view(-1,1).to(device)
    y_u = torch.tensor(data['y_train'], dtype=torch.float32).view(-1,1).to(device)
    u_raw = torch.tensor(data['u_train'], dtype=torch.float32).view(-1,1).to(device)

    model = PINN_2D().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    print("开始 Phase 1 训练...")
    for epoch in range(10001):
        optimizer.zero_grad()
        # Data Loss
        u_pred, _ = model(t_u, x_u, y_u)
        loss_data = torch.mean((u_pred - u_raw)**2)
        
        # PDE Loss
        tp = torch.rand(5000, 1).to(device).requires_grad_(True)
        xp = torch.rand(5000, 1).to(device).requires_grad_(True)
        yp = torch.rand(5000, 1).to(device).requires_grad_(True)
        u, f_pred = model(tp, xp, yp)
        k = get_k_torch(xp, yp)
        
        u_t = torch.autograd.grad(u, tp, torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, xp, torch.ones_like(u), create_graph=True)[0]
        u_y = torch.autograd.grad(u, yp, torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, xp, torch.ones_like(u_x), create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, yp, torch.ones_like(u_y), create_graph=True)[0]
        k_x = torch.autograd.grad(k, xp, torch.ones_like(k), create_graph=True)[0]
        k_y = torch.autograd.grad(k, yp, torch.ones_like(k), create_graph=True)[0]
        
        pde = u_t - (k*(u_xx+u_yy) + k_x*u_x + k_y*u_y) - f_pred
        loss = loss_data * 10 + torch.mean(pde**2)
        
        loss.backward()
        optimizer.step()
        if epoch % 1000 == 0: print(f"Epoch {epoch}, Loss: {loss.item():.5f}")

    # 保存预测
    x_g, y_g = np.meshgrid(data['x'], data['y'], indexing='ij')
    t_val = 0.5 * torch.ones((x_g.size, 1)).to(device)
    x_val = torch.tensor(x_g.flatten(), dtype=torch.float32).view(-1,1).to(device)
    y_val = torch.tensor(y_g.flatten(), dtype=torch.float32).view(-1,1).to(device)
    with torch.no_grad(): _, f_res = model(t_val, x_val, y_val)
    np.savez(os.path.join(target_dir, 'inverse_pred_2d.npz'), f_pred=f_res.cpu().numpy().reshape(x_g.shape))

if __name__ == "__main__": train_2d("Phase1_Data")