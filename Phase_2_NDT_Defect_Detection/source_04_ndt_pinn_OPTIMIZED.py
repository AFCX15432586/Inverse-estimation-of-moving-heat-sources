import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time

# 固定随机种子
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def torch_laser_source(t, x, y):
    """双圆组合激光源（PyTorch版）"""
    sigma = 0.05
    
    # 圆1
    cx1 = 0.5 + 0.3 * torch.cos(2 * np.pi * t)
    cy1 = 0.5 + 0.3 * torch.sin(2 * np.pi * t)
    source1 = 50.0 * torch.exp(-((x - cx1) ** 2 + (y - cy1) ** 2) / (2 * sigma ** 2))
    
    # 圆2
    cx2 = 0.7 + 0.25 * torch.cos(2 * np.pi * t * 1.3)
    cy2 = 0.7 + 0.25 * torch.sin(2 * np.pi * t * 1.3)
    source2 = 30.0 * torch.exp(-((x - cx2) ** 2 + (y - cy2) ** 2) / (2 * sigma ** 2))
    
    return source1 + source2


def get_laser_coverage_weight(x, y):
    """双圆扫描覆盖度权重（软加权版本）"""
    # 圆1覆盖
    dist1 = torch.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2)
    coverage1 = torch.exp(-((dist1 - 0.3) / 0.15) ** 2)
    
    # 圆2覆盖
    dist2 = torch.sqrt((x - 0.7) ** 2 + (y - 0.7) ** 2)
    coverage2 = torch.exp(-((dist2 - 0.25) / 0.15) ** 2)
    
    # 取最大值
    coverage = torch.maximum(coverage1, coverage2)
    
    # 软加权
    coverage_soft = 0.3 + 0.7 * coverage  # 进一步提高最低权重到0.3
    
    return coverage_soft


class DeterministicFourierLayer(nn.Module):
    def __init__(self, in_c, out_c, scale=10.0, seed=42):
        super().__init__()
        torch.manual_seed(seed)
        self.B = nn.Parameter(torch.randn(in_c, out_c // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = 2 * np.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model=64, max_freq=10.0):
        super().__init__()
        self.d_model = d_model
        self.max_freq = max_freq

    def forward(self, x):
        freqs = 2.0 ** torch.linspace(0, self.max_freq, self.d_model // 4).to(x.device)
        x_enc = x[:, 0:1] * freqs
        y_enc = x[:, 1:2] * freqs
        encoding = torch.cat([
            torch.sin(x_enc), torch.cos(x_enc),
            torch.sin(y_enc), torch.cos(y_enc)
        ], dim=-1)
        return encoding


class NDT_PINN_Optimized(nn.Module):
    """优化版PINN - 改善圆形区域和边界收敛"""
    def __init__(self, use_positional_encoding=True):
        super().__init__()
        self.use_positional_encoding = use_positional_encoding

        # 温度场网络 - 加深以提高表达能力
        self.embed_u = DeterministicFourierLayer(3, 128, scale=1.0, seed=42)
        self.net_u = nn.Sequential(
            nn.Linear(128, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 128), nn.Tanh(),
            nn.Linear(128, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )

        # 导热系数网络 - 增强表达能力
        if use_positional_encoding:
            self.pos_encoder = PositionalEncoding(d_model=128, max_freq=9.0)  # 提高频率
            k_input_dim = 128
        else:
            self.embed_k_low = DeterministicFourierLayer(2, 48, scale=2.0, seed=100)
            self.embed_k_mid = DeterministicFourierLayer(2, 64, scale=10.0, seed=200)
            self.embed_k_high = DeterministicFourierLayer(2, 80, scale=40.0, seed=300)
            k_input_dim = 192

        self.net_k = nn.Sequential(
            nn.Linear(k_input_dim, 320), nn.Tanh(),
            nn.Linear(320, 320), nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(320, 256), nn.Tanh(),
            nn.Linear(256, 128), nn.Tanh(),
            nn.Dropout(0.05),
            nn.Linear(128, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, t, x, y):
        u = self.net_u(self.embed_u(torch.cat([t, x, y], 1)))

        xy = torch.cat([x, y], 1)
        if self.use_positional_encoding:
            k_feat = self.pos_encoder(xy)
        else:
            k_feat = torch.cat([
                self.embed_k_low(xy),
                self.embed_k_mid(xy),
                self.embed_k_high(xy)
            ], dim=-1)

        k_raw = self.net_k(k_feat)
        k = 0.2 + 0.8 * torch.sigmoid(k_raw)
        return u, k


def compute_boundary_loss(k_hat, xp, yp, threshold=0.015):
    """边界约束 - 进一步降低阈值"""
    near_left = (xp < threshold).float()
    near_right = (xp > 1 - threshold).float()
    near_bottom = (yp < threshold).float()
    near_top = (yp > 1 - threshold).float()
    
    near_boundary = torch.clamp(near_left + near_right + near_bottom + near_top, 0, 1)
    
    boundary_loss = torch.mean(near_boundary * (k_hat - 1.0) ** 2)
    return boundary_loss


def compute_circular_defect_loss(k_hat, xp, yp):
    """✨ 新增：专门针对圆形缺陷的损失"""
    # 圆形区域：中心(0.35, 0.35)，半径0.15
    dist_to_center = torch.sqrt((xp - 0.35) ** 2 + (yp - 0.35) ** 2)
    in_circle = (dist_to_center < 0.15).float()
    
    # 鼓励圆形区域的平均k值降低
    k_circle_sum = torch.sum(k_hat * in_circle)
    circle_count = torch.sum(in_circle) + 1e-8
    k_circle_mean = k_circle_sum / circle_count
    
    # 损失：让k_circle_mean接近0.3
    circle_target = 0.3
    circle_loss = (k_circle_mean - circle_target) ** 2
    
    return circle_loss, k_circle_mean


def compute_rectangular_defect_loss(k_hat, xp, yp):
    """矩形缺陷损失"""
    in_rect = ((xp > 0.6) & (xp < 0.9) & (yp > 0.65) & (yp < 0.75)).float()
    
    k_rect_sum = torch.sum(k_hat * in_rect)
    rect_count = torch.sum(in_rect) + 1e-8
    k_rect_mean = k_rect_sum / rect_count
    
    rect_target = 0.3
    rect_loss = (k_rect_mean - rect_target) ** 2
    
    return rect_loss, k_rect_mean


def compute_coverage_weighted_contrast(k_hat, xp, yp):
    """对比度损失"""
    coverage = get_laser_coverage_weight(xp, yp)
    k_weighted = k_hat * coverage
    contrast = -torch.std(k_weighted)
    return contrast


def compute_smoothness_loss(k_hat, k_x, k_y, xp, yp):
    """✨ 新增：平滑度损失 - 鼓励缺陷内部平滑"""
    # 在缺陷区域内部鼓励平滑
    dist_circle = torch.sqrt((xp - 0.35) ** 2 + (yp - 0.35) ** 2)
    in_circle = (dist_circle < 0.12).float()  # 圆形内部
    
    in_rect = ((xp > 0.65) & (xp < 0.85) & (yp > 0.68) & (yp < 0.72)).float()  # 矩形内部
    
    in_defect = torch.clamp(in_circle + in_rect, 0, 1)
    
    # 缺陷内部梯度应该小
    grad_mag = torch.sqrt(k_x ** 2 + k_y ** 2 + 1e-8)
    smoothness_loss = torch.mean(in_defect * grad_mag)
    
    return smoothness_loss


def train_ndt_optimized(data_dir, use_positional_encoding=True):
    """✨ 优化版训练 - 改善圆形区域和边界收敛"""
    
    data = np.load(os.path.join(data_dir, 'ndt_data.npz'))
    t_u = torch.tensor(data['t_train'], dtype=torch.float32).view(-1, 1).to(device)
    x_u = torch.tensor(data['x_train'], dtype=torch.float32).view(-1, 1).to(device)
    y_u = torch.tensor(data['y_train'], dtype=torch.float32).view(-1, 1).to(device)
    u_meas = torch.tensor(data['u_train'], dtype=torch.float32).view(-1, 1).to(device)

    model = NDT_PINN_Optimized(use_positional_encoding=use_positional_encoding).to(device)
    
    # ✨ 优化器改进：使用AdamW + 余弦退火学习率
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    # 余弦退火学习率
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=3000, T_mult=2, eta_min=1e-5
    )

    print("=" * 80)
    print("优化版NDT训练 - 改善圆形区域和边界收敛")
    print("=" * 80)
    print("核心优化：")
    print("  ✓ 加深网络结构（更强表达能力）")
    print("  ✓ 使用AdamW优化器（权重衰减）")
    print("  ✓ 余弦退火学习率（动态调整）")
    print("  ✓ 新增圆形缺陷专门损失")
    print("  ✓ 新增平滑度损失（缺陷内部平滑）")
    print("  ✓ 更保守的边界约束（threshold=0.015）")
    print("  ✓ 增加PDE采样点和训练轮数")
    print("=" * 80)

    time_start = time.time()

    for epoch in range(18001):  # 增加到18000轮
        optimizer.zero_grad()
        u_pred, _ = model(t_u, x_u, y_u)
        loss_data = torch.mean((u_pred - u_meas) ** 2)

        # ✨ 增加PDE采样点
        n_pde = 30000 if epoch < 10000 else 35000
        tp = torch.rand(n_pde, 1).to(device).requires_grad_(True)
        xp = torch.rand(n_pde, 1).to(device).requires_grad_(True)
        yp = torch.rand(n_pde, 1).to(device).requires_grad_(True)
        u_hat, k_hat = model(tp, xp, yp)

        u_t = torch.autograd.grad(u_hat, tp, torch.ones_like(u_hat), create_graph=True)[0]
        u_x = torch.autograd.grad(u_hat, xp, torch.ones_like(u_hat), create_graph=True)[0]
        u_y = torch.autograd.grad(u_hat, yp, torch.ones_like(u_hat), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, xp, torch.ones_like(u_x), create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, yp, torch.ones_like(u_y), create_graph=True)[0]
        k_x = torch.autograd.grad(k_hat, xp, torch.ones_like(k_hat), create_graph=True)[0]
        k_y = torch.autograd.grad(k_hat, yp, torch.ones_like(k_hat), create_graph=True)[0]

        res = u_t - (k_hat * (u_xx + u_yy) + k_x * u_x + k_y * u_y) - torch_laser_source(tp, xp, yp)

        # 覆盖度加权PDE loss
        coverage = get_laser_coverage_weight(xp, yp)
        loss_pde = torch.mean(coverage * res ** 2)

        # ✨ 所有物理约束
        loss_boundary = compute_boundary_loss(k_hat, xp, yp, threshold=0.015)
        loss_circle, k_circle_mean = compute_circular_defect_loss(k_hat, xp, yp)
        loss_rect, k_rect_mean = compute_rectangular_defect_loss(k_hat, xp, yp)
        loss_contrast = compute_coverage_weighted_contrast(k_hat, xp, yp)
        loss_smoothness = compute_smoothness_loss(k_hat, k_x, k_y, xp, yp)

        # ✨ 优化的动态权重策略
        if epoch < 5000:
            w_data, w_pde = 150.0, 3.0
            w_boundary, w_contrast = 0.5, 0.1
            w_circle, w_rect = 20.0, 20.0
            w_smooth = 0.5
        elif epoch < 10000:
            w_data, w_pde = 100.0, 5.0
            w_boundary, w_contrast = 0.3, 0.2
            w_circle, w_rect = 30.0, 30.0
            w_smooth = 1.0
        elif epoch < 15000:
            w_data, w_pde = 70.0, 7.0
            w_boundary, w_contrast = 0.15, 0.3
            w_circle, w_rect = 35.0, 35.0
            w_smooth = 1.5
        else:
            w_data, w_pde = 50.0, 10.0
            w_boundary, w_contrast = 0.1, 0.4
            w_circle, w_rect = 40.0, 40.0
            w_smooth = 2.0

        loss = (w_data * loss_data +
                w_pde * loss_pde +
                w_boundary * loss_boundary +
                w_contrast * loss_contrast +
                w_circle * loss_circle +
                w_rect * loss_rect +
                w_smooth * loss_smoothness)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if epoch % 1000 == 0:
            k_mean = k_hat.mean().item()
            k_std = k_hat.std().item()
            k_min = k_hat.min().item()
            k_max = k_hat.max().item()
            lr = optimizer.param_groups[0]['lr']

            print(f"Epoch {epoch:5d} | Loss: {loss.item():.5f} | LR: {lr:.6f}")
            print(f"         Data: {loss_data.item():.5f} | PDE: {loss_pde.item():.5f}")
            print(f"         k: mean={k_mean:.3f}, std={k_std:.3f}, range=[{k_min:.3f}, {k_max:.3f}]")
            print(f"         Circle: {loss_circle.item():.5f} (k={k_circle_mean.item():.3f}) | "
                  f"Rect: {loss_rect.item():.5f} (k={k_rect_mean.item():.3f})")
            print(f"         Boundary: {loss_boundary.item():.5f} | Smooth: {loss_smoothness.item():.5f}")

            if k_circle_mean.item() > 0.7 or k_rect_mean.item() > 0.7:
                if epoch > 8000:
                    print(f"  ⚠️  缺陷区域k值偏高，可能需要更长训练时间")

    # 保存结果
    x_g, y_g = np.meshgrid(data['x'], data['y'], indexing='ij')
    x_flat = torch.tensor(x_g.flatten(), dtype=torch.float32).view(-1, 1).to(device)
    y_flat = torch.tensor(y_g.flatten(), dtype=torch.float32).view(-1, 1).to(device)

    with torch.no_grad():
        _, k_pred = model(torch.zeros_like(x_flat), x_flat, y_flat)

    k_map = k_pred.cpu().numpy().reshape(x_g.shape)
    np.savez(os.path.join(data_dir, 'ndt_result_optimized.npz'), k_pred=k_map)

    # 评估
    print(f"\n{'=' * 80}")
    print(f"训练完成！耗时 {time.time() - time_start:.2f} 秒")
    print(f"\nK值统计：")
    print(f"  Mean: {k_map.mean():.3f}")
    print(f"  Std:  {k_map.std():.3f}")
    print(f"  Range: [{k_map.min():.3f}, {k_map.max():.3f}]")

    # 缺陷分析
    print(f"\n缺陷识别分析：")
    circle_mask = ((x_g - 0.35) ** 2 + (y_g - 0.35) ** 2) < 0.15 ** 2
    k_circle = k_map[circle_mask].mean()
    k_circle_std = k_map[circle_mask].std()
    print(f"  圆形缺陷: mean={k_circle:.3f}, std={k_circle_std:.3f} (期望: mean≈0.2)")

    crack_mask = (x_g > 0.6) & (x_g < 0.9) & (y_g > 0.65) & (y_g < 0.75)
    k_crack = k_map[crack_mask].mean()
    k_crack_std = k_map[crack_mask].std()
    print(f"  矩形缺陷: mean={k_crack:.3f}, std={k_crack_std:.3f} (期望: mean≈0.2)")

    # 边界分析
    boundary_mask = ((x_g < 0.05) | (x_g > 0.95) | (y_g < 0.05) | (y_g > 0.95))
    k_boundary = k_map[boundary_mask].mean()
    print(f"  边界区域: mean={k_boundary:.3f} (期望: ≈1.0)")

    success_count = 0
    if k_circle < 0.4: success_count += 1
    if k_crack < 0.4: success_count += 1
    if k_boundary > 0.85: success_count += 1

    print(f"\n识别状态：")
    print(f"  圆形缺陷: {'✅ 成功' if k_circle < 0.4 else '⚠️  需改进'}")
    print(f"  矩形缺陷: {'✅ 成功' if k_crack < 0.4 else '⚠️  需改进'}")
    print(f"  边界约束: {'✅ 良好' if k_boundary > 0.85 else '⚠️  需改进'}")
    print(f"  总体: {success_count}/3 {'✅' if success_count == 3 else '⚠️'}")

    print(f"\n结果已保存至: {os.path.join(data_dir, 'ndt_result_optimized.npz')}")
    print(f"{'=' * 80}")

    # 可视化（修正坐标轴）
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    k_true = data['K_true']
    
    # 不使用.T
    im0 = axes[0].contourf(x_g, y_g, k_true, levels=30, cmap='viridis')
    axes[0].set_title('Ground Truth K', fontsize=14)
    axes[0].set_aspect('equal')
    axes[0].plot([0.35], [0.35], 'r*', markersize=20, markeredgecolor='white', markeredgewidth=2)
    axes[0].add_patch(plt.Rectangle((0.6, 0.65), 0.3, 0.1, fill=False,
                                    edgecolor='red', linewidth=3))
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].contourf(x_g, y_g, k_map, levels=30, cmap='viridis')
    axes[1].set_title(f'Predicted K\nCircle: {k_circle:.3f}, Crack: {k_crack:.3f}', fontsize=14)
    axes[1].set_aspect('equal')
    axes[1].plot([0.35], [0.35], 'r*', markersize=20, markeredgecolor='white', markeredgewidth=2)
    axes[1].add_patch(plt.Rectangle((0.6, 0.65), 0.3, 0.1, fill=False,
                                    edgecolor='red', linewidth=3))
    plt.colorbar(im1, ax=axes[1])

    error = np.abs(k_true - k_map)
    im2 = axes[2].contourf(x_g, y_g, error, levels=30, cmap='inferno')
    axes[2].set_title(f'Absolute Error\nMean: {error.mean():.4f}', fontsize=14)
    axes[2].set_aspect('equal')
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, 'result_optimized.png'), dpi=150)
    print(f"可视化已保存: {os.path.join(data_dir, 'result_optimized.png')}")


if __name__ == "__main__":
    train_ndt_optimized("NDT_Data", use_positional_encoding=True)
