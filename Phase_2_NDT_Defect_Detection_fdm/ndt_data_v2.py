"""
NDT Data Generation V2
======================
生成完整的时空温度场数据，用于后续FEM分析和缺陷检测

改进点:
1. 保存完整的时空场而非随机采样
2. 提供多个时间快照用于梯度分析
3. 记录激光轨迹信息
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def k_distribution_ground_truth(X, Y):
    """真实的热导率分布"""
    K_val = np.full_like(X, 1.0)
    # 圆形缺陷在(0.35, 0.35)，半径0.15
    mask_circle = ((X - 0.35)**2 + (Y - 0.35)**2) < 0.15**2
    K_val[mask_circle] = 0.2
    return K_val

def laser_source_known(t, X, Y):
    """激光热源：圆周扫描"""
    sigma = 0.05
    cx = 0.5 + 0.3 * np.cos(2 * np.pi * t)
    cy = 0.5 + 0.3 * np.sin(2 * np.pi * t)
    source = 50.0 * np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))
    return source

def solve_fdm_full(nx=80, ny=80, nt_save=100):
    """
    FDM求解完整时空场
    
    参数:
        nx, ny: 空间网格数
        nt_save: 保存的时间快照数量
    
    返回:
        完整的时空数据
    """
    print("="*70)
    print("NDT Data Generation V2 - FDM Solver")
    print("="*70)
    
    # 空间网格
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    
    # K分布（带平滑）
    X, Y = np.meshgrid(x, y, indexing='ij')
    K_raw = k_distribution_ground_truth(X, Y)
    K = gaussian_filter(K_raw, sigma=1.5)  # 增加平滑度
    
    # 时间步长（CFL条件）
    dt_stable = 0.08 * (dx**2) / np.max(K)
    t_total = 1.0
    nt_total = int(t_total / dt_stable) + 1
    
    # 保存时间点的索引
    save_indices = np.linspace(0, nt_total-1, nt_save, dtype=int)
    
    print(f"Spatial grid: {nx} x {ny}")
    print(f"Total time steps: {nt_total}")
    print(f"Saved snapshots: {nt_save}")
    print(f"Time step dt: {dt_stable:.6f}")
    print(f"Grid spacing: dx={dx:.4f}, dy={dy:.4f}")
    
    # 时间数组
    t_all = np.linspace(0, t_total, nt_total)
    dt = t_all[1] - t_all[0]
    
    # 初始化（仅保存选定的时间快照）
    U_saved = np.zeros((nt_save, nx, ny))
    t_saved = t_all[save_indices]
    
    # 当前状态
    u_current = np.zeros((nx, ny))
    
    # 预计算常数
    inv_dx2 = 1.0 / (dx**2)
    inv_dy2 = 1.0 / (dy**2)
    
    print("\n开始FDM求解...")
    save_counter = 0
    
    for n in range(nt_total):
        # 计算空间导数（中心差分）
        u_xx = (np.roll(u_current, -1, axis=0) - 2*u_current + np.roll(u_current, 1, axis=0)) * inv_dx2
        u_yy = (np.roll(u_current, -1, axis=1) - 2*u_current + np.roll(u_current, 1, axis=1)) * inv_dy2
        
        # K的梯度
        k_x, k_y = np.gradient(K, dx, dy)
        u_x, u_y = np.gradient(u_current, dx, dy)
        
        # 热传导方程右端: ∂u/∂t = ∇·(K∇u) + f
        rhs = K * (u_xx + u_yy) + k_x * u_x + k_y * u_y + laser_source_known(t_all[n], X, Y)
        
        # 时间步进（Euler）
        u_current = u_current + dt * rhs
        
        # Dirichlet边界条件 (u=0)
        u_current[0, :] = 0
        u_current[-1, :] = 0
        u_current[:, 0] = 0
        u_current[:, -1] = 0
        
        # 保存选定的时间快照
        if n in save_indices:
            U_saved[save_counter] = u_current.copy()
            save_counter += 1
            if save_counter % 10 == 0:
                print(f"  Saved snapshot {save_counter}/{nt_save}, t={t_all[n]:.3f}, max(u)={np.max(u_current):.2f}")
    
    print(f"\n求解完成！")
    print(f"温度场统计:")
    print(f"  最大值: {np.max(U_saved):.2f}")
    print(f"  最小值: {np.min(U_saved):.2f}")
    print(f"  平均值: {np.mean(U_saved):.2f}")
    
    return x, y, t_saved, U_saved, K, K_raw

def compute_laser_trajectory(t_array):
    """计算激光轨迹"""
    cx = 0.5 + 0.3 * np.cos(2 * np.pi * t_array)
    cy = 0.5 + 0.3 * np.sin(2 * np.pi * t_array)
    return cx, cy

def visualize_data(x, y, t, U, K, K_raw, save_dir):
    """可视化数据"""
    print("\n生成可视化...")
    
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # 1. K分布对比
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    im0 = axes[0].imshow(K_raw.T, origin='lower', extent=[0,1,0,1], cmap='RdYlBu_r')
    axes[0].set_title('True K (Raw)')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    plt.colorbar(im0, ax=axes[0], label='K')
    
    im1 = axes[1].imshow(K.T, origin='lower', extent=[0,1,0,1], cmap='RdYlBu_r')
    axes[1].set_title('True K (Smoothed)')
    axes[1].set_xlabel('x')
    plt.colorbar(im1, ax=axes[1], label='K')
    
    # 缺陷位置
    circle = plt.Circle((0.35, 0.35), 0.15, color='red', fill=False, linewidth=2)
    axes[2].add_patch(circle)
    axes[2].imshow(K.T, origin='lower', extent=[0,1,0,1], cmap='RdYlBu_r', alpha=0.7)
    axes[2].set_title('Defect Location')
    axes[2].set_xlabel('x')
    axes[2].plot([0.35], [0.35], 'r*', markersize=15)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'K_distribution.png'), dpi=150)
    plt.close()
    
    # 2. 温度场演化（4个时间快照）
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # 计算激光轨迹
    cx, cy = compute_laser_trajectory(t)
    
    time_indices = [0, len(t)//3, 2*len(t)//3, -1]
    for idx, t_idx in enumerate(time_indices):
        im = axes[idx].imshow(U[t_idx].T, origin='lower', extent=[0,1,0,1], cmap='hot')
        axes[idx].set_title(f't = {t[t_idx]:.3f}')
        axes[idx].set_xlabel('x')
        axes[idx].set_ylabel('y')
        
        # 标记激光位置
        axes[idx].plot(cx[t_idx], cy[t_idx], 'c*', markersize=15, label='Laser')
        
        # 标记缺陷位置
        circle = plt.Circle((0.35, 0.35), 0.15, color='cyan', fill=False, linewidth=2, linestyle='--')
        axes[idx].add_patch(circle)
        
        plt.colorbar(im, ax=axes[idx], label='Temperature')
        axes[idx].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'temperature_evolution.png'), dpi=150)
    plt.close()
    
    # 3. 激光轨迹
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(cx, cy, 'b-', linewidth=2, label='Laser trajectory')
    ax.plot(cx[0], cy[0], 'go', markersize=10, label='Start')
    ax.plot(cx[-1], cy[-1], 'ro', markersize=10, label='End')
    
    # 缺陷位置
    circle = plt.Circle((0.35, 0.35), 0.15, color='red', fill=False, linewidth=2, label='Defect')
    ax.add_patch(circle)
    ax.plot([0.35], [0.35], 'r*', markersize=15)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Laser Scanning Trajectory')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(save_dir, 'laser_trajectory.png'), dpi=150)
    plt.close()
    
    print(f"可视化已保存到 {save_dir}")

def main():
    save_dir = "NDT_Data_V2"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 生成数据
    x, y, t, U, K, K_raw = solve_fdm_full(nx=80, ny=80, nt_save=100)
    
    # 检查NaN
    if np.isnan(U).any():
        print("\n❌ 错误: 检测到NaN值！")
        return
    
    # 添加测量噪声
    noise_level = 0.02  # 2%噪声
    U_noisy = U + noise_level * np.std(U) * np.random.randn(*U.shape)
    
    # 保存数据
    print(f"\n保存数据到 {save_dir}...")
    np.savez(
        os.path.join(save_dir, 'ndt_data_full.npz'),
        x=x,
        y=y,
        t=t,
        U_true=U,           # 无噪声温度场
        U_measured=U_noisy,  # 带噪声的测量数据
        K_true=K,           # 平滑后的K
        K_raw=K_raw,        # 原始K
        laser_cx=compute_laser_trajectory(t)[0],
        laser_cy=compute_laser_trajectory(t)[1]
    )
    
    # 可视化
    visualize_data(x, y, t, U_noisy, K, K_raw, save_dir)
    
    print("\n" + "="*70)
    print("✓ 数据生成完成！")
    print(f"✓ 数据保存在: {save_dir}/ndt_data_full.npz")
    print(f"✓ 数据形状: U={U.shape}, K={K.shape}")
    print("="*70)

if __name__ == "__main__":
    main()
