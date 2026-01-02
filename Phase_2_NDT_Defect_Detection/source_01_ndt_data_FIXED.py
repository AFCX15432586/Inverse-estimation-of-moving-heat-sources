import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter  # 引入高斯滤波进行平滑

def k_distribution_ground_truth(X, Y):
    # 背景 k=1.0, 缺陷 k=0.2
    K_val = np.full_like(X, 1.0)
    # 缺陷 A: 圆孔
    mask_circle = ((X - 0.35)**2 + (Y - 0.35)**2) < 0.15**2
    # 缺陷 B: 裂纹
    mask_crack = (X > 0.6) & (X < 0.9) & (Y > 0.65) & (Y < 0.75)
    
    K_val[mask_circle] = 0.2
    K_val[mask_crack] = 0.2
    return K_val

def laser_source_known(t, X, Y):
    """双圆组合激光源 - 覆盖圆形和矩形缺陷"""
    sigma = 0.05
    
    # 圆1: 原始扫描路径（覆盖圆形缺陷区域）
    cx1 = 0.5 + 0.3 * np.cos(2 * np.pi * t)
    cy1 = 0.5 + 0.3 * np.sin(2 * np.pi * t)
    source1 = 50.0 * np.exp(-((X - cx1)**2 + (Y - cy1)**2) / (2 * sigma**2))
    
    # 圆2: 偏移扫描路径（专门覆盖矩形缺陷区域）
    # 中心在(0.7, 0.7)，半径0.25，不同频率避免重叠
    cx2 = 0.7 + 0.25 * np.cos(2 * np.pi * t * 1.3)
    cy2 = 0.7 + 0.25 * np.sin(2 * np.pi * t * 1.3)
    source2 = 30.0 * np.exp(-((X - cx2)**2 + (Y - cy2)**2) / (2 * sigma**2))
    
    return source1 + source2

def solve_fdm(nx=60, ny=60):
    print(">>> 正在初始化 FDM 求解器 (Robust Version)...")
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    
    # 1. 生成 K 场
    X, Y = np.meshgrid(x, y, indexing='ij')
    K_raw = k_distribution_ground_truth(X, Y)
    
    # --- 关键修复 1: 平滑 K 分布 ---
    # 使用 sigma=1.0 的高斯滤波平滑边界，消除无限大的梯度
    # 这模拟了材料界面的过渡区，对 FDM 稳定性至关重要
    K = gaussian_filter(K_raw, sigma=1.0)
    print("    已对导热系数 K 进行平滑处理以防止梯度爆炸。")

    # --- 关键修复 2: 极度保守的 CFL 条件 ---
    # 稳定性系数从 0.4 降为 0.1，牺牲计算时间换取绝对稳定
    dt_stable = 0.1 * (dx**2) / 1.0 
    t_total = 1.0
    nt = int(t_total / dt_stable) + 1
    
    print(f"    空间网格: {nx}x{ny}")
    print(f"    安全时间步数 nt: {nt} (CFL factor=0.1)")
    
    t = np.linspace(0, t_total, nt)
    dt = t[1] - t[0]
    U = np.zeros((nt, nx, ny))
    
    # 显式差分迭代
    print(f">>> 开始迭代求解 (共 {nt} 步)...")
    
    # 预计算梯度所需的除数，加速循环
    inv_dx2 = 1.0 / dx**2
    inv_dy2 = 1.0 / dy**2
    
    for n in range(nt-1):
        u = U[n]
        
        # 边界条件处理：确保边缘计算不越界，或者使用 np.roll (但要注意 roll 会导致周期性边界)
        # 只有内部点参与演化，Dirichlet 边界自动保持为 0
        
        # 为了清晰和稳定，不使用 np.roll，使用切片计算内部点
        # u_xx[1:-1, :] = (u[0:-2, :] - 2*u[1:-1, :] + u[2:, :]) / dx^2
        
        # 使用 np.gradient 计算一阶导 (二阶精度)
        # 使用 del2 计算二阶导 (或者手写五点差分)
        
        # 这里为了保持代码简单且兼容原来的逻辑，我们修正原来的 np.roll 方式：
        # np.roll 在边界处会循环 (左边连右边)，这对于绝热或 Dirichlet 是错的。
        # 但由于我们每一步最后都把边界设为0，所以 roll 的错误会被覆盖，暂时可以用。
        
        u_xx = (np.roll(u, -1, axis=0) - 2*u + np.roll(u, 1, axis=0)) * inv_dx2
        u_yy = (np.roll(u, -1, axis=1) - 2*u + np.roll(u, 1, axis=1)) * inv_dy2
        
        k_x, k_y = np.gradient(K, dx, dy)
        u_x, u_y = np.gradient(u, dx, dy)
        
        rhs = K*(u_xx+u_yy) + k_x*u_x + k_y*u_y + laser_source_known(t[n], X, Y)
        
        U[n+1] = u + dt * rhs
        
        # 强制 Dirichlet 边界条件 (Temperature = 0)
        U[n+1, 0, :] = 0; U[n+1, -1, :] = 0
        U[n+1, :, 0] = 0; U[n+1, :, -1] = 0
        
        # 安全检查
        if n % 5000 == 0:
            if np.isnan(U[n+1]).any() or np.max(np.abs(U[n+1])) > 1e4:
                print(f"❌ 爆炸发生在 step {n}")
                break
            print(f"    Step {n}/{nt} completed. Max Temp: {np.max(U[n+1]):.2f}")

    return x, y, t, U, K

if __name__ == "__main__":
    save_dir = "NDT_Data"
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    
    try:
        x, y, t, U, K = solve_fdm(nx=60, ny=60)
        
        if np.isnan(U).any():
            print("❌ 失败：数据仍包含 NaN。")
        else:
            print("✅ 数据生成成功！")
            
            # 采样
            n_samples = 15000
            it = np.random.choice(len(t), n_samples)
            ix = np.random.choice(len(x), n_samples)
            iy = np.random.choice(len(y), n_samples)
            
            u_noisy = U[it, ix, iy] + 0.05 * np.std(U) * np.random.randn(n_samples)
            
            np.savez(os.path.join(save_dir, 'ndt_data.npz'), 
                     x=x, y=y, t=t, K_true=K,
                     t_train=t[it], x_train=x[ix], y_train=y[iy], u_train=u_noisy)
            
            # 保存预览图看看 K 是否平滑了
            plt.figure()
            plt.imshow(K.T, origin='lower', extent=[0,1,0,1])
            plt.colorbar(label='K')
            plt.title('Smoothed K Distribution')
            plt.savefig(os.path.join(save_dir, 'K_preview.png'))
            print("✅ 数据已保存。请查看 K_preview.png 确认平滑效果。")
            
    except ImportError:
        print("❌ 缺少 scipy 库。请运行 `pip install scipy`")