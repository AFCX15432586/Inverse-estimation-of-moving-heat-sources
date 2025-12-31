import numpy as np
import matplotlib.pyplot as plt
import os

def k_func_2d(X, Y):
    # 模拟非均一介质，已知
    return 0.01 + 0.005 * np.sin(np.pi * X) * np.cos(np.pi * Y)

def source_moving_laser_2d(t, X, Y):
    # 待反演的目标：移动激光
    sigma = 0.06
    amplitude = 15.0
    cx = 0.5 + 0.25 * np.cos(2 * np.pi * t)
    cy = 0.5 + 0.25 * np.sin(2 * np.pi * t)
    return amplitude * np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))

def solve_fdm_2d(nx=60, ny=60, nt=800):
    x = np.linspace(0, 1, nx); y = np.linspace(0, 1, ny); t = np.linspace(0, 1, nt)
    dx, dy, dt = x[1]-x[0], y[1]-y[0], t[1]-t[0]
    X, Y = np.meshgrid(x, y, indexing='ij')
    K = k_func_2d(X, Y)
    
    U = np.zeros((nt, nx, ny))
    F_true = np.zeros((nt, nx, ny))
    
    for n in range(nt-1):
        F_true[n] = source_moving_laser_2d(t[n], X, Y)
        u = U[n]
        u_xx = (np.roll(u, -1, axis=0) - 2*u + np.roll(u, 1, axis=0)) / dx**2
        u_yy = (np.roll(u, -1, axis=1) - 2*u + np.roll(u, 1, axis=1)) / dy**2
        k_x, k_y = np.gradient(K, dx, dy)
        u_x, u_y = np.gradient(u, dx, dy)
        
        diffusion = K * (u_xx + u_yy) + (k_x * u_x + k_y * u_y)
        U[n+1] = u + dt * (diffusion + F_true[n])
        U[n+1, [0,-1], :] = 0; U[n+1, :, [0,-1]] = 0 # BC

    return x, y, t, U, F_true

if __name__ == "__main__":
    out_dir = "Phase1_Data"
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    x, y, t, U, F = solve_fdm_2d()
    
    # 采样
    n_samples = 8000
    it = np.random.choice(len(t), n_samples)
    ix = np.random.choice(len(x), n_samples)
    iy = np.random.choice(len(y), n_samples)
    
    # === 修改处：增加了 U_true=U ===
    np.savez(os.path.join(out_dir, 'source_data_2d.npz'), 
             x=x, y=y, t=t, F_true=F, U_true=U, 
             t_train=t[it], x_train=x[ix], y_train=y[iy], 
             u_train=U[it, ix, iy] + 0.01 * np.random.randn(n_samples))
    print("Phase 1 数据生成完成 (已包含完整温度场 U_true)。")