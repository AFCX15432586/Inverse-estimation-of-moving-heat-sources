"""
NDT FEM Solver
==============
基于FEM的热传导求解器，用于:
1. 求解k=1均匀情况的基准解
2. 在缺陷定位后，局部优化k值

特点:
- 高效的FEM实现
- 支持非均匀K分布
- 可用于正向求解和逆问题优化
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

class FEMSolver:
    """2D热传导FEM求解器"""
    
    def __init__(self, x, y, dt=0.001):
        """
        初始化FEM求解器
        
        参数:
            x, y: 空间网格（1D数组）
            dt: 时间步长
        """
        self.x = x
        self.y = y
        self.nx = len(x)
        self.ny = len(y)
        self.dx = x[1] - x[0]
        self.dy = y[1] - y[0]
        self.dt = dt
        
        # 网格
        self.X, self.Y = np.meshgrid(x, y, indexing='ij')
        
        print(f"FEM Solver initialized: {self.nx}x{self.ny} grid, dt={dt}")
    
    def laser_source(self, t, X, Y):
        """激光热源"""
        sigma = 0.05
        cx = 0.5 + 0.3 * np.cos(2 * np.pi * t)
        cy = 0.5 + 0.3 * np.sin(2 * np.pi * t)
        source = 50.0 * np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))
        return source
    
    def build_fem_matrices(self, K_field):
        """
        构建FEM刚度矩阵和质量矩阵
        
        参数:
            K_field: 热导率场 (nx, ny)
        
        返回:
            A: 刚度矩阵（稀疏）
            M: 质量矩阵（稀疏）
        """
        nx, ny = self.nx, self.ny
        n_total = nx * ny
        
        # 使用5点模板的有限差分逼近
        # 对于变系数情况: -∇·(K∇u) ≈ -(K_{i+1/2}(u_{i+1}-u_i)/dx^2 + ...)
        
        # 用于构建稀疏矩阵
        row_idx = []
        col_idx = []
        data = []
        
        inv_dx2 = 1.0 / (self.dx**2)
        inv_dy2 = 1.0 / (self.dy**2)
        
        def idx(i, j):
            """2D索引转1D"""
            return i * ny + j
        
        for i in range(nx):
            for j in range(ny):
                n = idx(i, j)
                
                # 边界节点（Dirichlet u=0）
                if i == 0 or i == nx-1 or j == 0 or j == ny-1:
                    row_idx.append(n)
                    col_idx.append(n)
                    data.append(1.0)
                else:
                    # 内部节点：五点模板
                    K_c = K_field[i, j]
                    
                    # 界面热导率（调和平均）
                    K_ip = 2 * K_field[i, j] * K_field[i+1, j] / (K_field[i, j] + K_field[i+1, j] + 1e-10)
                    K_im = 2 * K_field[i, j] * K_field[i-1, j] / (K_field[i, j] + K_field[i-1, j] + 1e-10)
                    K_jp = 2 * K_field[i, j] * K_field[i, j+1] / (K_field[i, j] + K_field[i, j+1] + 1e-10)
                    K_jm = 2 * K_field[i, j] * K_field[i, j-1] / (K_field[i, j] + K_field[i, j-1] + 1e-10)
                    
                    # 中心节点
                    diag = -(K_ip + K_im) * inv_dx2 - (K_jp + K_jm) * inv_dy2
                    row_idx.append(n)
                    col_idx.append(n)
                    data.append(diag)
                    
                    # x方向邻居
                    row_idx.append(n)
                    col_idx.append(idx(i+1, j))
                    data.append(K_ip * inv_dx2)
                    
                    row_idx.append(n)
                    col_idx.append(idx(i-1, j))
                    data.append(K_im * inv_dx2)
                    
                    # y方向邻居
                    row_idx.append(n)
                    col_idx.append(idx(i, j+1))
                    data.append(K_jp * inv_dy2)
                    
                    row_idx.append(n)
                    col_idx.append(idx(i, j-1))
                    data.append(K_jm * inv_dy2)
        
        A = sp.csr_matrix((data, (row_idx, col_idx)), shape=(n_total, n_total))
        
        # 质量矩阵（单位矩阵，集中质量）
        M = sp.identity(n_total, format='csr')
        
        return A, M
    
    def solve_transient(self, K_field, t_array, u0=None, verbose=True):
        """
        瞬态求解
        
        参数:
            K_field: 热导率场 (nx, ny)
            t_array: 时间数组
            u0: 初始条件（默认为0）
            verbose: 是否打印进度
        
        返回:
            U: 温度场历史 (nt, nx, ny)
        """
        nt = len(t_array)
        nx, ny = self.nx, self.ny
        
        # 初始化
        if u0 is None:
            u = np.zeros((nx, ny))
        else:
            u = u0.copy()
        
        U_history = np.zeros((nt, nx, ny))
        U_history[0] = u
        
        # 构建FEM矩阵
        if verbose:
            print("构建FEM矩阵...")
        A, M = self.build_fem_matrices(K_field)
        
        # 时间积分矩阵 (M - dt*A)
        LHS = M - self.dt * A
        
        # LU分解（加速）
        if verbose:
            print("LU分解...")
        lu = spla.splu(LHS.tocsc())
        
        # 时间步进
        if verbose:
            print(f"时间步进: {nt} steps...")
        
        for n in range(1, nt):
            t = t_array[n]
            
            # 热源项
            f = self.laser_source(t, self.X, self.Y).flatten()
            
            # 右端: M*u + dt*f
            rhs = M @ u.flatten() + self.dt * f
            
            # 求解
            u_new = lu.solve(rhs).reshape(nx, ny)
            
            # 强制边界条件
            u_new[0, :] = 0
            u_new[-1, :] = 0
            u_new[:, 0] = 0
            u_new[:, -1] = 0
            
            U_history[n] = u_new
            u = u_new
            
            if verbose and n % (nt // 10) == 0:
                print(f"  t={t:.3f}, max(u)={np.max(u):.2f}")
        
        if verbose:
            print("求解完成！")
        
        return U_history
    
    def compute_gradient_magnitude(self, U):
        """
        计算温度梯度的幅值
        
        参数:
            U: 温度场 (nt, nx, ny) 或 (nx, ny)
        
        返回:
            grad_mag: 梯度幅值
        """
        if U.ndim == 3:
            # 时间平均梯度
            grad_mag_sum = np.zeros((self.nx, self.ny))
            for t_idx in range(U.shape[0]):
                grad_x, grad_y = np.gradient(U[t_idx], self.dx, self.dy)
                grad_mag_sum += np.sqrt(grad_x**2 + grad_y**2)
            return grad_mag_sum / U.shape[0]
        else:
            grad_x, grad_y = np.gradient(U, self.dx, self.dy)
            return np.sqrt(grad_x**2 + grad_y**2)
    
    def interpolate_to_grid(self, U_measured, x_measured, y_measured, t_measured):
        """
        将测量数据插值到FEM网格
        
        参数:
            U_measured: 测量温度场 (nt, nx_m, ny_m)
            x_measured, y_measured, t_measured: 测量网格
        
        返回:
            U_interp: 插值到FEM网格的温度场 (nt, nx, ny)
        """
        # 检查是否需要插值
        if (len(x_measured) == self.nx and len(y_measured) == self.ny and
            np.allclose(x_measured, self.x) and np.allclose(y_measured, self.y)):
            return U_measured
        
        print("插值测量数据到FEM网格...")
        nt = len(t_measured)
        U_interp = np.zeros((nt, self.nx, self.ny))
        
        for t_idx in range(nt):
            interp_func = RegularGridInterpolator(
                (x_measured, y_measured),
                U_measured[t_idx],
                method='linear',
                bounds_error=False,
                fill_value=0.0
            )
            
            # 插值
            points = np.stack([self.X.ravel(), self.Y.ravel()], axis=-1)
            U_interp[t_idx] = interp_func(points).reshape(self.nx, self.ny)
        
        return U_interp

def compare_baseline_vs_measured(measured_data, save_dir='NDT_Data_V2'):
    """
    对比k=1基准解与测量数据
    
    参数:
        measured_data: 测量数据字典（从npz加载）
        save_dir: 保存目录
    
    返回:
        residual: 残差场
        grad_diff: 梯度差异
    """
    print("\n" + "="*70)
    print("对比基准解(K=1)与测量数据")
    print("="*70)
    
    # 加载测量数据
    x = measured_data['x']
    y = measured_data['y']
    t = measured_data['t']
    U_measured = measured_data['U_measured']
    
    # 创建FEM求解器
    dt = t[1] - t[0] if len(t) > 1 else 0.001
    solver = FEMSolver(x, y, dt=dt)
    
    # 求解k=1基准情况
    print("\n求解K=1基准情况...")
    K_baseline = np.ones_like(solver.X)
    U_baseline = solver.solve_transient(K_baseline, t, verbose=True)
    
    # 计算残差
    print("\n计算残差...")
    residual = U_measured - U_baseline
    
    # 计算梯度差异
    print("计算梯度差异...")
    grad_measured = solver.compute_gradient_magnitude(U_measured)
    grad_baseline = solver.compute_gradient_magnitude(U_baseline)
    grad_diff = np.abs(grad_measured - grad_baseline)
    
    # 时间平均残差
    residual_mean = np.mean(np.abs(residual), axis=0)
    
    # 统计
    print(f"\n残差统计:")
    print(f"  均值: {np.mean(residual_mean):.4f}")
    print(f"  最大值: {np.max(residual_mean):.4f}")
    print(f"  标准差: {np.std(residual_mean):.4f}")
    
    print(f"\n梯度差异统计:")
    print(f"  均值: {np.mean(grad_diff):.4f}")
    print(f"  最大值: {np.max(grad_diff):.4f}")
    
    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 第一行：温度场对比（最后时刻）
    im0 = axes[0, 0].imshow(U_baseline[-1].T, origin='lower', extent=[0,1,0,1], cmap='hot')
    axes[0, 0].set_title('Baseline (K=1)')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    plt.colorbar(im0, ax=axes[0, 0])
    
    im1 = axes[0, 1].imshow(U_measured[-1].T, origin='lower', extent=[0,1,0,1], cmap='hot')
    axes[0, 1].set_title('Measured')
    axes[0, 1].set_xlabel('x')
    plt.colorbar(im1, ax=axes[0, 1])
    
    im2 = axes[0, 2].imshow(residual_mean.T, origin='lower', extent=[0,1,0,1], cmap='RdBu_r')
    axes[0, 2].set_title('Time-Avg Residual')
    axes[0, 2].set_xlabel('x')
    plt.colorbar(im2, ax=axes[0, 2])
    
    # 第二行：梯度对比
    im3 = axes[1, 0].imshow(grad_baseline.T, origin='lower', extent=[0,1,0,1], cmap='viridis')
    axes[1, 0].set_title('Gradient (Baseline)')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')
    plt.colorbar(im3, ax=axes[1, 0])
    
    im4 = axes[1, 1].imshow(grad_measured.T, origin='lower', extent=[0,1,0,1], cmap='viridis')
    axes[1, 1].set_title('Gradient (Measured)')
    axes[1, 1].set_xlabel('x')
    plt.colorbar(im4, ax=axes[1, 1])
    
    im5 = axes[1, 2].imshow(grad_diff.T, origin='lower', extent=[0,1,0,1], cmap='hot')
    axes[1, 2].set_title('Gradient Difference')
    axes[1, 2].set_xlabel('x')
    plt.colorbar(im5, ax=axes[1, 2])
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/baseline_comparison.png', dpi=150)
    plt.close()
    
    print(f"\n对比图已保存: {save_dir}/baseline_comparison.png")
    
    return residual, grad_diff, U_baseline

if __name__ == "__main__":
    # 测试
    import os
    
    data_file = 'NDT_Data_V2/ndt_data_full.npz'
    if os.path.exists(data_file):
        data = np.load(data_file)
        residual, grad_diff, U_baseline = compare_baseline_vs_measured(data)
    else:
        print(f"请先运行 ndt_data_v2.py 生成数据！")
