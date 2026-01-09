"""
NDT Defect Localization and Quantification
==========================================
两阶段方法:
1. 阶段1: 通过梯度差异定位缺陷区域
2. 阶段2: 在缺陷区域局部优化K值

优势:
- 更鲁棒（基于物理的FEM）
- 更精确（局部优化问题规模小）
- 更快速（分阶段处理）
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, label
from scipy.optimize import minimize
import os
import sys

# 确保能找到同目录下的模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ndt_fem_solver import FEMSolver

class DefectLocalizer:
    """缺陷定位器"""
    
    def __init__(self, x, y, t, U_measured, threshold_percentile=92):
        """
        初始化
        
        参数:
            x, y: 空间网格
            t: 时间数组
            U_measured: 测量温度场
            threshold_percentile: 梯度差异阈值百分位数
        """
        self.x = x
        self.y = y
        self.t = t
        self.U_measured = U_measured
        self.threshold_percentile = threshold_percentile
        
        self.X, self.Y = np.meshgrid(x, y, indexing='ij')
        
        print(f"DefectLocalizer初始化: {len(x)}x{len(y)} 网格")
    
    def compute_gradient_anomaly(self, U_baseline):
        """
        计算梯度异常
        
        参数:
            U_baseline: k=1基准解
        
        返回:
            grad_anomaly: 梯度异常图
        """
        print("\n计算梯度异常...")
        
        dx = self.x[1] - self.x[0]
        dy = self.y[1] - self.y[0]
        
        # 时间平均梯度
        grad_mag_measured = np.zeros_like(self.X)
        grad_mag_baseline = np.zeros_like(self.X)
        
        for t_idx in range(len(self.t)):
            # 测量数据梯度
            grad_x_m, grad_y_m = np.gradient(self.U_measured[t_idx], dx, dy)
            grad_mag_measured += np.sqrt(grad_x_m**2 + grad_y_m**2)
            
            # 基准梯度
            grad_x_b, grad_y_b = np.gradient(U_baseline[t_idx], dx, dy)
            grad_mag_baseline += np.sqrt(grad_x_b**2 + grad_y_b**2)
        
        grad_mag_measured /= len(self.t)
        grad_mag_baseline /= len(self.t)
        
        # 梯度差异（归一化）
        grad_anomaly = np.abs(grad_mag_measured - grad_mag_baseline)
        grad_anomaly = grad_anomaly / (np.max(grad_anomaly) + 1e-10)
        
        print(f"梯度异常 - 最大值: {np.max(grad_anomaly):.4f}")
        
        return grad_anomaly
    
    def compute_residual_anomaly(self, U_baseline):
        """
        计算温度残差异常
        
        参数:
            U_baseline: k=1基准解
        
        返回:
            residual_anomaly: 残差异常图
        """
        print("计算残差异常...")
        
        # 时间平均绝对残差
        residual = self.U_measured - U_baseline
        residual_mean = np.mean(np.abs(residual), axis=0)
        
        # 归一化
        residual_anomaly = residual_mean / (np.max(residual_mean) + 1e-10)
        
        print(f"残差异常 - 最大值: {np.max(residual_anomaly):.4f}")
        
        return residual_anomaly
    
    def locate_defects(self, U_baseline, use_gradient=True, use_residual=True):
        """
        定位缺陷
        
        参数:
            U_baseline: k=1基准解
            use_gradient: 是否使用梯度异常
            use_residual: 是否使用残差异常
        
        返回:
            defect_mask: 缺陷掩码
            anomaly_map: 综合异常图
        """
        print("\n" + "="*70)
        print("阶段1: 缺陷定位")
        print("="*70)
        
        anomaly_maps = []
        
        if use_gradient:
            grad_anomaly = self.compute_gradient_anomaly(U_baseline)
            anomaly_maps.append(grad_anomaly)
        
        if use_residual:
            residual_anomaly = self.compute_residual_anomaly(U_baseline)
            anomaly_maps.append(residual_anomaly)
        
        # 综合异常图（加权平均）
        if len(anomaly_maps) == 0:
            raise ValueError("至少需要一种异常检测方法")
        
        anomaly_map = np.mean(anomaly_maps, axis=0)
        
        # 平滑异常图
        anomaly_smooth = gaussian_filter(anomaly_map, sigma=2.0)
        
        # 阈值分割
        threshold = np.percentile(anomaly_smooth, self.threshold_percentile)
        defect_mask = anomaly_smooth > threshold
        
        # 形态学处理（去除小区域）
        labeled, num_features = label(defect_mask)
        
        print(f"\n检测到 {num_features} 个候选缺陷区域")
        
        # 过滤小区域
        min_size = 20  # 最小缺陷尺寸（网格点数）
        filtered_mask = np.zeros_like(defect_mask)
        
        for i in range(1, num_features + 1):
            region = (labeled == i)
            if np.sum(region) >= min_size:
                filtered_mask |= region
                
                # 计算区域中心
                y_indices, x_indices = np.where(region)
                center_x = self.x[int(np.mean(x_indices))]
                center_y = self.y[int(np.mean(y_indices))]
                print(f"  缺陷 {i}: 中心 ({center_x:.3f}, {center_y:.3f}), 大小 {np.sum(region)} 点")
        
        print(f"\n保留 {np.sum(filtered_mask > 0)} 个有效缺陷区域")
        
        return filtered_mask, anomaly_smooth

class DefectQuantifier:
    """缺陷量化器（局部K值优化）"""
    
    def __init__(self, x, y, t, U_measured, defect_mask):
        """
        初始化
        
        参数:
            x, y: 空间网格
            t: 时间数组
            U_measured: 测量温度场
            defect_mask: 缺陷区域掩码
        """
        self.x = x
        self.y = y
        self.t = t
        self.U_measured = U_measured
        self.defect_mask = defect_mask
        
        self.X, self.Y = np.meshgrid(x, y, indexing='ij')
        
        # 创建FEM求解器
        dt = t[1] - t[0] if len(t) > 1 else 0.001
        self.solver = FEMSolver(x, y, dt=dt)
        
        print(f"\nDefectQuantifier初始化:")
        print(f"  缺陷区域: {np.sum(defect_mask)} 个点")
    
    def optimize_local_k(self, k_init=0.5, bounds=(0.1, 1.0), method='L-BFGS-B'):
        """
        局部优化缺陷区域的K值
        
        参数:
            k_init: 缺陷区域K的初始猜测
            bounds: K的界限
            method: 优化方法
        
        返回:
            K_optimal: 优化后的K场
            result: 优化结果
        """
        print("\n" + "="*70)
        print("阶段2: 缺陷量化（局部K值优化）")
        print("="*70)
        
        # 初始K场（非缺陷区域为1）
        K_field = np.ones_like(self.X)
        K_field[self.defect_mask] = k_init
        
        # 缺陷区域的平均K（优化变量）
        defect_indices = np.where(self.defect_mask)
        n_defect = len(defect_indices[0])
        
        print(f"\n优化参数:")
        print(f"  缺陷点数: {n_defect}")
        print(f"  初始K值: {k_init}")
        print(f"  K界限: {bounds}")
        
        # 定义目标函数
        def objective(k_defect):
            """目标函数：最小化测量与模拟的差异"""
            K_test = np.ones_like(self.X)
            K_test[self.defect_mask] = k_defect[0]
            
            # FEM求解
            U_sim = self.solver.solve_transient(K_test, self.t, verbose=False)
            
            # 计算误差（仅在关键区域）
            # 重点关注缺陷附近的温度差异
            weight = gaussian_filter(self.defect_mask.astype(float), sigma=3.0)
            weight = weight / (np.max(weight) + 1e-10)
            
            error = 0.0
            for t_idx in range(len(self.t)):
                diff = (U_sim[t_idx] - self.U_measured[t_idx]) * weight
                error += np.sum(diff**2)
            
            error /= len(self.t)
            
            return error
        
        # 优化
        print("\n开始优化...")
        result = minimize(
            objective,
            x0=[k_init],
            method=method,
            bounds=[bounds],
            options={'maxiter': 100, 'disp': True}
        )
        
        print("\n优化完成！")
        print(f"  最优K值: {result.x[0]:.4f}")
        print(f"  最终误差: {result.fun:.6f}")
        print(f"  迭代次数: {result.nit}")
        
        # 构建最优K场
        K_optimal = np.ones_like(self.X)
        K_optimal[self.defect_mask] = result.x[0]
        
        return K_optimal, result

def full_pipeline(data_path, save_dir='NDT_Data_V2', visualize=True):
    """
    完整的NDT流程
    
    参数:
        data_path: 数据文件路径
        save_dir: 保存目录
        visualize: 是否可视化
    
    返回:
        K_optimal: 优化后的K场
        defect_mask: 缺陷掩码
    """
    print("\n" + "="*70)
    print("NDT完整流程")
    print("="*70)
    
    # 加载数据
    print(f"\n加载数据: {data_path}")
    data = np.load(data_path)
    
    x = data['x']
    y = data['y']
    t = data['t']
    U_measured = data['U_measured']
    K_true = data['K_true']
    
    print(f"数据形状: U={U_measured.shape}, K={K_true.shape}")
    
    # 步骤0: 求解k=1基准
    print("\n步骤0: 求解K=1基准...")
    dt = t[1] - t[0]
    solver = FEMSolver(x, y, dt=dt)
    K_baseline = np.ones_like(solver.X)
    U_baseline = solver.solve_transient(K_baseline, t, verbose=True)
    
    # 步骤1: 缺陷定位
    localizer = DefectLocalizer(x, y, t, U_measured, threshold_percentile=92)
    defect_mask, anomaly_map = localizer.locate_defects(U_baseline)
    
    # 步骤2: 缺陷量化
    quantifier = DefectQuantifier(x, y, t, U_measured, defect_mask)
    K_optimal, opt_result = quantifier.optimize_local_k(k_init=0.25, bounds=(0.15, 0.8))
    
    # 验证
    print("\n" + "="*70)
    print("结果验证")
    print("="*70)
    
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # 真实缺陷区域
    true_defect = ((X - 0.35)**2 + (Y - 0.35)**2) < 0.15**2
    
    # IoU（交并比）
    intersection = np.sum(defect_mask & true_defect)
    union = np.sum(defect_mask | true_defect)
    iou = intersection / union if union > 0 else 0
    
    print(f"\n定位精度:")
    print(f"  IoU (交并比): {iou:.3f}")
    print(f"  检测到的缺陷点: {np.sum(defect_mask)}")
    print(f"  真实缺陷点: {np.sum(true_defect)}")
    
    # K值精度
    k_true_defect = K_true[true_defect].mean()
    k_pred_defect = K_optimal[defect_mask].mean()
    k_error = np.abs(k_true_defect - k_pred_defect)
    
    print(f"\nK值精度:")
    print(f"  真实K (缺陷): {k_true_defect:.4f}")
    print(f"  预测K (缺陷): {k_pred_defect:.4f}")
    print(f"  绝对误差: {k_error:.4f}")
    print(f"  相对误差: {k_error/k_true_defect*100:.2f}%")
    
    # 保存结果
    np.savez(
        os.path.join(save_dir, 'ndt_result_v2.npz'),
        K_optimal=K_optimal,
        defect_mask=defect_mask,
        anomaly_map=anomaly_map,
        opt_result=opt_result.x,
        iou=iou,
        k_error=k_error
    )
    print(f"\n结果已保存: {save_dir}/ndt_result_v2.npz")
    
    # 可视化
    if visualize:
        visualize_results(x, y, K_true, K_optimal, defect_mask, true_defect, 
                         anomaly_map, save_dir)
    
    return K_optimal, defect_mask

def visualize_results(x, y, K_true, K_pred, defect_mask, true_defect, 
                     anomaly_map, save_dir):
    """可视化结果"""
    print("\n生成可视化...")
    
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 第一行：K场对比
    im0 = axes[0, 0].imshow(K_true.T, origin='lower', extent=[0,1,0,1], 
                           cmap='RdYlBu_r', vmin=0.2, vmax=1.0)
    axes[0, 0].set_title('True K')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    plt.colorbar(im0, ax=axes[0, 0], label='K')
    
    im1 = axes[0, 1].imshow(K_pred.T, origin='lower', extent=[0,1,0,1], 
                           cmap='RdYlBu_r', vmin=0.2, vmax=1.0)
    axes[0, 1].set_title('Predicted K')
    axes[0, 1].set_xlabel('x')
    plt.colorbar(im1, ax=axes[0, 1], label='K')
    
    error = np.abs(K_true - K_pred)
    im2 = axes[0, 2].imshow(error.T, origin='lower', extent=[0,1,0,1], 
                           cmap='hot')
    axes[0, 2].set_title('Absolute Error')
    axes[0, 2].set_xlabel('x')
    plt.colorbar(im2, ax=axes[0, 2], label='|K_true - K_pred|')
    
    # 第二行：缺陷定位
    axes[1, 0].imshow(anomaly_map.T, origin='lower', extent=[0,1,0,1], 
                     cmap='hot')
    axes[1, 0].set_title('Anomaly Map')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')
    axes[1, 0].contour(X, Y, true_defect, colors='cyan', linewidths=2, levels=[0.5])
    
    axes[1, 1].imshow(defect_mask.T, origin='lower', extent=[0,1,0,1], 
                     cmap='binary')
    axes[1, 1].set_title('Detected Defect')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].contour(X, Y, true_defect, colors='red', linewidths=2, levels=[0.5])
    
    # 定位精度
    tp = defect_mask & true_defect
    fp = defect_mask & ~true_defect
    fn = ~defect_mask & true_defect
    
    vis_map = np.zeros_like(defect_mask, dtype=int)
    vis_map[tp] = 3  # True Positive
    vis_map[fp] = 2  # False Positive
    vis_map[fn] = 1  # False Negative
    
    cmap_custom = plt.matplotlib.colors.ListedColormap(['white', 'blue', 'red', 'green'])
    axes[1, 2].imshow(vis_map.T, origin='lower', extent=[0,1,0,1], cmap=cmap_custom)
    axes[1, 2].set_title('Localization Quality')
    axes[1, 2].set_xlabel('x')
    
    # 图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='True Positive'),
        Patch(facecolor='red', label='False Positive'),
        Patch(facecolor='blue', label='False Negative')
    ]
    axes[1, 2].legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ndt_results_v2.png'), dpi=150)
    plt.close()
    
    print(f"可视化已保存: {save_dir}/ndt_results_v2.png")

if __name__ == "__main__":
    data_path = 'NDT_Data_V2/ndt_data_full.npz'
    
    if os.path.exists(data_path):
        K_optimal, defect_mask = full_pipeline(data_path)
    else:
        print(f"错误: 数据文件不存在: {data_path}")
        print("请先运行: python ndt_data_v2.py")
