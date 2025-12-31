import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def eval_ndt_advanced(data_dir):
    print(f"正在生成详细诊断报告: {data_dir} ...")
    
    # 1. 加载数据
    data = np.load(os.path.join(data_dir, 'ndt_data.npz'))
    res = np.load(os.path.join(data_dir, 'ndt_result.npz'))
    
    k_true = data['K_true']
    k_pred = res['k_pred']
    x, y = data['x'], data['y']
    
    # 2. 计算核心数值指标
    # (1) MSE (均方误差)
    mse = np.mean((k_true - k_pred)**2)
    
    # (2) MAE (平均绝对误差)
    mae = np.mean(np.abs(k_true - k_pred))
    
    # (3) Relative L2 Error (相对误差 - 论文常用)
    rel_l2 = np.linalg.norm(k_true - k_pred) / np.linalg.norm(k_true)
    
    # (4) PSNR (峰值信噪比 - 衡量图像重建质量)
    data_range = k_true.max() - k_true.min()
    val_psnr = psnr(k_true, k_pred, data_range=data_range)
    
    # (5) SSIM (结构相似性 - 衡量缺陷形状是否准确)
    val_ssim = ssim(k_true, k_pred, data_range=data_range)
    
    # 3. 生成文本报告
    report_text = (
        f"=== NDT Diagnosis Report ===\n"
        f"Domain Size: {k_true.shape}\n"
        f"----------------------------\n"
        f"1. MSE (Mean Squared Error): {mse:.6f}\n"
        f"2. MAE (Mean Absolute Error): {mae:.6f}\n"
        f"3. Relative L2 Error       : {rel_l2:.2%}\n"
        f"4. PSNR (Signal-to-Noise)  : {val_psnr:.2f} dB\n"
        f"5. SSIM (Structural Sim.)  : {val_ssim:.4f}\n"
        f"----------------------------\n"
        f"Diagnosis: \n"
        f"SSIM > 0.85 indicates excellent structural reconstruction.\n"
        f"Rel L2 < 5% is typically considered 'High Precision' in NDT."
    )
    print(report_text)
    
    # 保存文本报告
    with open(os.path.join(data_dir, "ndt_metrics.txt"), "w") as f:
        f.write(report_text)

    # 4. 高级绘图 (2行3列)
    fig = plt.figure(figsize=(18, 10))
    
    # --- 第一行: 2D 热力图对比 ---
    ax1 = fig.add_subplot(2, 3, 1)
    im1 = ax1.contourf(x, y, k_true.T, levels=50, cmap='viridis')
    ax1.set_title("Ground Truth (Defects)")
    plt.colorbar(im1, ax=ax1)
    
    ax2 = fig.add_subplot(2, 3, 2)
    im2 = ax2.contourf(x, y, k_pred.T, levels=50, cmap='viridis')
    ax2.set_title(f"AI Prediction\n(SSIM: {val_ssim:.3f})")
    plt.colorbar(im2, ax=ax2)
    
    ax3 = fig.add_subplot(2, 3, 3)
    error_map = np.abs(k_true - k_pred)
    im3 = ax3.contourf(x, y, error_map.T, levels=50, cmap='inferno')
    ax3.set_title(f"Absolute Error Map\n(Max Err: {error_map.max():.4f})")
    plt.colorbar(im3, ax=ax3)
    
    # --- 第二行: 详细分析 ---
    
    # 图4: 1D 切片分析 (Cutline Analysis)
    # 选取穿过圆形缺陷中心的一条线 (比如 y=0.35 处)
    y_idx = int(len(y) * 0.35) 
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(x, k_true[:, y_idx], 'k--', linewidth=2, label='Truth')
    ax4.plot(x, k_pred[:, y_idx], 'r-', linewidth=2, alpha=0.8, label='Prediction')
    ax4.set_title(f"Cross-section Profile at y={y[y_idx]:.2f}")
    ax4.set_xlabel("x location")
    ax4.set_ylabel("Conductivity k")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 图5: 1D 切片分析 (穿过裂纹, y=0.7)
    y_idx2 = int(len(y) * 0.7)
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(x, k_true[:, y_idx2], 'k--', linewidth=2, label='Truth')
    ax5.plot(x, k_pred[:, y_idx2], 'r-', linewidth=2, alpha=0.8, label='Prediction')
    ax5.set_title(f"Cross-section Profile at y={y[y_idx2]:.2f}")
    ax5.set_xlabel("x location")
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 图6: 误差分布直方图 (Error Histogram)
    ax6 = fig.add_subplot(2, 3, 6)
    n, bins, patches = ax6.hist(error_map.flatten(), bins=50, color='purple', alpha=0.7, log=True)
    ax6.set_title("Error Distribution (Log Scale)")
    ax6.set_xlabel("Absolute Error")
    ax6.set_ylabel("Count (Pixels)")
    
    plt.suptitle(f"AI-Driven NDT Diagnostic Report\nRelative L2 Error: {rel_l2:.2%}", fontsize=16)
    plt.tight_layout()
    
    save_path = os.path.join(data_dir, "ndt_report_advanced.png")
    plt.savefig(save_path)
    print(f"✅ 图表报告已保存: {save_path}")
    print(f"✅ 文本数据已保存: {os.path.join(data_dir, 'ndt_metrics.txt')}")

if __name__ == "__main__":
    eval_ndt_advanced("NDT_Data")