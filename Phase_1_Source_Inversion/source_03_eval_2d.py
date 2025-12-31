import numpy as np
import matplotlib.pyplot as plt
import os

def eval_2d_advanced(target_dir):
    print(f"正在分析 {target_dir} 下的数据...")
    
    # 1. 加载数据
    data = np.load(os.path.join(target_dir, 'source_data_2d.npz'))
    pred = np.load(os.path.join(target_dir, 'inverse_pred_2d.npz'))
    
    # 获取网格
    x, y = data['x'], data['y']
    
    # 2. 选取一个特定的时间快照进行评估 (比如 t=0.5s 时)
    # data['t'] 是时间数组，我们要找到索引
    idx_t = int(len(data['t']) * 0.5) 
    
    # --- 准备绘图数据 ---
    
    # A. 热源 f (我们最关心的反演目标)
    f_true = data['F_true'][idx_t]
    f_pred = pred['f_pred']
    f_err = np.abs(f_true - f_pred)
    
    # B. 温度 u (用于检查数据拟合情况)
    # 注意：我们的预测文件里之前没保存 u_pred，这里主要画真实温度作为参考
    # 如果你也想看 u 的拟合，需要在 source_02 里保存 u_pred
    u_true = data['U_true'][idx_t] 
    
    # --- 开始绘图 (2行3列) ---
    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    
    # 第一行：热源反演结果 (Inversion Result)
    # 1. True Source
    im1 = axs[0,0].contourf(x, y, f_true.T, levels=50, cmap='viridis')
    axs[0,0].set_title(f"Ground Truth Source (t={data['t'][idx_t]:.2f}s)")
    plt.colorbar(im1, ax=axs[0,0])
    
    # 2. Predicted Source
    im2 = axs[0,1].contourf(x, y, f_pred.T, levels=50, cmap='viridis')
    axs[0,1].set_title("PINN Predicted Source")
    plt.colorbar(im2, ax=axs[0,1])
    
    # 3. Source Error
    im3 = axs[0,2].contourf(x, y, f_err.T, levels=50, cmap='inferno')
    axs[0,2].set_title("Inversion Error |True - Pred|")
    plt.colorbar(im3, ax=axs[0,2])
    
    # 第二行：物理场背景 (Temperature Field)
    # 4. True Temp
    im4 = axs[1,0].contourf(x, y, u_true.T, levels=50, cmap='plasma')
    axs[1,0].set_title("Observed Temperature Field")
    plt.colorbar(im4, ax=axs[1,0])
    
    # 5. 3D Plot (更直观地看热源形状)
    axs[1,1].remove() # 移除原来的 2D 轴
    ax_3d = fig.add_subplot(2, 3, 5, projection='3d')
    X, Y = np.meshgrid(x, y, indexing='ij')
    ax_3d.plot_surface(X, Y, f_pred, cmap='viridis', linewidth=0, antialiased=False)
    ax_3d.set_title("3D View of Predicted Source")
    
    # 6. 统计信息
    axs[1,2].axis('off')
    mse = np.mean(f_err**2)
    max_err = np.max(f_err)
    text_str = (f"Analysis Metrics:\n\n"
                f"Time Snapshot: t={data['t'][idx_t]:.2f}s\n"
                f"Source MSE: {mse:.2e}\n"
                f"Max Error: {max_err:.4f}\n\n"
                f"Observation:\n"
                f"The rough location is correct,\n"
                f"but the shape is blurred.\n"
                f"This motivates the use of\n"
                f"Fourier Features in Phase 2.")
    axs[1,2].text(0.1, 0.5, text_str, fontsize=12, verticalalignment='center')

    plt.tight_layout()
    save_path = os.path.join(target_dir, "result_phase1_detailed.png")
    plt.savefig(save_path)
    print(f"✅ 详细评估报告已生成: {save_path}")

if __name__ == "__main__":
    eval_2d_advanced("Phase1_Data")