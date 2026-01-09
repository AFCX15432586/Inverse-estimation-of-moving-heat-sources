import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def eval_ndt_advanced(data_dir='NDT_Data_V2'):

    print(f"Loading data from: {data_dir} ...")
    
    # load data
    data_file = os.path.join(data_dir, 'ndt_data_full.npz')
    result_file = os.path.join(data_dir, 'ndt_result_v2.npz')
    
    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found!")
        print(f"Please run: python demo.py or python ndt_main.py --mode all")
        return
    
    if not os.path.exists(result_file):
        print(f"Error: {result_file} not found!")
        print(f"Please run: python demo.py or python ndt_main.py --mode all")
        return
    
    data = np.load(data_file)
    res = np.load(result_file, allow_pickle=True)

    k_true = data['K_true']
    k_pred = res['K_optimal']
    x, y = data['x'], data['y']
    
    print(f"✓ Data loaded: K_true={k_true.shape}, K_pred={k_pred.shape}")
    
    # ==================== Calculate evaluation indicators ====================
    
    # 1. MSE
    mse = np.mean((k_true - k_pred)**2)
    
    # 2. MAE
    mae = np.mean(np.abs(k_true - k_pred))
    
    # 3. Relative L2 Error
    rel_l2 = np.linalg.norm(k_true - k_pred) / np.linalg.norm(k_true)
    
    # 4. PSNR (Peak Signal-to-Noise Ratio)
    data_range = k_true.max() - k_true.min()
    val_psnr = psnr(k_true, k_pred, data_range=data_range)
    
    # 5. SSIM (Structural Similarity Index)
    val_ssim = ssim(k_true, k_pred, data_range=data_range)

    # 6. Precision of defect area
    X, Y = np.meshgrid(x, y, indexing='ij')

    defect_mask_circle = ((X - 0.35) ** 2 + (Y - 0.35) ** 2) < 0.15 ** 2
    defect_mask = defect_mask_circle

    if np.sum(defect_mask) > 0:
        k_true_defect = k_true[defect_mask].mean()
        k_pred_defect = k_pred[defect_mask].mean()
        print("Defect Area:", k_true_defect, k_pred_defect)
        defect_error = np.abs(k_true_defect - k_pred_defect)
        defect_rel_error = defect_error / k_true_defect
    else:
        k_true_defect = 0
        k_pred_defect = 0
        defect_error = 0
        defect_rel_error = 0
    
    # ==================== Generate report text ====================
    
    report_text = (
        f"=== NDT Diagnosis Report (V2) ===\n"
        f"Data Directory: {data_dir}\n"
        f"Domain Size: {k_true.shape[0]} x {k_true.shape[1]}\n"
        f"----------------------------------------\n"
        f"Global Accuracy Metrics:\n"
        f"1. MSE (Mean Squared Error)    : {mse:.6f}\n"
        f"2. MAE (Mean Absolute Error)   : {mae:.6f}\n"
        f"3. Relative L2 Error           : {rel_l2:.2%}\n"
        f"4. PSNR (Signal-to-Noise Ratio): {val_psnr:.2f} dB\n"
        f"5. SSIM (Structural Similarity): {val_ssim:.4f}\n"
        f"----------------------------------------\n"
        f"Defect Region Accuracy:\n"
        f"6. True K (defect)             : {k_true_defect:.4f}\n"
        f"7. Predicted K (defect)        : {k_pred_defect:.4f}\n"
        f"8. Defect Error                : {defect_error:.4f}\n"
        f"9. Defect Relative Error       : {defect_rel_error:.2%}\n"
        f"----------------------------------------\n"
        f"Diagnosis:\n"
    )
    
    # Diagnosis recommendation
    if val_ssim > 0.85 and rel_l2 < 0.05:
        report_text += "✓ EXCELLENT: High structural similarity and low error.\n"
    elif val_ssim > 0.70 and rel_l2 < 0.10:
        report_text += "✓ GOOD: Acceptable reconstruction quality.\n"
    elif val_ssim > 0.50 and rel_l2 < 0.20:
        report_text += "✗ FAIR: Moderate accuracy, consider parameter tuning.\n"
    else:
        report_text += "✗ POOR: Significant errors detected, check TUNING_GUIDE.md\n"
    
    report_text += (
        f"\nBenchmarks:\n"
        f"- SSIM > 0.85: Excellent structural reconstruction\n"
        f"- Rel L2 < 5%: High precision (typical in NDT)\n"
        f"- PSNR > 30dB: Very good signal quality\n"
        f"- Defect Error < 0.05: Accurate defect quantification\n"
    )
    
    print("\n" + report_text)

    report_path = os.path.join(data_dir, "ndt_metrics_advanced.txt")
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"✓ Report saved to: {report_path}")
    
    # ==================== visualize ====================
    
    fig = plt.figure(figsize=(18, 10))
    
    # 1. Ground Truth K
    ax1 = fig.add_subplot(2, 3, 1)
    im1 = ax1.contourf(x, y, k_true.T, levels=50, cmap='viridis')
    ax1.set_title("Ground Truth K Field", fontsize=12, fontweight='bold')
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    circle = Circle((0.35, 0.35), 0.15, color='red', fill=False, linewidth=2, linestyle='--')
    ax1.add_patch(circle)
    
    plt.colorbar(im1, ax=ax1, label='K')
    
    # 2. Predicted K
    ax2 = fig.add_subplot(2, 3, 2)
    im2 = ax2.contourf(x, y, k_pred.T, levels=50, cmap='viridis')
    ax2.set_title(f"Predicted K Field\n(SSIM: {val_ssim:.3f}, L2 Error: {rel_l2:.2%})", 
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")

    if 'defect_mask' in res:
        detected_mask = res['defect_mask']
        ax2.contour(X, Y, detected_mask, colors='cyan', linewidths=2, levels=[0.5])
    
    plt.colorbar(im2, ax=ax2, label='K')
    
    # 3. Absolute Error Map
    ax3 = fig.add_subplot(2, 3, 3)
    error_map = np.abs(k_true - k_pred)
    im3 = ax3.contourf(x, y, error_map.T, levels=50, cmap='hot')
    ax3.set_title(f"Absolute Error Map\n(Max: {error_map.max():.4f}, Mean: {mae:.4f})", 
                  fontsize=12, fontweight='bold')
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    plt.colorbar(im3, ax=ax3, label='|K_true - K_pred|')
    
    # 4. Section Analysis 1 (Abnormal Area, y = 0.35)
    y_idx = np.argmin(np.abs(y - 0.35))
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(x, k_true[:, y_idx], 'k-', linewidth=2.5, label='Ground Truth', alpha=0.8)
    ax4.plot(x, k_pred[:, y_idx], 'r--', linewidth=2, label='Prediction', alpha=0.8)
    ax4.axvline(0.35, color='blue', linestyle=':', alpha=0.5, label='Defect Center')
    ax4.fill_between(x, 0.2, 0.5, where=((x > 0.20) & (x < 0.50)), 
                     color='yellow', alpha=0.2, label='Defect Region')
    ax4.set_title(f"Cross-section at y={y[y_idx]:.2f} (Through Defect)", fontsize=11)
    ax4.set_xlabel("x location")
    ax4.set_ylabel("Conductivity K")
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0.1, 1.1])
    
    # 5. Section Analysis 2 (Normal Area, y = 0.70)
    y_idx2 = np.argmin(np.abs(y - 0.70))
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(x, k_true[:, y_idx2], 'k-', linewidth=2.5, label='Ground Truth', alpha=0.8)
    ax5.plot(x, k_pred[:, y_idx2], 'r--', linewidth=2, label='Prediction', alpha=0.8)
    ax5.set_title(f"Cross-section at y={y[y_idx2]:.2f} (Normal Region)", fontsize=11)
    ax5.set_xlabel("x location")
    ax5.set_ylabel("Conductivity K")
    ax5.legend(loc='best', fontsize=9)
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim([0.1, 1.1])
    
    # 6. Error Distribution
    ax6 = fig.add_subplot(2, 3, 6)
    error_flat = error_map.flatten()
    n, bins, patches = ax6.hist(error_flat, bins=50, color='purple', alpha=0.7, 
                                edgecolor='black', log=True)
    
    # statistical information
    error_median = np.median(error_flat)
    error_95th = np.percentile(error_flat, 95)
    
    ax6.axvline(error_median, color='green', linestyle='--', linewidth=2, 
                label=f'Median: {error_median:.4f}')
    ax6.axvline(error_95th, color='red', linestyle='--', linewidth=2, 
                label=f'95th: {error_95th:.4f}')
    
    ax6.set_title("Error Distribution (Log Scale)", fontsize=11)
    ax6.set_xlabel("Absolute Error")
    ax6.set_ylabel("Count (Log Scale)")
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    
    # Title
    quality = "EXCELLENT" if (val_ssim > 0.85 and rel_l2 < 0.05) else \
              "GOOD" if (val_ssim > 0.70 and rel_l2 < 0.10) else \
              "FAIR" if (val_ssim > 0.50 and rel_l2 < 0.20) else "POOR"
    
    plt.suptitle(
        f"AI-Driven NDT Diagnostic Report (FEM-Guided V2)\n"
        f"SSIM: {val_ssim:.4f} | Rel L2 Error: {rel_l2:.2%} | PSNR: {val_psnr:.1f}dB | "
        f"Quality: {quality}", 
        fontsize=16, fontweight='bold'
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # save image
    save_path = os.path.join(data_dir, "ndt_report_advanced.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Figure saved to: {save_path}")
    
    # ==================== Generate performance summary ====================
    
    print("\n" + "="*60)
    print("Performance Summary:")
    print("="*60)
    print(f"{'Metric':<30} {'Value':<15} {'Status':<10}")
    print("-"*60)
    
    def get_status(value, good, excellent):
        if value >= excellent:
            return "✓✓ Excellent"
        elif value >= good:
            return "✓ Good"
        else:
            return "✗ Fair"
    
    def get_status_inverse(value, good, excellent):
        if value <= excellent:
            return "✓✓ Excellent"
        elif value <= good:
            return "✓ Good"
        else:
            return "✗ Fair"
    
    print(f"{'SSIM':<30} {val_ssim:<15.4f} {get_status(val_ssim, 0.70, 0.85):<10}")
    print(f"{'Relative L2 Error':<30} {rel_l2:<15.2%} {get_status_inverse(rel_l2, 0.10, 0.05):<10}")
    print(f"{'PSNR (dB)':<30} {val_psnr:<15.2f} {get_status(val_psnr, 25, 30):<10}")
    print(f"{'MAE':<30} {mae:<15.6f} {get_status_inverse(mae, 0.05, 0.03):<10}")
    print(f"{'Defect Relative Error':<30} {defect_rel_error:<15.2%} {get_status_inverse(defect_rel_error, 0.15, 0.10):<10}")
    print("="*60)
    
    return {
        'mse': mse,
        'mae': mae,
        'rel_l2': rel_l2,
        'psnr': val_psnr,
        'ssim': val_ssim,
        'defect_error': defect_error,
        'defect_rel_error': defect_rel_error
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Advanced NDT Evaluation')
    parser.add_argument('--data-dir', type=str, default='NDT_Data_V2',
                       help='Data directory (default: NDT_Data_V2)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Advanced NDT Evaluation and Visualization")
    print("="*60)

    if not os.path.exists(args.data_dir):
        print(f"\nWarning: Directory {args.data_dir} not found!")
        print("Available directories:")
        for d in ['NDT_Data_V2', 'NDT_Demo', 'NDT_Comparison']:
            if os.path.exists(d):
                print(f"  ✓ {d}")

        if os.path.exists('NDT_Demo'):
            print(f"\nUsing NDT_Demo instead...")
            args.data_dir = 'NDT_Demo'
        else:
            print("\nPlease run: python demo.py")
            exit(1)

    metrics = eval_ndt_advanced(args.data_dir)
    
    print(f"\n✓ Evaluation complete!")
    print(f"✓ Check: {args.data_dir}/ndt_report_advanced.png")
    print(f"✓ Check: {args.data_dir}/ndt_metrics_advanced.txt")
