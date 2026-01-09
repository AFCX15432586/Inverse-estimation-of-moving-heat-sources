import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class NDTVisualizer:
    
    def __init__(self, data_dir='NDT_Data_V2'):
        self.data_dir = data_dir
        self.load_data()
    
    def load_data(self):

        print(f"loading data from {self.data_dir}...")
        
        # primitive solution
        data_file = os.path.join(self.data_dir, 'ndt_data_full.npz')
        if os.path.exists(data_file):
            self.data = np.load(data_file)
            print("Primitive data find")
        else:
            print("no Primitive data")
            self.data = None
        
        # Reference solution
        baseline_file = os.path.join(self.data_dir, 'baseline_solution.npz')
        if os.path.exists(baseline_file):
            self.baseline = np.load(baseline_file)
            print("Reference solution find")
        else:
            print("no Reference solution")
            self.baseline = None
        
        # testing result
        result_file = os.path.join(self.data_dir, 'ndt_result_v2.npz')
        if os.path.exists(result_file):
            self.result = np.load(result_file, allow_pickle=True)
            print("testing result output")
        else:
            print("no testing result")
            self.result = None
    
    def plot_overview(self):

        if self.data is None or self.result is None:
            print("no enough data, can't plot")
            return
        
        print("\ngenerate figure...")
        
        x = self.data['x']
        y = self.data['y']
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # First line: Data and baseline
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(self.data['K_true'].T, origin='lower', extent=[0,1,0,1], 
                        cmap='RdYlBu_r', vmin=0.2, vmax=1.0)
        ax1.set_title('True K Field', fontsize=12, fontweight='bold')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        circle = Circle((0.35, 0.35), 0.15, color='black', fill=False, linewidth=2)
        ax1.add_patch(circle)
        plt.colorbar(im1, ax=ax1, label='K')
        
        if self.baseline is not None:
            ax2 = fig.add_subplot(gs[0, 1])
            im2 = ax2.imshow(self.baseline['residual'][-1].T, origin='lower', 
                           extent=[0,1,0,1], cmap='RdBu_r')
            ax2.set_title('Temperature Residual (Last Time)', fontsize=12, fontweight='bold')
            ax2.set_xlabel('x')
            plt.colorbar(im2, ax=ax2)
        
        if self.baseline is not None:
            ax3 = fig.add_subplot(gs[0, 2])
            im3 = ax3.imshow(self.baseline['grad_diff'].T, origin='lower', 
                           extent=[0,1,0,1], cmap='hot')
            ax3.set_title('Gradient Difference', fontsize=12, fontweight='bold')
            ax3.set_xlabel('x')
            plt.colorbar(im3, ax=ax3)
        
        # Second line: Test results
        ax4 = fig.add_subplot(gs[1, 0])
        im4 = ax4.imshow(self.result['anomaly_map'].T, origin='lower', 
                        extent=[0,1,0,1], cmap='hot')
        ax4.set_title('Anomaly Map', fontsize=12, fontweight='bold')
        ax4.set_xlabel('x')
        ax4.set_ylabel('y')
        ax4.contour(X, Y, ((X-0.35)**2 + (Y-0.35)**2 < 0.15**2),
                   colors='cyan', linewidths=2, levels=[0.5])
        plt.colorbar(im4, ax=ax4)
        
        ax5 = fig.add_subplot(gs[1, 1])
        im5 = ax5.imshow(self.result['defect_mask'].T, origin='lower', 
                        extent=[0,1,0,1], cmap='binary')
        ax5.set_title('Detected Defect Mask', fontsize=12, fontweight='bold')
        ax5.set_xlabel('x')
        circle = Circle((0.35, 0.35), 0.15, color='red', fill=False, linewidth=2, label='True')
        ax5.add_patch(circle)
        ax5.legend()
        
        ax6 = fig.add_subplot(gs[1, 2])
        im6 = ax6.imshow(self.result['K_optimal'].T, origin='lower', 
                        extent=[0,1,0,1], cmap='RdYlBu_r', vmin=0.2, vmax=1.0)
        ax6.set_title('Predicted K Field', fontsize=12, fontweight='bold')
        ax6.set_xlabel('x')
        plt.colorbar(im6, ax=ax6, label='K')
        
        # Third line: Error analysis
        error = np.abs(self.data['K_true'] - self.result['K_optimal'])
        
        ax7 = fig.add_subplot(gs[2, 0])
        im7 = ax7.imshow(error.T, origin='lower', extent=[0,1,0,1], cmap='hot')
        ax7.set_title('Absolute Error', fontsize=12, fontweight='bold')
        ax7.set_xlabel('x')
        ax7.set_ylabel('y')
        plt.colorbar(im7, ax=ax7, label='|K_true - K_pred|')
        
        # statistical histogram
        ax8 = fig.add_subplot(gs[2, 1])
        ax8.hist(self.data['K_true'].flatten(), bins=30, alpha=0.7, label='True K', color='blue')
        ax8.hist(self.result['K_optimal'].flatten(), bins=30, alpha=0.7, label='Predicted K', color='red')
        ax8.set_xlabel('K value')
        ax8.set_ylabel('Frequency')
        ax8.set_title('K Distribution', fontsize=12, fontweight='bold')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        # performance index
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')
        
        metrics_text = f"""
Performance Metrics

IoU: {self.result['iou']:.3f}

K Error: {self.result['k_error']:.4f}

True K (defect): 
  {self.data['K_true'][((X-0.35)**2 + (Y-0.35)**2 < 0.15**2)].mean():.4f}

Pred K (defect): 
  {self.result['opt_result'][0]:.4f}

Error Stats:
  Mean: {np.mean(error):.4f}
  Max: {np.max(error):.4f}
  Std: {np.std(error):.4f}
"""
        ax9.text(0.1, 0.5, metrics_text, fontsize=11, verticalalignment='center',
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.savefig(os.path.join(self.data_dir, 'overview_dashboard.png'), 
                   dpi=150, bbox_inches='tight')
        print(f"Overview Dashboard saved: {self.data_dir}/overview_dashboard.png")
        plt.close()
    
    def plot_temporal_evolution(self):

        if self.data is None:
            print("no data!")
            return
        
        print("\nGenerate time evolution diagram...")
        
        t = self.data['t']
        U = self.data['U_measured']
        
        # Select several crucial moments
        time_indices = [0, len(t)//4, len(t)//2, 3*len(t)//4, -1]
        
        fig, axes = plt.subplots(1, len(time_indices), figsize=(20, 4))
        
        for idx, t_idx in enumerate(time_indices):
            im = axes[idx].imshow(U[t_idx].T, origin='lower', extent=[0,1,0,1], 
                                 cmap='hot', vmin=0, vmax=np.max(U))
            axes[idx].set_title(f't = {t[t_idx]:.3f}')
            axes[idx].set_xlabel('x')
            if idx == 0:
                axes[idx].set_ylabel('y')
            
            # Laser position
            if 'laser_cx' in self.data:
                cx = self.data['laser_cx'][t_idx]
                cy = self.data['laser_cy'][t_idx]
                axes[idx].plot(cx, cy, 'c*', markersize=15)
            
            # Defect location
            circle = Circle((0.35, 0.35), 0.15, color='cyan', fill=False, 
                          linewidth=2, linestyle='--')
            axes[idx].add_patch(circle)
            
            plt.colorbar(im, ax=axes[idx], label='T')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.data_dir, 'temporal_evolution.png'), 
                   dpi=150, bbox_inches='tight')
        print(f"The time evolution graph has been saved: {self.data_dir}/temporal_evolution.png")
        plt.close()
    
    def generate_report(self):

        if self.data is None or self.result is None:
            print("no data")
            return
        
        print("\nGenerate a detailed report...")
        
        x = self.data['x']
        y = self.data['y']
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        true_defect = ((X-0.35)**2 + (Y-0.35)**2 < 0.15**2)
        detected_defect = self.result['defect_mask']
        
        # Calculate various indicators
        tp = np.sum(detected_defect & true_defect)
        fp = np.sum(detected_defect & ~true_defect)
        fn = np.sum(~detected_defect & true_defect)
        tn = np.sum(~detected_defect & ~true_defect)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        k_true = self.data['K_true']
        k_pred = self.result['K_optimal']
        
        report = f"""
# NDT Detection Report

## System Information
- Method: FEM-Guided Two-Stage Detection
- Grid Size: {len(x)} x {len(y)}
- Time Snapshots: {len(self.data['t'])}

## Detection Performance

### Localization Accuracy
- **IoU**: {self.result['iou']:.4f}
- **Precision**: {precision:.4f} (TP/{tp+fp} = {tp}/{tp+fp})
- **Recall**: {recall:.4f} (TP/{tp+fn} = {tp}/{tp+fn})
- **F1 Score**: {f1:.4f}

### Confusion Matrix
|              | Predicted Defect | Predicted Normal |
|--------------|-----------------|------------------|
| True Defect  | {tp:6d} (TP)    | {fn:6d} (FN)    |
| True Normal  | {fp:6d} (FP)    | {tn:6d} (TN)    |

### K Value Accuracy
- **True K (defect region)**: {k_true[true_defect].mean():.4f}
- **Predicted K (detected region)**: {k_pred[detected_defect].mean():.4f}
- **Absolute Error**: {self.result['k_error']:.4f}
- **Relative Error**: {self.result['k_error']/k_true[true_defect].mean()*100:.2f}%

## Spatial Error Distribution
- **Mean Error**: {np.mean(np.abs(k_true - k_pred)):.4f}
- **Max Error**: {np.max(np.abs(k_true - k_pred)):.4f}
- **Std Error**: {np.std(np.abs(k_true - k_pred)):.4f}

## K Field Statistics

### True K Field
- Mean: {np.mean(k_true):.4f}
- Std: {np.std(k_true):.4f}
- Range: [{np.min(k_true):.4f}, {np.max(k_true):.4f}]

### Predicted K Field
- Mean: {np.mean(k_pred):.4f}
- Std: {np.std(k_pred):.4f}
- Range: [{np.min(k_pred):.4f}, {np.max(k_pred):.4f}]

## Conclusions

1. The FEM-guided method successfully detected the defect with IoU = {self.result['iou']:.3f}
2. K value prediction accuracy: error = {self.result['k_error']:.4f} ({self.result['k_error']/k_true[true_defect].mean()*100:.1f}%)
3. Localization performance: Precision = {precision:.3f}, Recall = {recall:.3f}

## Recommendations
{'- ✓ Detection successful! Results are reliable.' if self.result['iou'] > 0.8 else '- ! Consider adjusting threshold_percentile for better localization.'}
{'- ✓ K prediction is accurate!' if self.result['k_error'] < 0.05 else '- ! Consider running more optimization iterations.'}
"""
        
        with open(os.path.join(self.data_dir, 'detection_report.md'), 'w') as f:
            f.write(report)
        
        print(f"Report saved: {self.data_dir}/detection_report.md")
        print("\n" + "="*70)
        print(report)
        print("="*70)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='NDT Results Visualizer')
    parser.add_argument('--data-dir', type=str, default='NDT_Data_V2',
                       help='Data directory')
    parser.add_argument('--overview', action='store_true',
                       help='Generate overview dashboard')
    parser.add_argument('--temporal', action='store_true',
                       help='Generate temporal evolution plot')
    parser.add_argument('--report', action='store_true',
                       help='Generate detailed report')
    parser.add_argument('--all', action='store_true',
                       help='Generate all visualizations')
    
    args = parser.parse_args()
    
    # If no options are specified, all will be generated by default
    if not (args.overview or args.temporal or args.report or args.all):
        args.all = True
    
    visualizer = NDTVisualizer(args.data_dir)
    
    if args.all or args.overview:
        visualizer.plot_overview()
    
    if args.all or args.temporal:
        visualizer.plot_temporal_evolution()
    
    if args.all or args.report:
        visualizer.generate_report()
    
    print("\n✓ Visualization completed!")

if __name__ == "__main__":
    main()
